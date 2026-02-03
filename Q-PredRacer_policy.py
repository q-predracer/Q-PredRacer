import json
import logging
import os
import pickle
import random
from abc import abstractmethod
from .input_event import KeyEvent, IntentEvent, KillAppEvent, ExitEvent, SpawnEvent, SetTextEvent, ScrollEvent, \
    SwipeEvent, SelectEvent, LongTouchEvent, TouchEvent
from .utg import UTG

MAX_NUM_RESTARTS = 5
MAX_NUM_STEPS_OUTSIDE = 5
MAX_NUM_STEPS_OUTSIDE_KILL = 10
MAX_REPLY_TRIES = 5
EVENT_FLAG_STARTED = "+started"
EVENT_FLAG_START_APP = "+start_app"
EVENT_FLAG_STOP_APP = "+stop_app"
EVENT_FLAG_EXPLORE = "+explore"
EVENT_FLAG_NAVIGATE = "+navigate"
EVENT_FLAG_TOUCH = "+touch"
POLICY_NAIVE_DFS = "dfs_naive"
POLICY_GREEDY_DFS = "dfs_greedy"
POLICY_NAIVE_BFS = "bfs_naive"
POLICY_GREEDY_BFS = "bfs_greedy"
POLICY_REPLAY = "replay"
POLICY_MANUAL = "manual"
POLICY_MONKEY = "monkey"
POLICY_NONE = "none"
POLICY_MEMORY_GUIDED = "memory_guided"
POLICY_LLM_GUIDED = "llm_guided"
POLICY_Q_LEARNING = "q_learning"

KEY_KeyEvent = "key"
KEY_ManualEvent = "manual"
KEY_ExitEvent = "exit"
KEY_TouchEvent = "touch"
KEY_LongTouchEvent = "long_touch"
KEY_SelectEvent = "select"
KEY_UnselectEvent = "unselect"
KEY_SwipeEvent = "swipe"
KEY_ScrollEvent = "scroll"
KEY_SetTextEvent = "set_text"
KEY_IntentEvent = "intent"
KEY_SpawnEvent = "spawn"
KEY_KillAppEvent = "kill_app"

class InputInterruptedException(Exception):
    pass

class InputPolicy(object):
    def __init__(self, device, app, load_model_path="droidbot/model/"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.app = app
        self.my_structed = set()
        self.action_count = 0
        self.master = None
        self.count = 1
        self.q1_table = {}
        self.q2_table = {}
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = None
        self.last_log_index = 0
        self.log_file_path = "output/fourth/test/com.cashtoutiao2/logcat.txt"
        self.selected_events = []
        self.delta = 500
        self.my_structed_para1 = 1000
        self.reward = 1000
        self.w1_concurrency = 0.5
        self.w2_new_index = 0.55
        self.load_model_path = load_model_path
        self.lock = True

        if load_model_path is None:
            self.q1_table = {}
            self.q2_table = {}
        else:
            try:
                print("Attempting to load model...")
                self.load_model(self.load_model_path + app.get_package_name() + "_momodel.pkl")
                print("Model loaded successfully")
            except FileNotFoundError:
                print("Model file not found, initializing q_table as empty dictionary")
                self.q1_table = {}
                self.q2_table = {}
            except Exception as e:
                print(f"Error occurred while loading model: {e}")
                self.q1_table = {}
                self.q2_table = {}

    def transfer_index(self,event):
        if event.event_type == "kill_app":
            return event.stop_intent
        elif event.event_type == "intent":
            return event.intent
        elif event.event_type == "key":
            return event.name
        else:
            return event.get_views()[0]["view_str"]

    def save_RLmodel(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump((self.q1_table, self.q2_table), file)

    def load_model(self, file_path):
        with open(file_path, 'rb') as file:
            self.q1_table, self.q2_table = pickle.load(file)

    def _get_reward(self):
        if not os.path.exists(self.log_file_path):
            print("Log file not found!")
            return 0
        try:
            with open(self.log_file_path, "r", encoding='utf-8') as file:
                lines = file.readlines()
                total_lines = len(lines)
                if lines:
                    start_index = self.last_log_index + 1 if self.last_log_index < total_lines else total_lines
                    for log in lines[start_index - 1:]:
                        if "LoggingAspect" in log:
                            return 1
                return 0
        except FileNotFoundError:
            print("Log file not found!")
            return 0
        finally:
            self.last_log_index = total_lines if total_lines else 0

    def start(self, input_manager):
        self.action_count = 0
        self.epsilon = input_manager.epsilon
        while input_manager.enabled and self.action_count < input_manager.event_count:
            try:
                if self.action_count == 0 and self.master is None:
                    kill_event = KillAppEvent(app=self.app)
                    event = kill_event
                else:
                    print(self.epsilon)
                    event = self.generate_event()
                current_state = self.device.get_current_state()
                state_str = current_state.state_str
                old_state_str = state_str
                possible_events = current_state.get_possible_input()

                if state_str not in self.q1_table:
                  try:
                    self.q1_table[state_str] = {
                        self.transfer_index(event): {'event': event, 'q_value': 0} for event in possible_events}

                  except:
                    self.q1_table[state_str] = {
                        self.transfer_index(event): {'event': event, 'q_value': 0} for event in
                        possible_events}
                if state_str not in self.q2_table:

                    try:
                        self.q2_table[state_str] = {
                            self.transfer_index(event): {'event': event, 'q_value': 0} for event in possible_events}

                    except:
                        self.q2_table[state_str] = {
                            self.transfer_index(event): {'event': event, 'q_value': 0} for event in possible_events}

                if event == kill_event:

                    self.q1_table[old_state_str][ self.transfer_index(event)] = {'event': event, 'q_value': 0}

                    self.q2_table[old_state_str][ self.transfer_index(event)] = {'event': event, 'q_value': 0}

                print("第" + str(self.action_count + 1) + "次执行")

                input_manager.add_event(event)

                if self.lock == True:

                    if self.utg.is_event_explored(event=event, state=current_state):
                        print("重复执行")
                    if self.transfer_index(event) not in self.q1_table[old_state_str]:
                        self.q1_table[old_state_str][self.transfer_index(event)] = {
                            'event': event, 'q_value': 0}
                    if self.transfer_index(event) not in self.q2_table[old_state_str]:
                        self.q2_table[old_state_str][self.transfer_index(event)] = {
                            'event': event, 'q_value': 0}

                    old_q1_value = self.q1_table[old_state_str][self.transfer_index(event)][
                        'q_value']
                    old_q2_value = self.q2_table[old_state_str][self.transfer_index(event)][
                        'q_value']

                    current_state = self.device.get_current_state()
                    state_str = current_state.state_str
                    possible_events = current_state.get_possible_input()

                    if state_str not in self.q1_table:
                        try:
                            self.q1_table[state_str] = {
                                self.transfer_index(event): {'event': event, 'q_value': 0} for event in possible_events}
                        except:
                            self.q1_table[state_str] = {
                                self.transfer_index(event): {'event': event,'q_value': 0} for event in
                                possible_events}

                    if state_str not in self.q2_table:

                        try:
                            self.q2_table[state_str] = {
                                self.transfer_index(event): {'event': event, 'q_value': 0} for event in
                                possible_events}
                        except:
                            self.q2_table[state_str] = {
                                self.transfer_index(event): {'event': event, 'q_value': 0} for event in
                                possible_events}

                    max_future_q1_value = \
                        max(self.q1_table[state_str].values(), key=lambda event_data: event_data['q_value'])['q_value']
                    max_future_q2_value = \
                        max(self.q2_table[state_str].values(), key=lambda event_data: event_data['q_value'])['q_value']
                    long_my_structed = len(self.my_structed)
                    self.my_structed.add(state_str)

                    concurrency_reward = 0
                    new_index_reward = 0
                    error = 0

                    if self._get_reward() == 1:
                        concurrency_reward = self.reward
                        print("并发操作")

                    if long_my_structed < len(self.my_structed):
                        new_index_reward = self.my_structed_para1

                    if concurrency_reward == 0 and new_index_reward == 0:
                        error = self.delta

                    combined_reward = self.w1_concurrency * concurrency_reward + self.w2_new_index * new_index_reward - error

                    if random.random() < 0.5:
                        try:
                            self.q1_table[old_state_str][self.transfer_index(event)][
                                'q_value'] = old_q1_value + self.alpha * (
                                    combined_reward + self.gamma * max_future_q2_value - old_q1_value)
                        except:
                            self.q1_table[old_state_str][ self.transfer_index(event)][
                                'q_value'] = old_q1_value + self.alpha * (
                                    combined_reward + self.gamma * max_future_q2_value - old_q1_value)
                    else:
                        try:
                            self.q2_table[old_state_str][self.transfer_index(event)][
                                'q_value'] = old_q2_value + self.alpha * (
                                    combined_reward + self.gamma * max_future_q1_value - old_q2_value)
                        except:
                            self.q2_table[old_state_str][ self.transfer_index(event)][
                                'q_value'] = old_q2_value + self.alpha * (
                                    combined_reward + self.gamma * max_future_q1_value - old_q2_value)

            except KeyboardInterrupt:
                break
            except InputInterruptedException as e:
                self.logger.warning("stop sending events: %s" % e)
                break
            except Exception as e:
                self.logger.warning("exception during sending events: %s" % e)
                import traceback
                traceback.print_exc()
                continue
            self.action_count += 1
        self.save_RLmodel(self.load_model_path + self.app.get_package_name() + "_momodel.pkl")

    @abstractmethod
    def generate_event(self):
        pass

class NoneInputPolicy(InputPolicy):
    def __init__(self, device, app):
        super(NoneInputPolicy, self).__init__(device, app)

    def generate_event(self):
        return None

class UtgBasedInputPolicy(InputPolicy):
    def __init__(self, device, app, random_input):
        super(UtgBasedInputPolicy, self).__init__(device, app)
        self.random_input = random_input
        self.script = None
        self.master = None
        self.script_events = []
        self.last_event = None
        self.last_state = None
        self.current_state = None
        self.utg = UTG(device=device, app=app, random_input=random_input)
        self.script_event_idx = 0
        if self.device.humanoid is not None:
            self.humanoid_view_trees = []
            self.humanoid_events = []

    def generate_event(self):
        self.current_state = self.device.get_current_state()
        if self.current_state is None:
            import time
            time.sleep(5)
            return KeyEvent(name="BACK")
        self.__update_utg()
        if self.device.humanoid is not None:
            self.humanoid_view_trees = self.humanoid_view_trees + [self.current_state.view_tree]
            if len(self.humanoid_view_trees) > 4:
                self.humanoid_view_trees = self.humanoid_view_trees[1:]
        event = None
        if len(self.script_events) > self.script_event_idx:
            event = self.script_events[self.script_event_idx].get_transformed_event(self)
            self.script_event_idx += 1
        if event is None and self.script is not None:
            operation = self.script.get_operation_based_on_state(self.current_state)
            if operation is not None:
                self.script_events = operation.events
                event = self.script_events[0].get_transformed_event(self)
                self.script_event_idx = 1
        if event is None:
            event = self.generate_event_based_on_utg()
        if self.device.humanoid is not None:
            self.humanoid_events = self.humanoid_events + [event]
            if len(self.humanoid_events) > 3:
                self.humanoid_events = self.humanoid_events[1:]
        self.last_state = self.current_state
        self.last_event = event
        return event

    def __update_utg(self):
        self.utg.add_transition(self.last_event, self.last_state, self.current_state)

    @abstractmethod
    def generate_event_based_on_utg(self):
        pass

class UtgGreedySearchPolicy(UtgBasedInputPolicy):
    def __init__(self, device, app, random_input, search_method):
        super(UtgGreedySearchPolicy, self).__init__(device, app, random_input)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.search_method = search_method
        self.preferred_buttons = ["yes", "ok", "activate", "detail", "more", "access",
                                  "allow", "check", "agree", "try", "go", "next"]
        self.__nav_target = None
        self.__nav_num_steps = -1
        self.__num_restarts = 0
        self.__num_steps_outside = 0
        self.__event_trace = ""
        self.__missed_states = set()
        self.__random_explore = False

    def generate_event_based_on_utg(self):
        current_state = self.current_state
        self.logger.info("Current state: %s" % current_state.state_str)
        if current_state.state_str in self.__missed_states:
            self.__missed_states.remove(current_state.state_str)
        if current_state.get_app_activity_depth(self.app) < 0:
            start_app_intent = self.app.get_start_intent()
            if self.__event_trace.endswith(EVENT_FLAG_START_APP + EVENT_FLAG_STOP_APP) \
                    or self.__event_trace.endswith(EVENT_FLAG_START_APP):
                self.__num_restarts += 1
                self.logger.info("The app had been restarted %d times.", self.__num_restarts)
            else:
                self.__num_restarts = 0
            if not self.__event_trace.endswith(EVENT_FLAG_START_APP):
                if self.__num_restarts > MAX_NUM_RESTARTS:
                    msg = "The app had been restarted too many times. Entering random mode."
                    self.logger.info(msg)
                    self.__random_explore = True
                else:
                    self.__event_trace += EVENT_FLAG_START_APP
                    self.logger.info("Trying to start the app...")
                    return IntentEvent(intent=start_app_intent)
        elif current_state.get_app_activity_depth(self.app) > 0:
            self.__num_steps_outside += 1
            if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE:
                if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE_KILL:
                    stop_app_intent = self.app.get_stop_intent()
                    go_back_event = IntentEvent(stop_app_intent)
                else:
                    go_back_event = KeyEvent(name="BACK")
                self.__event_trace += EVENT_FLAG_NAVIGATE
                self.logger.info("Going back to the app...")
                return go_back_event
        else:
            self.__num_steps_outside = 0
        possible_events = current_state.get_possible_input()
        if self.random_input:
            random.shuffle(possible_events)
        if self.search_method == POLICY_GREEDY_DFS:
            possible_events.append(KeyEvent(name="BACK"))
        elif self.search_method == POLICY_GREEDY_BFS:
            possible_events.insert(0, KeyEvent(name="BACK"))
        for input_event in possible_events:
            if not self.utg.is_event_explored(event=input_event, state=current_state):
                self.logger.info("Trying an unexplored event.")
                self.__event_trace += EVENT_FLAG_EXPLORE
                return input_event
        target_state = self.__get_nav_target(current_state)
        if target_state:
            navigation_steps = self.utg.get_navigation_steps(from_state=current_state, to_state=target_state)
            if navigation_steps and len(navigation_steps) > 0:
                self.logger.info("Navigating to %s, %d steps left." % (target_state.state_str, len(navigation_steps)))
                self.__event_trace += EVENT_FLAG_NAVIGATE
                return navigation_steps[0][1]
        if self.__random_explore:
            self.logger.info("Trying random event.")
            random.shuffle(possible_events)
            return possible_events[0]
        stop_app_intent = self.app.get_stop_intent()
        self.logger.info("Cannot find an exploration target. Trying to restart app...")
        self.__event_trace += EVENT_FLAG_STOP_APP
        return IntentEvent(intent=stop_app_intent)

    def __get_nav_target(self, current_state):
        if self.__nav_target and self.__event_trace.endswith(EVENT_FLAG_NAVIGATE):
            navigation_steps = self.utg.get_navigation_steps(from_state=current_state, to_state=self.__nav_target)
            if navigation_steps and 0 < len(navigation_steps) <= self.__nav_num_steps:
                self.__nav_num_steps = len(navigation_steps)
                return self.__nav_target
            else:
                self.__missed_states.add(self.__nav_target.state_str)
        reachable_states = self.utg.get_reachable_states(current_state)
        if self.random_input:
            random.shuffle(reachable_states)
        for state in reachable_states:
            if state.get_app_activity_depth(self.app) != 0:
                continue
            if state.state_str in self.__missed_states:
                continue
            if self.utg.is_state_explored(state):
                continue
            self.__nav_target = state
            navigation_steps = self.utg.get_navigation_steps(from_state=current_state, to_state=self.__nav_target)
            if len(navigation_steps) > 0:
                self.__nav_num_steps = len(navigation_steps)
                return state
        self.__nav_target = None
        self.__nav_num_steps = -1
        return None

class RR2UtgGreedySearchPolicy(UtgBasedInputPolicy):
    def __init__(self, device, app, random_input, search_method=None, load_model_path=None):
        super(RR2UtgGreedySearchPolicy, self).__init__(device, app, random_input)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.search_method = "DFSG"
        self.preferred_buttons = ["yes", "ok", "activate", "detail", "more", "access",
                                  "allow", "check", "agree", "try", "go", "next"]

        self.__nav_target = None
        self.__nav_num_steps = -1
        self.__num_restarts = 0
        self.__num_steps_outside = 0

        self.__event_trace = ""
        self.__missed_states = set()
        self.__random_explore = False

    def generate_event_based_on_utg(self):
        current_state = self.device.get_current_state()
        possible_events = current_state.get_possible_input()
        state_str = current_state.state_str

        if state_str not in self.q1_table:
            self.q1_table[state_str] = {
                self.transfer_index(event): {'event': event, 'q_value': 0} for event in possible_events}
            self.q2_table[state_str] = {
                self.transfer_index(event): {'event': event, 'q_value': 0} for event in possible_events}

        self.logger.info("Current state: %s" % current_state.state_str)
        if current_state.state_str in self.__missed_states:
            self.__missed_states.remove(current_state.state_str)
        if current_state.get_app_activity_depth(self.app) < 0:
            start_app_intent = self.app.get_start_intent()
            if self.__event_trace.endswith(EVENT_FLAG_START_APP + EVENT_FLAG_STOP_APP) \
                    or self.__event_trace.endswith(EVENT_FLAG_START_APP):
                self.__num_restarts += 1
                self.logger.info("The app had been restarted %d times.", self.__num_restarts)
            else:
                self.__num_restarts = 0
            if not self.__event_trace.endswith(EVENT_FLAG_START_APP):
                if self.__num_restarts > MAX_NUM_RESTARTS:
                    msg = "The app had been restarted too many times. Entering random mode."
                    self.logger.info(msg)
                    self.__random_explore = True
                else:
                    self.__event_trace += EVENT_FLAG_START_APP
                    self.logger.info("Trying to start the app...")
                    current_state = self.device.get_current_state()
                    state_str = current_state.state_str
                    my_intent = IntentEvent(intent=start_app_intent)
                    self.q1_table[state_str][self.transfer_index(my_intent)] = {
                        'event': my_intent, 'q_value': 0}
                    self.q2_table[state_str][self.transfer_index(my_intent)] = {
                        'event': my_intent, 'q_value': 0}
                    return my_intent
        elif current_state.get_app_activity_depth(self.app) > 0:
            self.__num_steps_outside += 1
            if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE:
                if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE_KILL:
                    stop_app_intent = self.app.get_stop_intent()
                    go_back_event = IntentEvent(stop_app_intent)
                else:
                    go_back_event = KeyEvent(name="BACK")
                self.__event_trace += EVENT_FLAG_NAVIGATE
                self.logger.info("Going back to the app...")
                return go_back_event
        else:
            self.__num_steps_outside = 0
        if self.random_input:
            random.shuffle(possible_events)
        if self.search_method == "DFSG":
            key_event = KeyEvent(name="BACK")
            possible_events.append(key_event)
            self.q1_table[state_str][self.transfer_index(key_event)] = {
                'event': key_event, 'q_value': 0}
            self.q2_table[state_str][self.transfer_index(key_event)] = {
                'event': key_event, 'q_value': 0}
        elif self.search_method == "BFSG":
            key_event = KeyEvent(name="BACK")
            possible_events.insert(0, key_event)
            self.q1_table[state_str][self.transfer_index(key_event)] = {
                'event': key_event, 'q_value': 0}
            self.q2_table[state_str][self.transfer_index(key_event)] = {
                'event': key_event, 'q_value': 0}
        if random.uniform(0, 1) < self.epsilon:
            print("random")
            temp_input_event = None
            for input_event in possible_events:
                if not self.utg.is_event_explored(event=input_event, state=current_state):
                    temp_input_event = input_event
                    self.__event_trace += EVENT_FLAG_EXPLORE
                    self.logger.info("Trying an unexplored event.random")
                    return temp_input_event

            if temp_input_event == None:
                self.logger.info("Trying an back event.random")
                self.__event_trace += EVENT_FLAG_NAVIGATE
                return random.choice(possible_events)

        else:

            combined_q_values = {}

            for event_str in self.q1_table[state_str]:

                q1_value = self.q1_table[state_str][event_str]['q_value']
                q2_value = self.q2_table[state_str][event_str]['q_value']

                combined_q_value = (q1_value + q2_value) / 2

                combined_q_values[event_str] = {
                    "q_value": combined_q_value,
                    "event": self.q1_table[state_str][event_str]["event"]
                    }

            max_q_value = max(combined_q_values.values(), key=lambda x: x["q_value"])["q_value"]

            max_events = [event for event, q_value in combined_q_values.items() if q_value["q_value"] == max_q_value]

            editable_event = None
            for event in possible_events:
                if event.event_type == "set_text" and not self.utg.is_event_explored(event=event, state=current_state):
                    editable_event = event

                    break
            if editable_event is not None:
                selected_event = editable_event
            else:
                selected_event = None

                for event in max_events:

                    if self.q1_table[state_str][event]["event"] == event:
                        for event1 in possible_events:
                            self.q1_table[state_str][event]["event"] = event1
                    if not self.utg.is_event_explored(event=self.q1_table[state_str][event]["event"], state=current_state):
                        selected_event = self.q1_table[state_str][event]["event"]
                        break
                selected_random_event = None
                if selected_event == None:
                    possible_events_set = set()
                    for event in possible_events:
                        possible_events_set.add(self.transfer_index(event))
                        if self.transfer_index(event) not in self.q1_table[state_str]:
                            self.q1_table[state_str] = {
                                self.transfer_index(event): {'event': event, 'q_value': 0}}
                            self.q2_table[state_str] = {
                                self.transfer_index(event): {'event': event, 'q_value': 0}}
                    max_events_set = set(max_events)
                    difference_set = possible_events_set - max_events_set

                    average_q_values = {}
                    event_q_value_list = []
                    for event_key in difference_set:
                        if event_key not in self.q1_table[state_str]:
                            for event in possible_events:
                                if self.transfer_index(event) == event_key:
                                    self.q1_table[state_str] = {
                                        self.transfer_index(event): {'event': event, 'q_value': 0}}
                                    self.q2_table[state_str] = {
                                        self.transfer_index(event): {'event': event, 'q_value': 0}}

                        average_q_value = (self.q1_table[state_str][event_key]['q_value'] +
                                           self.q2_table[state_str][event_key]['q_value']) / 2
                        average_q_values[event_key] = average_q_value

                        event_q_value_list.append((event_key, average_q_value))

                    sorted_event_q_value_list = sorted(event_q_value_list, key=lambda x: x[1], reverse=True)

                    sorted_difference_list = []
                    for event, _ in sorted_event_q_value_list:
                        try:
                          sorted_difference_list.append(self.q1_table[state_str][event]["event"])
                        except:
                            for event_temp in possible_events:
                                if self.transfer_index(event_temp) == event:
                                    self.q1_table[state_str] = {
                                        self.transfer_index(event_temp): {'event': event, 'q_value': 0}}
                                    self.q2_table[state_str] = {
                                        self.transfer_index(event_temp): {'event': event, 'q_value': 0}}
                                    sorted_difference_list.append(self.q1_table[state_str][event]["event"])

                    if len(sorted_difference_list) > 0 and type(sorted_difference_list[0] ) is str:
                        temp = []
                        for str1 in sorted_difference_list:
                            for event in possible_events:
                                if str1 == self.transfer_index(event):
                                    temp.append(event)
                        sorted_difference_list = temp

                    for event in sorted_difference_list:
                        if not self.utg.is_event_explored(event = event, state = current_state):
                            selected_random_event = event
                            selected_event = selected_random_event
                            break

                    if selected_random_event == None:
                        selected_event = KeyEvent(name="BACK")

            if selected_event.event_type == KeyEvent(name="BACK").event_type:
                self.__event_trace += EVENT_FLAG_NAVIGATE
            else:
                self.__event_trace += EVENT_FLAG_EXPLORE

            return selected_event
        target_state = self.__get_nav_target(current_state)
        if target_state:
            navigation_steps = self.utg.get_navigation_steps(from_state=current_state, to_state=target_state)
            if navigation_steps and len(navigation_steps) > 0:
                self.logger.info("Navigating to %s, %d steps left." % (target_state.state_str, len(navigation_steps)))
                self.__event_trace += EVENT_FLAG_NAVIGATE
                return navigation_steps[0][1]
        stop_app_intent = self.app.get_stop_intent()
        self.logger.info("Cannot find an exploration target. Trying to restart app...")
        self.__event_trace += EVENT_FLAG_STOP_APP
        return IntentEvent(intent=stop_app_intent)

        return event_real
    def __get_nav_target(self, current_state):
        if self.__nav_target and self.__event_trace.endswith(EVENT_FLAG_NAVIGATE):
            navigation_steps = self.utg.get_navigation_steps(from_state=current_state, to_state=self.__nav_target)
            if navigation_steps and 0 < len(navigation_steps) <= self.__nav_num_steps:
                self.__nav_num_steps = len(navigation_steps)
                return self.__nav_target
            else:
                self.__missed_states.add(self.__nav_target.state_str)
        reachable_states = self.utg.get_reachable_states(current_state)
        if self.random_input:
            random.shuffle(reachable_states)
        for state in reachable_states:
            if state.get_app_activity_depth(self.app) != 0:
                continue
            if state.state_str in self.__missed_states:
                continue
            if self.utg.is_state_explored(state):
                continue
            self.__nav_target = state
            navigation_steps = self.utg.get_navigation_steps(from_state=current_state, to_state=self.__nav_target)
            if len(navigation_steps) > 0:
                self.__nav_num_steps = len(navigation_steps)
                return state
        self.__nav_target = None
        self.__nav_num_steps = -1
        return None

    def log_execution(self, action, reward):
        with open("droidbot/model/log3", 'a', encoding='utf-8') as file:
            file.write("Executed action: {}, Reward: {}\n".format(action, reward))

class UtgNaiveSearchPolicy(UtgBasedInputPolicy):
    
    def __init__(self, device, app, random_input, search_method):
        super(UtgNaiveSearchPolicy, self).__init__(device, app, random_input)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.explored_views = set()
        self.state_transitions = set()
        self.search_method = search_method

        self.last_event_flag = ""
        self.last_event_str = None
        self.last_state = None

        self.preferred_buttons = ["yes", "ok", "activate", "detail", "more", "access",
                                  "allow", "check", "agree", "try", "go", "next"]

    def generate_event_based_on_utg(self):
        
        self.save_state_transition(self.last_event_str, self.last_state, self.current_state)

        if self.device.is_foreground(self.app):

            self.last_event_flag = EVENT_FLAG_STARTED
        else:
            number_of_starts = self.last_event_flag.count(EVENT_FLAG_START_APP)

            if number_of_starts > MAX_NUM_RESTARTS:
                raise InputInterruptedException("The app cannot be started.")

            if self.last_event_flag.endswith(EVENT_FLAG_START_APP):

                self.logger.info("The app had been restarted %d times.", number_of_starts)
                self.logger.info("Trying to restart app...")
                pass
            else:
                start_app_intent = self.app.get_start_intent()

                self.last_event_flag += EVENT_FLAG_START_APP
                self.last_event_str = EVENT_FLAG_START_APP
                return IntentEvent(start_app_intent)

        view_to_touch = self.select_a_view(self.current_state)

        if view_to_touch is None:
            stop_app_intent = self.app.get_stop_intent()
            self.last_event_flag += EVENT_FLAG_STOP_APP
            self.last_event_str = EVENT_FLAG_STOP_APP
            return IntentEvent(stop_app_intent)

        view_to_touch_str = view_to_touch['view_str']
        if view_to_touch_str.startswith('BACK'):
            result = KeyEvent('BACK')
        else:
            result = TouchEvent(view=view_to_touch)

        self.last_event_flag += EVENT_FLAG_TOUCH
        self.last_event_str = view_to_touch_str
        self.save_explored_view(self.current_state, self.last_event_str)
        return result

    def select_a_view(self, state):
        
        views = []
        for view in state.views:
            if view['enabled'] and len(view['children']) == 0:
                views.append(view)

        if self.random_input:
            random.shuffle(views)

        mock_view_back = {'view_str': 'BACK_%s' % state.foreground_activity,
                          'text': 'BACK_%s' % state.foreground_activity}
        if self.search_method == POLICY_NAIVE_DFS:
            views.append(mock_view_back)
        elif self.search_method == POLICY_NAIVE_BFS:
            views.insert(0, mock_view_back)

        for view in views:
            view_text = view['text'] if view['text'] is not None else ''
            view_text = view_text.lower().strip()
            if view_text in self.preferred_buttons \
                    and (state.foreground_activity, view['view_str']) not in self.explored_views:
                self.logger.info("selected an preferred view: %s" % view['view_str'])
                return view

        for view in views:
            if (state.foreground_activity, view['view_str']) not in self.explored_views:
                self.logger.info("selected an un-clicked view: %s" % view['view_str'])
                return view

        if self.random_input:
            random.shuffle(views)
        transition_views = {transition[0] for transition in self.state_transitions}
        for view in views:
            if view['view_str'] in transition_views:
                self.logger.info("selected a transition view: %s" % view['view_str'])
                return view

        self.logger.info("no view could be selected in state: %s" % state.tag)
        return None

    def save_state_transition(self, event_str, old_state, new_state):
        
        if event_str is None or old_state is None or new_state is None:
            return
        if new_state.is_different_from(old_state):
            self.state_transitions.add((event_str, old_state.tag, new_state.tag))

    def save_explored_view(self, state, view_str):
        
        if not state:
            return
        state_activity = state.foreground_activity
        self.explored_views.add((state_activity, view_str))
