# 🌟 Q-PredRacer 🌟

## 📱 Project Overview

Q-PredRacer is a Q-learning-based approach for detecting data races in Android applications that integrates predictive techniques with PredRacer. The Android platform adopts a hybrid concurrency model that combines multithreading with asynchronous message passing, making data race detection particularly challenging. Traditional dynamic methods often produce false negatives, while predictive techniques relying on random exploration strategies (like Monkey) result in redundant UI traversals and insufficient code coverage.

Q-PredRacer addresses these challenges by leveraging reinforcement learning to guide dynamic execution toward components likely to expose data races, generating execution traces with a higher probability of data race occurrences. It then utilizes the existing predictive technique PredRacer to detect data races in mobile apps. Experimental results demonstrate that Q-PredRacer outperforms state-of-the-art approaches in both false positive and false negative rates.

## 🎯 Key Components

- **Q-PredRacer_policy.py**: 🧠 Our reinforcement learning-based policy for guiding GUI exploration
- **DroidBot Integration**: 🤖 Uses DroidBot as the underlying input generation framework
- **PredRacer Integration**: 🔍 Leverages PredRacer for data race detection

## 🚀 Installation and Setup

### 1. Prerequisites ✅

- Python (both 2 and 3 are supported)
- Java
- Android SDK (with platform-tools added to PATH)
- Android Virtual Device (AVD) or physical Android device
- Git

### 2. Install DroidBot 📦

Q-PredRacer uses DroidBot as the input generation framework. Follow these steps to install DroidBot:

```bash
git clone https://github.com/honeynet/droidbot.git
cd droidbot
pip install -e .
```

Verify the installation by running:

```bash
droidbot -h
```

### 3. Setup Q-PredRacer ⚙️

1. Clone this repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Copy the `Q-PredRacer_policy.py` file to the DroidBot directory:

```bash
cp Q-PredRacer_policy.py /path/to/droidbot/droidbot/
```

## 🎮 Usage

### 1. Prepare Android Environment 📱

- Start your Android Virtual Device (AVD) or connect a physical Android device
- Enable USB debugging on the device
- Verify the device is connected:

```bash
adb devices
```

### 2. Run Q-PredRacer for App Exploration 🔄

Use DroidBot with our Q-PredRacer policy to explore the target app:

```bash
# Replace <path-to-apk> with the path to your target APK file
# Replace <output-dir> with the directory to store exploration results
droidbot -a <path-to-apk> -o <output-dir> -policy q_predracer
```

### 3. Run PredRacer for Data Race Detection 🕵️‍♂️

After exploration, use PredRacer to detect data races in the generated execution traces:

```bash
# Follow PredRacer's instructions to analyze the traces
# This typically involves running PredRacer on the collected execution data
```

## ⚙️ Configuration

### Q-PredRacer Policy Parameters 🎛️

The `Q-PredRacer_policy.py` file contains several configurable parameters:

- `alpha`: Learning rate (default: 0.5)
- `gamma`: Discount factor (default: 0.9)
- `epsilon`: Exploration rate (default: varies during training)
- `reward`: Reward for detecting concurrency events
- `delta`: Penalty for non-productive actions

## 📊 Evaluation

Experimental results demonstrate that Q-PredRacer outperforms state-of-the-art approaches in both false positive and false negative rates. The reinforcement learning-based guidance effectively reduces redundant UI traversals and focuses on components likely to expose data races.

## 🔗 References

- [DroidBot GitHub Repository](https://github.com/honeynet/droidbot) 🐝
- Li, Yuanchun, et al. "DroidBot: a lightweight UI-guided test input generator for Android." In Proceedings of the 39th International Conference on Software Engineering Companion (ICSE-C '17). Buenos Aires, Argentina, 2017.
- [PredRacer Documentation](<predracer-documentation-link>) 📚

## 💬 Contact

For questions or issues, please contact:

- <your-name> 👤
- <your-email> 📧

---

✨ Happy racing against data races! ✨
