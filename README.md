# RL_final_project
## Project Abstract

This project investigates the application of Reinforcement Learning (RL) to robotic control, with a focus
on sim-to-real transfer—training policies in simula-
tion that generalize to real-world conditions. The
first phase used the Gym Hopper environment, where
foundational (REINFORCE, Actor-Critic) and state-
of-the-art (PPO) algorithms were implemented for lo-
comotion. To simulate the reality gap, two custom
domains were created: the target with nominal dy-
namics, and the source with a 30% torso mass re-
duction. Uniform Domain Randomization (UDR)
was applied to the remaining three link masses in
the source domain, using manually defined uniform
distributions. Mass values were sampled at each
episode to promote robustness across dynamic vari-
ations. The project then extended to a Webots-based
scenario, where a Tesla Model 3 learned obstacle
avoidance using LiDAR, GPS, and IMU. PPO and
SAC were employed, with UDR applied to obstacle
positions and to the vehicle’s initial conditions.

---
# 1.Gym Hopper

This repository contains reinforcement learning agents for training a one-legged Hopper robot in simulation using custom environments and widely used RL algorithms.

The project features two Hopper environment variants:
- **Source Domain**: torso mass reduced by 30%.
- **Target Domain**: original environment, serving as a proxy for the real world.

Uniform Domain Randomization (UDR) is applied to all link masses except the torso in the source domain to improve policy robustness for sim-to-real transfer.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [How It Works](#how-it-works)
- [Training Instructions](#training-instructions)
- [Evaluation](#evaluation)
- [Logging and Monitoring](#logging-and-monitoring)
- [Troubleshooting](#troubleshooting)

---

## Features
- Custom Gym Hopper environments with source and target domains.
- REINFORCE (with and without baseline) and Actor-Critic algorithms implemented from scratch.
- PPO and SAC agents implemented using Stable Baselines3.
- Domain randomization (UDR) on link masses (torso excluded).
- Integrated logging and monitoring with Weights & Biases (wandb) and TensorBoard.

---

## Installation

Follow these steps to set up the environment:

1. **Clone the repository**
   ```bash
   git clone https://github.com/Polixide/RL_final_project.git
   cd RL_final_project/Gym_Hopper
   ```

2. **Create and activate a Python virtual environment**
   ```bash
   python3 -m venv mldl_env
   source mldl_env/bin/activate  # On Windows: rl_env\Scripts\activate
   ```

3. **Upgrade pip**
   ```bash
   python -m pip install --upgrade pip
   ```

4. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install MuJoCo**
   - Download MuJoCo from [https://mujoco.org/](https://mujoco.org/)
   - Extract it to `~/.mujoco/mujoco210`
   - Set environment variables:
     ```bash
     export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
     ```
   - Install `mujoco-py`:
     ```bash
     pip install mujoco-py==2.1
     ```

6. **Configure Weights & Biases (optional)**
   ```bash
   pip install wandb
   wandb login
   ```

7. **Verify installation**
   Run the test script to confirm MuJoCo and Gym integration:
   ```bash
   python test_random_policy.py
   ```

---

## Folder Structure
```
Gym_Hopper/
├── ACTOR_CRITIC/
│   ├── agent_ACTOR_CRITIC.py         # Actor-Critic agent implementation
│   ├── train_ACTOR_CRITIC.py         # Training script
│   └── test_ACTOR_CRITIC.py          # Evaluation script
│
├── REINFORCE/
│   ├── agent_REINFORCE.py            # REINFORCE agent implementation
│   ├── train_REINFORCE.py            # Training script
│   └── test_REINFORCE.py             # Evaluation script
│
├── PPO/
│   ├── train_PPO.py                  # PPO training using Stable Baselines3
│   └── test_PPO.py                   # PPO evaluation script
│
├── SAC/
│   ├── train_SAC.py                  # SAC training using Stable Baselines3
│   └── test_SAC.py                   # SAC evaluation script
│
├── env/
│   ├── __init__.py                   # Package initializer
│   ├── custom_hopper.py              # Custom Hopper environment with domain randomization
│   ├── mujoco_env.py                 # MuJoCo wrapper class
│   └── assets/
│       └── hopper.xml                 # Hopper robot definition
│
├── mldl_env/                         # Placeholder for additional environments
├── models/                           # Saved models during training
├── checkpoints/                      # Training checkpoints
├── tb_logs/                          # TensorBoard logs
├── wandb/                             # Weights & Biases logs
├── videos/                           # Simulation videos
├── test_random_policy.py              # Script for testing random policies
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Git ignored files
└── README.md                          # Project documentation
```

---

## How It Works
- The `env/custom_hopper.py` defines two Gym environments:
  - `CustomHopper-source-v0`: torso mass reduced by 30%.
  - `CustomHopper-target-v0`: original Hopper model.
- REINFORCE and Actor-Critic are implemented from scratch.
- PPO and SAC are implemented using Stable Baselines3.
- UDR randomizes all link masses except the torso in the source domain.

---

## Training
To train an agent, run the corresponding training script for the desired algorithm. Replace `ALGO` with the algorithm name (e.g., `ACTOR_CRITIC`, `REINFORCE`, `PPO`, `SAC`):

```bash
python ALGO/train_ALGO.py
```

If you want to enable **Uniform Domain Randomization (UDR)** during training, you need to set the `enable_udr` variable to `True` directly in the `custom_hopper.py` file before starting the training script.

---

## Evaluation
To evaluate a trained agent, run the corresponding evaluation script for the desired algorithm. Replace `ALGO` with the algorithm name (e.g., `ACTOR_CRITIC`, `REINFORCE`, `PPO`, `SAC`):

```bash
python ALGO/test_ALGO.py
---

## Logging and Monitoring
- Authenticate with wandb:
  ```bash
  wandb login
  ```
---

## Troubleshooting
- Confirm MuJoCo is installed and licensed correctly.
- Verify `mujoco-py` dependencies.
- Run `wandb login` if authentication is required.

---
# 2.Project_extension


## Table of Contents

- [Features](#features-1)
- [Installation](#installation-1)
- [Folder Structure](#folder-structure-1)
- [How It Works](#how-it-works-1)
- [Training Instructions](#training-instructions-1)
- [Evaluation](#evaluation-1)
- [Logging and Monitoring](#logging-and-monitoring)
- [Troubleshooting](#troubleshooting-1)
- [Authors](#authors)

---

## Features

- Webots-based Tesla simulation environment
- Custom Gymnasium-compatible Python environment
- Socket-based communication between Webots and Python
- Reinforcement Learning with Stable-Baselines3 (e.g., PPO, SAC)
- Parallel training via `SubprocVecEnv`
- Reward shaping for safe driving and obstacle avoidance
- Model checkpointing and logging (W&B + TensorBoard)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Polixide/RL_final_project.git
cd RL_project_final/Project_extension
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)
```
python3 -m venv rl_env
source rl_env/bin/activate      
# On Windows: rl_env\Scripts\activate
```
### 3. Install All Python Dependencies
```
pip install -r requirements.txt
```
### 4. Install Webots 2025
Go to the official Cyberbotics website: https://cyberbotics.com/#download ,
or use the direct link:
```
wget https://github.com/cyberbotics/webots/releases/download/R2025a/webots-R2025a-x86-64.tar.bz2
```
Extract and install:
```
tar -xjf webots-R2025a-x86-64.tar.bz2
sudo mv webots /opt/webots
```
You can now run Webots using:
```
webots
```


## Folder Structure
```bash
Projects_extension/
├── controllers/              # Webots controllers (e.g., tesla_controller.py)
├── worlds/                   # Webots simulation world (.wbt files)
├── model_dir/                # Final saved models (e.g., best_model.zip)
├── checkpoint_dir/           # Intermediate checkpoints (only contents ignored)
│   └── .gitkeep
├── tb_logs/                  # TensorBoard logs (only contents ignored)
│   └── .gitkeep
├── wandb/                    # Weights & Biases logs (only contents ignored)
│   └── .gitkeep
├── rl_env/                   # your python virtual environment with all the dependencies
│  
├── webots_remote_env.py      # Webots socket communication handler
├── train_PPO.py              # PPO training script
├── train_SAC.py              # SAC training script
├── test_PPO.py               # PPO evaluation script
├── test_SAC.py               # SAC evaluation script
├── run_PPO.sh                # Shell script to launch PPO training
├── run_SAC.sh                # Shell script to launch SAC training
├── requirements.txt          # All Python dependencies
└── README.md                 # Project documentation

```
## How It Works
- Webots runs a 3D world with a Tesla robot.

- The robot uses tesla_controller.py which acts as a socket server.

- Python scripts (e.g., train.py) act as socket clients and communicate with the controller.

- The RL agent sends reset, step, and exit commands.

- Observations are gathered from sensors (e.g., distances, camera).

- Rewards guide learning to avoid obstacles and drive safely.

- Logging is done via W&B and TensorBoard.

## Training Instructions



Start Training
```
# For PPO training
./run_PPO.sh

# For SAC training
./run_SAC.sh
```
Training uses multiple environments in parallel via SubprocVecEnv. You can adjust the number of instances and hyperparameters inside train.py.

## Evaluation
Step 1 — Open Webots
Launch the simulation:


```
webots worlds/your_world.wbt
```
**WARNING: Make sure your robot (Tesla) uses tesla_controller.py as the controller.**

Step 2 - Launch the test.py file
```
python test_ALGO.py 
```
**You can see the car movements in webots while the test is running.**

## Logging and Monitoring

Before first use:
```
wandb login
```
Training logs and plots are available on https://wandb.ai under your account.

## Troubleshooting

- GitHub Password Auth Failed:
Use a Personal Access Token (PAT) instead of your GitHub password.
→ Create Token

- Folders Not Being Pushed to Git:
Add a .gitkeep file inside empty folders that are otherwise ignored by .gitignore.

- Socket Connection Error:
Ensure Webots is running and listening on the correct port.
You can set PORT manually in your environment or use the default (e.g., 10000).

## Authors

- Daniele Catalano (@Polixide) - Politecnico di Torino , Data Science & Engineering
- Samuele Caruso (@Knightmare2002) - Politecnico di Torino , Data Science & Engineering

- Francesco Dal Cero (@Dalceeee) - Politecnico di Torino , Data Science & Engineering

- Ramadan Mehmetaj (@Danki02) - Politecnico di Torino , Data Science & Engineering
