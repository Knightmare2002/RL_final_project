import sys
import os
# Setup progetto e path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from env.custom_hopper import *
import torch
import gym
from stable_baselines3 import PPO
import wandb
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback


CHECKPOINT_DIR = "checkpoints/"
MODEL_DIR = "models/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
env:CustomHopper = gym.make('CustomHopper-source-v0')


print("Using device:" , device)
print("State space:" , env.observation_space)
print("Action space:" , env.action_space)
print("Parameters:", env.get_parameters())
print(f"Using UDR: {env.enable_udr}")
#----------------WANDB----------------------------------------
config = {
    "env_name": "CustomHopper-source-v0",
    "algorithm": "PPO",
    "training_time_steps": 1_000_000,
    "learning_rate": 2.5e-4,             
    "gamma": 0.995,                      
    "gae_lambda": 0.97,                  
    "clip_range": 0.1,               
    "n_steps": 4096,                 
    "ent_coef": 0.01,                     
    "batch_size": 128,                
    "n_epochs": 15,
    "device": device,
    "udr_enabled": env.enable_udr
}

wandb.init(
        project="RL_gym_hopper",          
        name="PPO-1M-train-source-UDR-run02",    
        config=config,
        sync_tensorboard=True,             
        monitor_gym=True,                  
        save_code=True
    )
# Checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,                  
    save_path=CHECKPOINT_DIR,
    name_prefix="PPO-1M-Hopper-source-UDR-v2"
)

# Wandb callback
wandb_callback = WandbCallback(
    gradient_save_freq=0,
    model_save_path=MODEL_DIR,
    verbose=2,
)

class WandbTimestepLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        wandb.log({
            "custom/timesteps": self.num_timesteps
        }, step=self.num_timesteps)
        return True
#----------------------------------------------------

training_time_steps = 1_000_000

model = PPO(
    "MlpPolicy",
    env=env,
    learning_rate=2.5e-4,
    gamma=0.995,
    gae_lambda=0.97,
    clip_range=0.1,
    n_steps=4096,
    ent_coef=0.01,
    batch_size=128,
    n_epochs=15,
    verbose=1,
    tensorboard_log="./tb_logs/"
)

model.learn(training_time_steps,progress_bar=True,callback=[checkpoint_callback,wandb_callback,WandbTimestepLogger()])

model.save(MODEL_DIR + "PPO_1M_SOURCE_UDR_V2.mdl")
wandb.finish()
