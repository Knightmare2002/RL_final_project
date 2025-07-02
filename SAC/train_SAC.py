import torch
import gym
from stable_baselines3 import SAC
import sys
sys.path.append('/home/dani/RL_project_colab')
from env.custom_hopper import *

model_path = "/home/dani/RL_project_colab/models/"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Using device:" , device)

env = gym.make('CustomHopper-source-v0')
print("State space:" , env.observation_space)
print("Action space:" , env.action_space)
print("Parameters:", env.get_parameters())

training_time_steps = 1000000
model = SAC("MlpPolicy",env=env,learning_rate=3e-4,gamma=0.99,verbose=0) #we imported PPO from stable baselines 3 
model.learn(training_time_steps,progress_bar=True)
model.save(model_path + "model_SAC.mdl")

