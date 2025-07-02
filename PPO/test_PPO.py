import torch
from stable_baselines3 import PPO
import gym
import sys
sys.path.append('/home/dani/RL_project_colab')
from env.custom_hopper import *

model = PPO.load("models/model_PPO.mdl")
env = gym.make('CustomHopper-source-v0')
test_episodes = 5000

for episode in range(test_episodes):

    observation = env.reset()
    done = False
    total_reward = 0

    while not done:
        action , _ = model.predict(observation=observation,deterministic=True)
        observation ,reward , done,info = env.step(action)
        total_reward += reward
        env.render()

    print(f"Episode: {episode+1}  Reward: {total_reward}")

