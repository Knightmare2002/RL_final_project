import argparse
import torch
import gym
import sys
sys.path.append('/home/dani/RL_project_colab')

from env.custom_hopper import *
from REINFORCE.agent_REINFORCE import Agent, Policy

n_episodes = 50000
print_every = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

env = gym.make('CustomHopper-source-v0')
# env = gym.make('CustomHopper-target-v0')

print('Action space:', env.action_space)
print('State space:', env.observation_space)
print('Dynamics parameters:', env.get_parameters())

"""
  Training
"""
observation_space_dim = env.observation_space.shape[-1]
action_space_dim = env.action_space.shape[-1]

policy = Policy(observation_space_dim, action_space_dim)
agent = Agent(policy, device)

for episode in range(n_episodes):
  done = False
  train_reward = 0
  state = env.reset()  # Reset the environment and observe the initial state

  #after the while loop we will have the trajectory (the set of env.step)
  
  while not done:  # Loop until the episode is over
    action, action_probabilities = agent.get_action(state) #sampling the action : a
    previous_state = state
    
    state, reward, done, info = env.step(action.detach().cpu().numpy()) #obtaining s' , r_t
    agent.store_outcome(previous_state, state, action_probabilities, reward, done)
    
    train_reward += reward

  agent.update_policy()

  if (episode+1)%print_every == 0:
    print('Training episode:', episode)
    print('Episode return:', train_reward)



torch.save(agent.policy.state_dict(), "model_actor_critic.mdl")