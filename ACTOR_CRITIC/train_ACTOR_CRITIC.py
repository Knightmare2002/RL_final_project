import argparse
import torch
import gym
import sys
from tqdm import tqdm
sys.path.append('/home/dani/RL_project_colab')  
from env.custom_hopper import *
from ACTOR_CRITIC.agent_ACTOR_CRITIC import Agent, Policy


n_episodes = 20000
print_every = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

env:CustomHopper = gym.make('CustomHopper-source-v0') #cast to CustomHopper

# env = gym.make('CustomHopper-target-v0')

print('Action space:', env.action_space)
print('State space:', env.observation_space)
body_names,masses = env.get_parameters()
body_names , selected_masses = env.get_parameters()
for body_name,selected_mass in zip(body_names,selected_masses):
    print(f"[DEFAULT] --> {body_name} : {selected_mass}")
"""
  Training
"""
observation_space_dim = env.observation_space.shape[-1]
action_space_dim = env.action_space.shape[-1]
print(f"UDR: {env.enable_udr}")
policy = Policy(observation_space_dim, action_space_dim)
agent = Agent(policy, device)

for episode in tqdm(range(n_episodes)):
  done = False
  train_reward = 0
  state = env.reset()  # Reset the environment and observe the initial state
  
  #after the while loop we will have the trajectory (the set of env.step)
  
  while not done:  # Loop until the episode is over
    action, action_probabilities = agent.get_action(state) #sampling the action : a
    previous_state = state
    
    state, reward, done, info = env.step(action.detach().cpu().numpy()) #obtaining s' , r_t
    
    agent.update_policy(previous_state, state, reward, done, action_probabilities)
    train_reward += reward


  if (episode+1)%print_every == 0:
    print('Training episode:', episode)
    print('Episode return:', train_reward)



torch.save(agent.policy.state_dict(), "ACTOR_CRITIC_3k.mdl")