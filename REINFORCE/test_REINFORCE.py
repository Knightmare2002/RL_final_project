import torch
import gym

from env.custom_hopper import *
from REINFORCE.agent_REINFORCE import Agent, Policy

model = str(input("Model name: ")) # Fill in model path
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using {device}")

episodes = 50000

env = gym.make('CustomHopper-source-v0')
# env = gym.make('CustomHopper-target-v0')

print('Action space:', env.action_space)
print('State space:', env.observation_space)
print('Dynamics parameters:', env.get_parameters())

observation_space_dim = env.observation_space.shape[-1]
action_space_dim = env.action_space.shape[-1]

policy = Policy(observation_space_dim, action_space_dim)
policy.load_state_dict(torch.load(model), strict=True)

agent = Agent(policy, device=device)

for episode in range(episodes):
  done = False
  test_reward = 0
  state = env.reset()

  while not done:
    action, _ = agent.get_action(state, evaluation=True)

    state, reward, done, info = env.step(action.detach().cpu().numpy())
    env.render()
    test_reward += reward


  print(f"Episode: {episode} | Return: {test_reward}")