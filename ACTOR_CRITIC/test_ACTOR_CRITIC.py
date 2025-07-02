import torch
import gym
import sys
sys.path.append('/home/dani/RL_project_colab')  
from ACTOR_CRITIC.agent_ACTOR_CRITIC import Agent, Policy
from env.custom_hopper import *



#model = str(input("Model name: ")) # Fill in model path
model = "models/model_ACTOR_CRITIC.mdl"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using {device}")

episodes = 1000

env:CustomHopper = gym.make('CustomHopper-source-v0')

# env = gym.make('CustomHopper-target-v0')

print('Action space:', env.action_space)
print('State space:', env.observation_space)
body_names , selected_masses = env.get_parameters()
for body_name,selected_mass in zip(body_names,selected_masses):
    print(f"[DEFAULT] --> {body_name} : {selected_mass}")

observation_space_dim = env.observation_space.shape[-1]
action_space_dim = env.action_space.shape[-1]

policy = Policy(observation_space_dim, action_space_dim)
policy.load_state_dict(torch.load(model), strict=True)

agent = Agent(policy, device=device)

for episode in range(episodes):
  done = False
  test_reward = 0
  state = env.reset()
  body_names , selected_masses = env.get_parameters()
  for body_name,selected_mass in zip(body_names,selected_masses):
    print(f"{body_name} : {selected_mass}")

  while not done:
    action, _ = agent.get_action(state, evaluation=True)

    state, reward, done, info = env.step(action.detach().cpu().numpy())
    env.render()
    test_reward += reward


  #print(f"Episode: {episode} | Return: {test_reward}")