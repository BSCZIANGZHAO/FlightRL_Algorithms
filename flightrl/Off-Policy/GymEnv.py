import random
import sys
import time
import gym
import os
import math
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import rl_utils
from SAC import SAC

# env_name = 'Pendulum-v0'
env_name = "LunarLanderContinuous-v2"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0] # the maximum value of the action
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)

actor_lr = 3e-4
critic_lr = 3e-3
alpha_lr = 3e-4
num_episodes = 100
hidden_dim = 128
gamma = 0.99
tau = 0.005
buffer_size = 100000
minimal_size = 1000
batch_size = 64
target_entropy = -env.action_space.shape[0]
device = torch.device(
    "cpu")

replay_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = SAC(state_dim, hidden_dim, action_dim, action_bound,
                      actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                      gamma, device,action_space=env.action_space)

return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes,
                                              replay_buffer, minimal_size,
                                              batch_size)
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('SAC on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('SAC on {}'.format(env_name))
plt.show()

# Save model after training
save_path = "sac_pendulum.pth"
torch.save(agent.actor.state_dict(), save_path)
print(f"Final model saved at {save_path}")

# Load the model and render the result
agent.actor.load_state_dict(torch.load("sac_Lunar.pth"))
state = env.reset()
while True:
    action = agent.take_action(state)
    state, reward, done, _ = env.step(action)
    env.render()
    if done:
        break

env.close()
