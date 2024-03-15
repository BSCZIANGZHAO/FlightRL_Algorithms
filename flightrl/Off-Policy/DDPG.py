import random
import gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound is the maximum value of the action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action = torch.tanh(self.fc2(x)) * self.action_bound
        return action

# uncomment the following code to use the more complex policy network

class PolicyNet2(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet2, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)  # expand the dimension
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)  # compress the dimension
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = torch.tanh(self.fc4(x)) * self.action_bound
        return action

class QValueNet2(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet2, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)  # keep the dimension

    def forward(self, x, a):
        if x.dim() > 2:
            x = x.squeeze(1)
        if a.dim() > 2:
            a = a.squeeze(1)
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        # remove extra dimensions if the second dim is 1
        if x.dim() > 2:
            x = x.squeeze(1)
        if a.dim() > 2:
            a = a.squeeze(1)
        # print("x shape after squeeze:", x.shape)  # expect [64, 12]
        # print("a shape after squeeze:", a.shape)  # expect [64, 4]
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class DDPG:

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device,
                 action_space):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # Initialized target critic network
        self.target_critic.load_state_dict(self.critic.state_dict())
        # Initialized target actor network
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma
        self.tau = tau
        self.action_dim = action_dim
        self.device = device
        self.action_space = action_space
        self.actor_loss = None
        self.critic_loss = None

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = self.actor(state).cpu().detach().numpy()
        # Add noise to the action, increase exploration
        action = action + self.sigma * np.random.randn(self.action_dim)
        action = np.ascontiguousarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    # def update(self, transition_dict):
    #     states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
    #     # print(states.shape,"states shape")
    #     actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
    #     # print(actions.shape,"actions shape")
    #     rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
    #     # print(rewards.shape,"rewards shape")
    #     next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
    #     # print(next_states.shape,"next_states shape")
    #     dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
    #     # print(dones.shape,"dones shape")
    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # soft update actor
        self.soft_update(self.critic, self.target_critic)  # soft update critic

        self.actor_loss = actor_loss
        self.critic_loss = critic_loss

    # get the last losses for tensorboard
    def get_last_losses(self):
        return self.actor_loss.item(), self.critic_loss.item()
