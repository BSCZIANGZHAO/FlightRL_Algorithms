import random
import gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)  # Expand the dimension
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)  # Compress the dimension
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = torch.tanh(self.fc4(x)) * self.action_bound
        return action

class QValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)  # Keep the dimension

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
class TD3:

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma,
                 policy_noise, noise_clip, policy_freq, device, action_space):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.actor_target = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.action_space = action_space
        self.action_dim = action_dim
        self.sigma = sigma
        self.action_bound = action_bound
        self.total_it = 0

        self.actor_loss = None
        self.critic_1_loss = None
        self.critic_2_loss = None

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = self.actor(state).cpu().detach().numpy()
        # Add noise to the action
        action = action + self.sigma * np.random.randn(self.action_dim)
        action = np.ascontiguousarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def update(self, transition_dict):
        self.total_it += 1
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        # states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        # actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        # rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        # next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        # dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # print(states.shape,"states shape")
        # print(actions.shape,"actions shape")
        # print(rewards.shape,"rewards shape")
        # print(next_states.shape,"next_states shape")
        # print(dones.shape,"dones shape")
        actions = actions.squeeze(1) # Remove the dimension
        with torch.no_grad():
            # Add noise to the target action, increase exploration
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.action_bound, self.action_bound)

            # Calculate the target Q value
            target_Q1 = self.critic_target_1(next_states, next_actions)
            target_Q2 = self.critic_target_2(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + ((1 - dones) * self.gamma * target_Q).detach()

        # Calculate the current Q value
        current_Q1 = self.critic_1(states, actions)
        current_Q2 = self.critic_2(states, actions)

        # Calculate the critic loss
        critic_loss_1 = F.mse_loss(current_Q1, target_Q)
        critic_loss_2 = F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()


        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # actor loss
            actor_loss = -self.critic_1(states, self.actor(states)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update the target networks
            self.soft_update(self.critic_1, self.critic_target_1)
            self.soft_update(self.critic_2, self.critic_target_2)
            self.soft_update(self.actor, self.actor_target)

            self.actor_loss = actor_loss
            self.critic_1_loss = critic_loss_1
            self.critic_2_loss = critic_loss_2


    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def get_last_losses(self):
        if self.actor_loss is not None:
            return self.actor_loss.item(), self.critic_1_loss.item(), self.critic_2_loss.item()
        else:
            return None, None, None