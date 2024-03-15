import random
import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import rl_utils


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        # Calculate prob
        log_prob = log_prob.sum(-1, keepdim=True)
        action = torch.tanh(normal_sample)  # Adjust the log_prob by tanh function

        log_prob -= torch.log(1 - action.pow(2) + 1e-7).sum(-1, keepdim=True)
        action = action * self.action_bound
        # print("log_prob shape:", log_prob.shape)
        # print("action shape:", action.shape)
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        # remove unexpected dim
        if x.dim() > 2:
            x = x.squeeze(1)
        if a.dim() > 2:
            a = a.squeeze(1)
        # print("x shape after squeeze:", x.shape)  # expected [64, 12]
        # print("a shape after squeeze:", a.shape)  # expected [64, 4]
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class SAC:

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device, action_space):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         action_bound).to(device)  # actor network
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # Q network1
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # Q network2
        self.target_critic_1 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
            device)  # target Q network1
        self.target_critic_2 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
            device)  # target Q network2
        # Initialize target critic network with the same weight
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.action_space = action_space

        self.actor_loss = None
        self.critic_1_loss = None
        self.critic_2_loss = None

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action, _ = self.actor(state)
        action = action.cpu().detach().numpy()
        # Clip the action by the action space
        return np.clip(action, self.action_space.low, self.action_space.high)

    def calc_target(self, rewards, next_states, dones):  # Calculate the target value
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob  # This should be torch.Size([64, 1])

        # print("Dim of next_actions:", next_actions.shape)
        # print("Dim of next_actions:", next_actions.shape)
        # print("Dim of entropy:", entropy.shape)

        # print("Dim of next_actions:", next_actions.shape)
        q1_value = self.target_critic_1(next_states, next_actions)  # This should be torch.Size([64, 1])
        # print("Dim of q1_value:", q1_value.shape)
        q2_value = self.target_critic_2(next_states, next_actions)  # This should be torch.Size([64, 1])
        # print("Dim of q2_value:", q2_value.shape)
        # next_value = torch.min(q1_value,
        #                        q2_value) + self.log_alpha.exp() * entropy
        # Check if the min operation is correct
        min_q_value = torch.min(q1_value, q2_value)  # This should be torch.Size([64, 1])
        # print("Dim of min_q_value:", min_q_value.shape)

        next_value = min_q_value + self.log_alpha.exp() * entropy

        # print("Dim of reward:", rewards.shape)
        # print("Dim of next_value:", next_value.shape)
        # print("Dim of dones:", dones.shape)
        td_target = rewards + self.gamma * next_value * (1 - dones)
        # print("Dim of td_target:", td_target.shape)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        # states = torch.tensor(transition_dict['states'],
        #                       dtype=torch.float).to(self.device)
        # actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        #
        # rewards = torch.tensor(transition_dict['rewards'],
        #                        dtype=torch.float).view(-1, 1).to(self.device)
        # next_states = torch.tensor(transition_dict['next_states'],
        #                            dtype=torch.float).to(self.device)
        # dones = torch.tensor(transition_dict['dones'],
        #                      dtype=torch.float).view(-1, 1).to(self.device)
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        # Update Q network
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Update actor network
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        # Update loss for tensorboard
        self.actor_loss = actor_loss.item()
        self.critic_1_loss = critic_1_loss.item()
        self.critic_2_loss = critic_2_loss.item()

    def get_last_losses(self):
        return self.actor_loss, self.critic_1_loss, self.critic_2_loss
