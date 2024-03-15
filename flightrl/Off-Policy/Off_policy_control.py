#!/usr/bin/env python3
import sys
import time
from ruamel.yaml import YAML, dump, RoundTripDumper
import gym
import os
import math
import argparse
import tensorflow as tf
from io import StringIO
from SAC import SAC
from DDPG import DDPG
from rpg_baselines.ppo.ppo2_test import test_model
import wrapper as wrapper
import rpg_baselines.common.util as U
from flightgym import QuadrotorEnv_v1
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import rl_utils
from TD3 import TD3
from torch.utils.tensorboard import SummaryWriter


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1,
                        help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=0,
                        help="Enable Unity Render")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('--algorithm',"-a", type=str, default='SAC',
                        help="Algorithm name (SAC, DDPG, TD3)")
    parser.add_argument('-w', '--weight', type=str, default='./saved_sac/actor.pth13063',  # default='
                        # ./saved_sac/actor.pth13063'
                        help='trained weight path')
    return parser


def main():
    args = parser().parse_args()
    yaml = YAML(typ='unsafe', pure=True)
    cfg_path = "./config_output.yaml"
    log_dir = os.path.join(args.save_dir, "tensorboard_logs")

    with open(cfg_path, 'r') as file:
        cfg = yaml.load(file)

    if not args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1

    if args.render:
        cfg["env"]["render"] = "yes"
    else:
        cfg["env"]["render"] = "no"

    # Dump configuration to a string using StringIO
    cfg_dump = StringIO()
    yaml.dump(cfg, cfg_dump)
    cfg_dump.seek(0)  # Rewind to the beginning of the StringIO object

    # Set Number of environments to 1 for algorithms that do not support multiple environments
    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(cfg_dump.getvalue(), False))
    # env = gym.make("CartPole-v1")
    # set random seed
    configure_random_seed(args.seed, env=env)

    #
    if args.train:
        env_name = "Drone"
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high[0]  # Maximum value of the action
        random.seed(0)
        np.random.seed(0)
        env.seed(0)
        torch.manual_seed(0)
        actor_lr = 3e-4
        critic_lr = 3e-3
        alpha_lr = 3e-4
        num_episodes = 1000
        hidden_dim = 128
        gamma = 0.99
        tau = 0.005
        buffer_size = 100000
        minimal_size = 1000
        batch_size = 256
        target_entropy = -env.action_space.shape[0]
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Device: {device}")

        action_space = env.action_space
        replay_buffer = rl_utils.ReplayBuffer(buffer_size)
        writer = SummaryWriter(
            log_dir + "/" + args.algorithm)
        if args.algorithm == 'SAC':
            print("SAC Training")
            start = time.time()
            agent = SAC(state_dim, hidden_dim, action_dim, action_bound,
                        actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                        gamma, device, action_space)

            return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes,
                                                          replay_buffer, minimal_size,
                                                          batch_size, total_iteration=10, writer=writer)
            end = time.time()
            print(f"Training time: {end - start}")

            episodes_list = list(range(len(return_list)))
            # save model
            print(f'Models saved_sac after training.')
            save_model(agent.actor, "./saved_sac/actor.pth" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
            # save_model(agent.critic_1, "./saved_sac/critic_1.pth" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
            # save_model(agent.critic_2, "./saved_sac/critic_2.pth" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

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


        elif args.algorithm == 'DDPG':
            # DDPG
            print("DDPG Training")
            start = time.time()
            sigma = 0.2
            agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma,
                         device,
                         action_space)

            return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size,
                                                          batch_size, total_iteration=10, writer=writer)
            end = time.time()
            print(f"Training time: {end - start}")
            episodes_list = list(range(len(return_list)))

            # save model
            print(f'Models saved_DDPG after training.')
            save_model(agent.actor, "./saved_DDPG/actor_DDPG.pth" + time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                                  time.localtime()) + "-sigma=" + str(
                sigma))

            plt.plot(episodes_list, return_list)
            plt.xlabel('Episodes')
            plt.ylabel('Returns')
            plt.title('DDPG on {}'.format(env_name))
            plt.show()

            mv_return = rl_utils.moving_average(return_list, 9)
            plt.plot(episodes_list, mv_return)
            plt.xlabel('Episodes')
            plt.ylabel('Returns')
            plt.title('DDPG smoothed on {}'.format(env_name))
            plt.show()


        elif args.algorithm == 'TD3':
            print("TD3 Training")
            start = time.time()
            # TD3
            sigma = 0.35
            policy_noise = 0.2
            noise_clip = 0.5
            policy_freq = 2
            agent = TD3(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma,
                        policy_noise
                        , policy_noise, policy_freq, device, action_space)
            return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size,
                                                          batch_size, total_iteration=10, writer=writer)
            end = time.time()
            print(f"Training time: {end - start}")
            episodes_list = list(range(len(return_list)))

            # save model
            print(f'Models saved_TD3 after training.')
            save_model(agent.actor, "./saved_TD3/actor_TD3.pth" + time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                                time.localtime()) + "-sigma=" + str(
                sigma) + "-policy_noise=" + str(policy_noise) + "-noise_clip="
                       + str(noise_clip) + "-policy_freq=" + str(policy_freq))

            plt.plot(episodes_list, return_list)
            plt.xlabel('Episodes')
            plt.ylabel('Returns')
            plt.title("TD3 on Drone")
            plt.show()
            mv_return = rl_utils.moving_average(return_list, 9)
            plt.plot(episodes_list, mv_return)
            plt.xlabel('Episodes')
            plt.ylabel('Returns')
            plt.title('DDPG on {}'.format(env_name))
            plt.show()
        else:
            raise ValueError("Algorithm not implemented")


    else:
        # Load model
        actor_lr = 3e-4
        critic_lr = 3e-3
        alpha_lr = 3e-4
        hidden_dim = 128
        gamma = 0.99
        tau = 0.005
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high[0]
        random.seed(0)
        np.random.seed(0)
        env.seed(0)
        torch.manual_seed(0)
        target_entropy = -env.action_space.shape[0]
        action_space = env.action_space
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")

        if args.algorithm == 'SAC':
            print("SAC Rendering")
            agent = SAC(state_dim, hidden_dim, action_dim, action_bound,
                        actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                        gamma, device, action_space)
            # actor_model_path = "./saved_sac/"+"actor.pth13063" # best model
            if args.weight == "./saved_sac/actor.pth13063":
                print("Using default model")
                actor_model_path = args.weight
            else:
                actor_model_path = "./saved_sac/" + args.weight
            # actor_model_path = "./saved_sac/"+"actor.pth21005"
            # actor_model_path = "./saved_sac/"+ "actor.pth56001"
            # critic_1_model_path = "./saved_sac/critic_1.pth1"
            # critic_2_model_path = "./saved_sac/critic_2.pth1"
            agent.actor.load_state_dict(torch.load(actor_model_path))
            #         agent.critic_1.load_state_dict(torch.load(critic_1_model_path))
            #         agent.critic_2.load_state_dict(torch.load(critic_2_model_path))
            model = PyTorchModelWrapper(agent.actor, "SAC", device)
            test_model(env, model, True)

        elif args.algorithm == 'DDPG':
            print("DDPG Rendering")
            # DDPG
            sigma = 0.1
            agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma,
                         device,
                         action_space)
            # actor_model_path = "./saved_DDPG/" + "actor_DDPG.pth1514.0"
            actor_model_path = "./saved_DDPG/" + args.weight
            agent.actor.load_state_dict(torch.load(actor_model_path))

            print('Models loaded.')
            model = PyTorchModelWrapper(agent.actor, "DDPG", device)
            test_model(env, model, True)

        elif args.algorithm == 'TD3':
            print("TD3 Rendering")

            # TD3
            sigma = 0.3
            policy_noise = 0.2
            noise_clip = 0.5
            policy_freq = 2
            agent = TD3(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma,
                        policy_noise
                        , noise_clip, policy_freq, device, action_space)
            # actor_model_path = "./saved_TD3/" + "actor_TD3.pth431.0"
            actor_model_path = "./saved_TD3/" + args.weight
            print(actor_model_path)
            agent.actor.load_state_dict(torch.load(actor_model_path))

            print('Models loaded.')
            model = PyTorchModelWrapper(agent.actor, "TD3", device)
            test_model(env, model, True)


class PyTorchModelWrapper:
    def __init__(self, pytorch_model, model_type, device):
        self.model = pytorch_model
        self.model_type = model_type
        self.device = device

    def predict(self, observation, deterministic=True):
        # Transform observation to tensor
        # Add tensor to device to prevent error when using cuda
        obs_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device)
        if obs_tensor.ndim == 1:  # If the observation is a vector, add a batch dimension
            obs_tensor = obs_tensor.unsqueeze(0)

        # Use the model to predict the action
        with torch.no_grad():
            if self.model_type == 'SAC':
                action, _ = self.model(obs_tensor) if deterministic else self.model.sample(obs_tensor)
            elif self.model_type in ['DDPG', 'TD3']:
                action = self.model(obs_tensor)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        # Return the predicted action as a numpy array
        return action.cpu().numpy(), None


if __name__ == "__main__":
    main()
