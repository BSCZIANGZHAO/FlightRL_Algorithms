import os
import math
import argparse
import numpy as np
import torch
import random
from ruamel.yaml import YAML, dump, RoundTripDumper

from stable_baselines3.common import logger
from stable_baselines3 import DQN
#
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rpg_baselines2.ppo.ppo2_test import test_model
from rpg_baselines2.envs import vec_env_wrapper as wrapper
import rpg_baselines2.common.util as U
#
from flightgym import QuadrotorEnv_v1


# import gym
# env_name = "LunarLander-v2"
# env = gym.make(env_name)

# def configure_random_seed(seed, env=None):
#     if env is not None:
#         env.seed(seed)
#     np.random.seed(seed)
#     tf.set_random_seed(seed)

# Fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=0,
                        help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=0,
                        help="Enable Unity Render")
    parser.add_argument('--save_interval', type=int, default=1000000,
                        help="save_interval")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('-w', '--weight', type=str, default='./saved/2024-03-14-22-03-10/10000.zip',
                        help='trained weight path')
    return parser

def dqn_api(train=0, render=0, save_interval=1000000,
             seed = 0, weight='./saved/2024-03-14-22-03-10/20000.zip'):
    # save_dir = os.path.dirname(os.path.realpath(__file__))
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                           "/flightlib/configs/vec_env.yaml", 'r'))
    if not train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1

    if render:
        cfg["env"]["render"] = "yes"
    else:
        cfg["env"]["render"] = "no"

    env = wrapper.FlightEnvVec2(QuadrotorEnv_v1(
        dump(cfg, Dumper=RoundTripDumper), False))

    # configure_random_seed(args.seed, env=env)
    same_seeds(seed)
    env.seed(seed)

    #
    if train:
        # save the configuration and other files
        rsg_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = rsg_root + '/saved'
        saver = U.ConfigurationSaver(log_dir=log_dir)
        model = DQN(
            tensorboard_log=saver.data_dir,
            policy='MlpPolicy',
            policy_kwargs=dict(
                net_arch=[128, 128]),
            batch_size=64,
            env=env,
            # nminibatches=1,
            # noptepochs=10,
            # cliprange=0.2,
            verbose=1,
        )
        # print(model.policy)
        logger.configure(folder=saver.data_dir)
        for i in range(25):
            model.learn(
                total_timesteps=save_interval,
                reset_num_timesteps=False,
            )
            model.save(f"{saver.data_dir}/{(i + 1) * save_interval}")

    # # Testing mode with a trained weight
    else:
        model = DQN.load(weight)
        test_model(env, model, render=render)


def main():
    args = parser().parse_args()
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                           "/flightlib/configs/vec_env.yaml", 'r'))
    if not args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1

    if args.render:
        cfg["env"]["render"] = "yes"
    else:
        cfg["env"]["render"] = "no"

    env = wrapper.FlightEnvVec2(QuadrotorEnv_v1(
        dump(cfg, Dumper=RoundTripDumper), False))

    # configure_random_seed(args.seed, env=env)
    same_seeds(args.seed)
    env.seed(args.seed)

    #
    if args.train:
        # save the configuration and other files
        rsg_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = rsg_root + '/saved'
        saver = U.ConfigurationSaver(log_dir=log_dir)
        model = DQN(
            tensorboard_log=saver.data_dir,
            policy='MlpPolicy',
            policy_kwargs=dict(
                net_arch=[128, 128]),
            batch_size=64,
            env=env,
            # nminibatches=1,
            # noptepochs=10,
            # cliprange=0.2,
            verbose=1,
        )
        # print(model.policy)
        logger.configure(folder=saver.data_dir)
        for i in range(25):
            model.learn(
                total_timesteps=args.save_interval,
                reset_num_timesteps=False,
            )
            model.save(f"{saver.data_dir}/{(i + 1) * args.save_interval}")

    # # Testing mode with a trained weight
    else:
        model = DQN.load(args.weight)
        test_model(env, model, render=args.render)


if __name__ == "__main__":
    main()
