#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper
 
#
import os
import math
import argparse
import numpy as np
import tensorflow as tf
from io import StringIO
#
from stable_baselines import logger
#
from rpg_baselines.common.policies import MlpPolicy
from rpg_baselines.ppo.ppo2 import PPO2
from rpg_baselines.ppo.ppo2_test import test_model
from sac_example.sac.sac2 import SAC
from rpg_baselines.envs import vec_env_wrapper as wrapper
import rpg_baselines.common.util as U
#
from stable_baselines.sac.policies import SACPolicy

from flightgym import QuadrotorEnv_v1
 
 
def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
 
 
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
    # parser.add_argument('-w', '--weight', type=str, default='./saved/quadrotor_env.zip',
    #                     help='trained weight path')
    parser.add_argument('-w', '--weight', type=str, default='./saved/2024-02-25-14-57-37_Iteration_219.zip',
                        help='trained weight path')
    return parser
 
 
def main():
    args = parser().parse_args()
    yaml = YAML(typ='unsafe', pure=True)
    cfg_path = "./vec_env.yaml"
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
 
    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(cfg_dump.getvalue(), False))
 
    # set random seed
    configure_random_seed(args.seed, env=env)
 
    #
    if args.train:
        # save the configuration and other files
        rsg_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = rsg_root + '/saved'
        saver = U.ConfigurationSaver(log_dir=log_dir)
        model = SAC(
            policy="MlpPolicy",
            env=env,
            gamma=0.99,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=100,
            train_freq=1,
            gradient_steps=1,
            batch_size=256,
            tau=0.005,
            ent_coef='auto',
            target_update_interval=1,
            action_noise=None,
            verbose=1,
        )
 
        # tensorboard
        # Make sure that your chrome browser is already on.
        # TensorboardLauncher(saver.data_dir + '/PPO2_1')
 
        # PPO run
        # Originally the total timestep is 5 x 10^8
        # 10 zeros for nupdates to be 4000
        # 1000000000 is 2000 iterations and so
        # 2000000000 is 4000 iterations.
        logger.configure(folder=saver.data_dir)
        model.learn(
            total_timesteps=int(25000000))
        model.save(saver.data_dir)
 
    # # Testing mode with a trained weight
    else:
        model = PPO2.load(args.weight)
        test_model(env, model, render=args.render)
 
 
if __name__ == "__main__":
    main()
