#!/usr/bin/env python3
import time

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
from stable_baselines.common.policies import MlpPolicy as SBMlpPolicy
from rpg_baselines.common.policies import MlpPolicy
from ppo_penalty import PPO
from A2C import A2C
from rpg_baselines.ppo.ppo2_test import test_model
from rpg_baselines.envs import vec_env_wrapper as wrapper
import rpg_baselines.common.util as U
#
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
    parser.add_argument('--algorithm',"-a", type=str, default="PPO",
                        help="Algorithm to use")

    # parser.add_argument('-w', '--weight', type=str, default='./saved_sac/quadrotor_env.zip',
    #                     help='trained weight path')
    # panalty = 0.5, kl_target = 0.01 1000 iterations
    # parser.add_argument('-w', '--weight', type=str, default='./saved_sac/kt=0.01,pc=0.5-1000.zip',
    #                     help='trained weight path')
    # panalty = 0.5, kl_target = 0.05 1000 iterations
    # parser.add_argument('-w', '--weight', type=str, default='./saved_sac/2024-02-22-17-31-23.zip',
    #                     help='trained weight path')



    # parser.add_argument('-w', '--weight', type=str, default='./saved_sac/960iteration_target=0.02_withclip.zip',
    #                    help='trained weight path')

    # parser.add_argument('-w', '--weight', type=str, default='./saved_sac/1012Itration_no_clip.zip',
    #                     help='trained weight path')
    # parser.add_argument('-w', '--weight', type=str, default='./saved_sac/Iteration_2701_withclip_overfit.zip',
    #                      help='trained weight path')
    parser.add_argument('-w', '--weight', type=str, default='./saved/960iteration_target=0.02_withclip.zip',
                         help='trained weight path')
    return parser



def main():
    args = parser().parse_args()
    yaml = YAML(typ='unsafe', pure=True)
    cfg_path = os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/vec_env.yaml"
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
        if args.algorithm == "A2C":
            log_dir = rsg_root + '/saved_a2c'
        else:
            log_dir = rsg_root + '/saved'
        saver = U.ConfigurationSaver(log_dir=log_dir)
        if args.algorithm == "PPO":
            print("Training PPO")
            model = PPO(
                tensorboard_log=saver.data_dir,
                policy=MlpPolicy,  # check activation function
                policy_kwargs=dict(
                    net_arch=[dict(pi=[128, 128], vf=[128, 128])], act_fun=tf.nn.relu),
                env=env,
                lam=0.95,
                gamma=0.99,  # lower 0.9 ~ 0.99
                # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
                n_steps=250,
                ent_coef=0.00,
                learning_rate=3e-4,
                vf_coef=0.5,
                max_grad_norm=0.5,
                nminibatches=1,
                noptepochs=10,
                cliprange=0.2,
                verbose=1,
                beta=0.5,
                kl_target=0.005,
            )
        elif args.algorithm == "A2C":
            print("Training A2C")
            model = A2C(
                tensorboard_log=saver.data_dir,
                policy=SBMlpPolicy,  # check activation function
                policy_kwargs=dict(
                    net_arch=[dict(pi=[128, 128], vf=[128, 128])], act_fun=tf.nn.relu),
                env=env,
                gamma=0.99,  # lower 0.9 ~ 0.99
                # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
                n_steps=500,
                ent_coef=0.01,
                learning_rate=3e-3,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=1,
            )
        else:
            raise ValueError("Algorithm not implemented")
        # tensorboard
        # Make sure that your chrome browser is already on.
        # TensorboardLauncher(saver.data_dir + '/PPO2_1')

        # On-Policy run
        # Originally the total timestep is 5 x 10^8
        # 10 zeros for nupdates to be 4000
        # 1000000000 is 2000 iterations and so
        # 2000000000 is 4000 iterations.
        logger.configure(folder=saver.data_dir)
        start = time.time()
        model.learn(
            total_timesteps=int(250000000),
            log_dir=saver.data_dir, logger=logger)
        model.save(saver.data_dir)
        print("Training time: ", time.time() - start)

    # # Testing mode with a trained weight
    else:
        if args.algorithm == "PPO":
            model = PPO.load(args.weight)
        elif args.algorithm == "A2C":
            model = A2C.load(args.weight)
        else:
            raise ValueError("Algorithm not implemented")
        test_model(env, model, render=args.render)


if __name__ == "__main__":
    main()
