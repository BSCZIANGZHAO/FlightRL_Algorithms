#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper
import io
#
import os
import math
import argparse
import numpy as np
import tensorflow as tf

#
from stable_baselines import logger

#
from rpg_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy as MlpPolicy2
from rpg_baselines.ppo.ppo2 import PPO2
from rpg_baselines.ppo.ppo2_test import test_model
from rpg_baselines.vpg.vpg import VPG
from rpg_baselines.trpo.trpo import TRPO
from rpg_baselines.a2c.a2c import A2C
from rpg_baselines.acktr.acktr import ACKTR


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
    parser.add_argument('-w', '--weight', type=str, default='./saved/quadrotor_env.zip',
                        help='trained weight path')
    parser.add_argument('--model', type=str, default='PPO',
                        help="Model name")
    return parser


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
    cfg["env"]["num_envs"] = 1

    # env = wrapper.FlightEnvVec(QuadrotorEnv_v1(
    #     dump(cfg, Dumper=RoundTripDumper), False))

    yaml = YAML()
    cfg_str = io.StringIO()
    yaml.dump(cfg, cfg_str)
    cfg_str.seek(0)
    cfg_yaml_str = cfg_str.read()

    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(cfg_yaml_str, False))


    # set random seed
    configure_random_seed(args.seed, env=env)

    #
    if args.train:
        # save the configuration and other files
        rsg_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = rsg_root + '/saved'
        saver = U.ConfigurationSaver(log_dir=log_dir)
        if args.model == "PPO":
            model = PPO2(
                tensorboard_log=saver.data_dir,
                policy=MlpPolicy, 
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
            )
        elif args.model == "VPG":
            model = VPG(
                tensorboard_log=saver.data_dir,
                policy=MlpPolicy,
                policy_kwargs=dict(
                    net_arch=[dict(pi=[128, 128], vf=[128, 128])], act_fun=tf.nn.relu),
                env=env,
                gamma=0.99,
                n_steps=1024,
                ent_coef=0.5,
                learning_rate=1e-4,
                vf_coef=0.5,
                max_grad_norm=0.5,
                nminibatches=8,
                noptepochs=1,
                verbose=1,
            )
        elif args.model == "TRPO":
            model = TRPO(
                tensorboard_log=saver.data_dir,
                policy=MlpPolicy2,
                policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])], act_fun=tf.nn.relu),
                env=env,
                gamma=0.99,
                timesteps_per_batch=2048,
                max_kl=0.02,
                cg_iters=15,
                lam=0.95,
                entcoeff=0.01,
                cg_damping=1e-2,
                vf_stepsize=1e-3,
                vf_iters=5,
                verbose=1,
            )
        elif args.model == "A2C":
            model = A2C(
                tensorboard_log=saver.data_dir,
                policy=MlpPolicy, 
                policy_kwargs=dict(
                    net_arch=[dict(pi=[128, 128], vf=[128, 128])], act_fun=tf.nn.relu),
                env=env,
                # lam=0.95,
                gamma=0.99,  # lower 0.9 ~ 0.99
                # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
                n_steps=350,
                ent_coef=0.02,
                # kl_coef=0.5,
                learning_rate=3e-4,
                vf_coef=0.25,
                max_grad_norm=0.5,
                # nminibatches=10,
                # noptepochs=10,
                verbose=1,
            )
        elif args.model == "ACKTR":
            model = ACKTR(
                tensorboard_log=saver.data_dir,
                policy=MlpPolicy,  
                policy_kwargs=dict(
                    net_arch=[dict(pi=[128, 128], vf=[128, 128])], act_fun=tf.nn.relu),
                env=env,
                gae_lambda=0.95,
                gamma=0.99,  # lower 0.9 ~ 0.99
                # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
                n_steps=250,
                ent_coef=0.02,
                learning_rate=3e-2,
                max_grad_norm=0.5,
                verbose=2,
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
        # model.learn(
        #     total_timesteps=int(25000000),
        #     log_dir=saver.data_dir, logger=logger)
        model.learn(total_timesteps=int(5000000),log_interval=10)
        model.save(saver.data_dir)

    # # Testing mode with a trained weight
    else:
        if args.model == "PPO":
            model = PPO2.load(args.weight)
        elif args.model == "VPG":
            model = VPG.load(args.weight)
        elif args.model == "TRPO":
            model = TRPO.load(args.weight)
        elif args.model == "A2C":
            model = A2C.load(args.weight)
        elif args.model == "ACKTR":
            model = ACKTR.load(args.weight)
        test_model(env, model, render=args.render)


if __name__ == "__main__":
    main()
