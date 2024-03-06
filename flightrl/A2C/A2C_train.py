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

from A2C import A2C
from rpg_baselines.ppo.ppo2_test import test_model
from rpg_baselines.envs import vec_env_wrapper as wrapper
import rpg_baselines.common.util as U
#
from flightgym import QuadrotorEnv_v1

from stable_baselines.common.policies import MlpPolicy, CnnPolicy


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
    parser.add_argument('-w', '--weight', type=str, default='./saved_sac/2024-02-27-14-36-26_Iteration_2379.zip',
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
        log_dir = rsg_root + '/saved_sac'
        saver = U.ConfigurationSaver(log_dir=log_dir)
        '''eslf, policy, env, gamma=0.99, n_steps=5, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5,
                 learning_rate=7e-4, alpha=0.99, momentum=0.0, epsilon=1e-5, lr_schedule='constant',
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None)'''
        model = A2C(
            tensorboard_log=saver.data_dir,
            policy=MlpPolicy,  # check activation function
            env=env,
            gamma=0.99,  # lower 0.9 ~ 0.99
            # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
            n_steps=350,
            ent_coef=0.02,
            learning_rate=3e-3,
            vf_coef=0.25,
            max_grad_norm=0.5,
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
            total_timesteps=int(2000000000),log_dir=saver.data_dir,logger=logger)
        model.save(saver.data_dir)

    # # Testing mode with a trained weight
    else:
        model = A2C.load(args.weight)
        test_model(env, model, render=args.render)


if __name__ == "__main__":
    main()
