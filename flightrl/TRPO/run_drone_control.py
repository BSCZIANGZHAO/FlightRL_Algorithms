#!/usr/bin/env python3
import sys

from ruamel.yaml import YAML, dump, RoundTripDumper

#
import os
import math
import argparse
import numpy as np
import tensorflow as tf
from io import StringIO
#
from stable_baselines import logger,DQN
from SAC import SAC
from rpg_baselines.ppo.ppo2_test import test_model
import wrapper as wrapper
import rpg_baselines.common.util as U

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
    parser.add_argument('-w', '--weight', type=str, default='./saved/2024-02-28-20-34-14.zip',
                        help='trained weight path')
    return parser



def main():
    args = parser().parse_args()
    yaml = YAML(typ='unsafe', pure=True)
    cfg_path = "./config_output.yaml"
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

    # 如果是不支持多线程的环境在参数中设置num_envs=1
    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(cfg_dump.getvalue(), False))

    # set random seed
    configure_random_seed(args.seed, env=env)

    #
    if args.train:
        # save the configuration and other files
        rsg_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = rsg_root + '/saved'
        saver = U.ConfigurationSaver(log_dir=log_dir)
        # self, policy, env, gamma=0.99, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, lam=0.98,
        #                  entcoeff=0.0, cg_damping=1e-2, vf_stepsize=3e-4, vf_iters=3, verbose=0, tensorboard_log=None,
        #                  _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
        #                  seed=None, n_cpu_tf_sess=1
        model = SAC('MlpPolicy', env=env, verbose=1,
                    tensorboard_log=saver.data_dir,)

        # tensorboard
        # Make sure that your chrome browser is already on.
        # TensorboardLauncher(saver.data_dir + '/PPO2_1')

        # PPO run
        # Originally the total timestep is 5 x 10^8
        # 10 zeros for nupdates to be 4000
        # 1000000000 is 2000 iterations and so
        # 2000000000 is 4000 iterations.
        # obs = env.reset()  # 重置环境，获取初始观测
        # done = False
        # while not done:
        #     # actions, values, states, _ = model.step(obs)  # pytype: disable=attribute-error
        #     actions = env.sample_actions()
        #     print(actions.shape)
        #     print(f"Action: {actions}")
        #     obs, reward, done, info = env.step(actions)  # 执行动作
        #     print(f"Observation: {obs}, Reward: {reward}, Done: {done}")
        #     if done:
        #         obs = env.reset()  # 如果一回合结束，重置环境
        #
        # sys.exit(0)
        logger.configure(folder=saver.data_dir)
        model.learn(total_timesteps=250000)
        model.save(saver.data_dir)

    # # Testing mode with a trained weight
    else:
        model = TRPO.load(args.weight)
        test_model(env, model, render=args.render)


if __name__ == "__main__":
    main()
