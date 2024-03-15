#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper

#
import os
import math
import argparse
import numpy as np
import tensorflow as tf
from io import StringIO

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



    # parser.add_argument('-w', '--weight', type=str, default='./saved_sac/960iteration_target=0.02_withclip.zip',
    #                    help='trained weight path')

    # parser.add_argument('-w', '--weight', type=str, default='./saved_sac/1012Itration_no_clip.zip',
    #                     help='trained weight path')
    # parser.add_argument('-w', '--weight', type=str, default='./saved_sac/Iteration_2701_withclip_overfit.zip',
    #                      help='trained weight path')
    parser.add_argument('-w', '--weight', type=str, default='./saved/960iteration_target=0.02_withclip.zip',
                         help='trained weight path')
    return parser


import numpy as np
from PID_Controller import PIDController
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

    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(cfg_dump.getvalue(), False))

    # set random seed
    configure_random_seed(args.seed, env=env)

    #
    if args.train:
        # save the configuration and other files

        target_height = 5.0
        pid_controller_z = PIDController(P=0.1, I=0.01, D=0.001, target=target_height)
        pid_controller_x = PIDController(P=0.1, I=0.01, D=0.001, target=0)
        pid_controller_y = PIDController(P=0.1, I=0.01, D=0.001, target=0)
        num_episodes = 20  # 总共运行的episode数量
        env.connectUnity()  # 建立Unity渲染连接
        for i_episode in range(num_episodes):
            obs = env.reset()  # 重置环境状态
            total_reward = 0
            done = False
            while not done:
                # 使用PID控制器和转换函数控制无人机
                current_z = obs[0][2]  # 假设第三个值是无人机的高度
                current_x = obs[0][0]  # X轴位置
                current_y = obs[0][1]  # Y轴位置
                z_signal = pid_controller_z.update(current_z)
                x_signal = pid_controller_x.update(current_x)
                y_signal = pid_controller_y.update(current_y)
                action = np.array([0] * 4).reshape(1, 4).astype(np.float32)

                obs, reward, done, info = env.step(action)  # 执行动作并获取环境反馈
                total_reward += reward

                # 渲染当前环境状态
                # env.render()  # 如果env支持render()方法
                # 或者使用其他特定于环境的渲染方法

                if done:
                    print(f"Episode {i_episode + 1}: Total rewards: {total_reward}.")
                    break


if __name__ == "__main__":
    main()
