from runtd3 import td3_api
from runsac import sac_api
from runddpg import ddpg_api
from rundqn import dqn_api
from runppo import ppo_api

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1,
                        help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=0,
                        help="Enable Unity Render")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--save_interval', type=int, default=1000000,
                        help="save_interval")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('-w', '--weight', type=str, default='./saved/2024-03-07-01-20-29/2024-03-07-01-20-29_7000000.zip',
                        help='trained weight path')
    parser.add_argument('--method', type=str, default='td3',
                        help='training method, support td3, ddpg, ppo, sac, dqn')
    return parser

def main():
    method_list = ['td3', 'ddpg', 'ppo', 'sac', 'dqn']
    args = parser().parse_args()
    if args.method not in method_list:
        raise ValueError('Invalid method: %s' % args.method)
    if args.method == 'td3':
        td3_api(args.train, args.render, args.save_interval, args.seed, args.weight)
    if args.method == 'sac':
        sac_api(args.train, args.render, args.save_interval, args.seed, args.weight)
    if args.method == 'ddpg':
        ddpg_api(args.train, args.render, args.save_interval, args.seed, args.weight)
    if args.method == 'dqn':
        dqn_api(args.train, args.render, args.save_interval, args.seed, args.weight)
    if args.method == 'ppo':
        ppo_api(args.train, args.render, args.save_interval, args.seed, args.weight)

if __name__ == "__main__":
    main()
