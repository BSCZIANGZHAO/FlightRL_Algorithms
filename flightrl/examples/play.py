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
from rpg_baselines.envs import vec_env_wrapper as wrapper
import rpg_baselines.common.util as U
#
from flightgym import QuadrotorEnv_v1

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', type=str, default='./saved/quadrotor_env.zip',
                        help='trained weight path')
    return parser

args = parser().parse_args()
yaml = YAML(typ='unsafe', pure=True)
cfg_path = os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/vec_env.yaml"
with open(cfg_path, 'r') as file:
    cfg = yaml.load(file)

cfg["env"]["num_envs"] = 1
cfg["env"]["num_threads"] = 1
cfg["env"]["render"] = "yes"

cfg_dump = StringIO()
yaml.dump(cfg, cfg_dump)
cfg_dump.seek(0)  # Rewind to the beginning of the StringIO object

env = wrapper.FlightEnvVec(QuadrotorEnv_v1(cfg_dump.getvalue(), False))

model = PPO2.load(args.weight)

env.connectUnity()

obs, done, ep_len = env.reset(), False, 0
while not (done or (ep_len >= 300)):
    act, _ = model.predict(obs, deterministic=True)
    print(act)
    obs, rew, done, infos = env.step(act)

env.disconnectUnity()