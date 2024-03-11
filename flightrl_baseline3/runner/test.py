from ruamel.yaml import YAML, dump, RoundTripDumper

#
import os
import math
import argparse
import numpy as np
import sys
import os

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))




#from rpg_baselines.envs import vec_env_wrapper as wrapper
from flightgym import QuadrotorEnv_v1


cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                           "/flightlib/configs/vec_env.yaml", 'r'))

a = QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
import pdb; pdb.set_trace()
env = wrapper.FlightEnvVec(QuadrotorEnv_v1(
    dump(cfg, Dumper=RoundTripDumper), False))

import pdb; pdb.set_trace()
print(env.observation_space)