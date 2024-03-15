# FlightRL - Reinforcement Learning for Flightmare
Flightmare is a powerful quadrotor simulator, but there are too many bugs in the installation process. This project is to provide a instruction to install  Flightmare and use it to train the RL controller.
Also, I have added some of my own code to train the model. Few of them is edited base on Stable Baselines, and rest of them is written by myself. As the result, some of the code is not perfect, but it is still a good start for the beginners to learn how to use Flightmare to train the RL controller.


# Install with pip


## Prerequisites

If you have not done so already, please install the following packages:

```bash
sudo apt-get update && apt-get install -y --no-install-recommends \
   build-essential \
   cmake \
   libzmqpp-dev \
   libopencv-dev
```
## Python environment
It is a good idea to use virtual environments (virtualenvs) or Anaconda to make sure packages from different projects do not interfere with each other. Check here for Anaconda installation.

1. To create an environment with python3.6

```bash
conda create --name ENVNAME python=3.6
```
2. Activate a named Conda environment
    
```bash
conda activate ENVNAME
```
## Install Flightmare(Or directly using this project)
```bash
cd ~/Desktop
git clone https://github.com/uzh-rpg/flightmare.git
```
## Add Environment Variable
Add FLIGHTMARE_PATH environment variable to your .bashrc file:
    
```bash
echo "export FLIGHTMARE_PATH=~/Desktop/flightmare" >> ~/.bashrc
source ~/.bashrc
```
## Install dependencies
```bash
conda activate ENVNAME
cd flightmare/
# install tensorflow GPU (for non-gpu user, use pip install tensorflow==1.14)
pip install tensorflow-gpu==1.14
 
# install scikit
pip install scikit-build
```
## Install FlightLib
```bash
cd flightmare/flightlib
# it first compile the flightlib and then install it as a python package.
pip install .
```
## Flightmare Bug Fix
Bug 1: Build wheel for flightgym error:
This is a known issue with the current version of flightgym. To fix this, you need to modify gtest_download.cmake
```bash
# change gtest_download.cmake
cd ~/Desktop/RL_Algorithms-main/flightlib/cmake
#change the line 8
#from   
GIT_TAG           master
#to    
GIT_TAG           main
```
Bug 2: Build wheel for OpenAI gym error:
You need to manually install the gym package from the source code.  3.4.16-dev is tested to work with the current 
version of flightgym.

```bash
pip install opencv-python==3.4.16-dev
```
bug 3: Build wheel for rpg-baseline error:
```bash
cd /path/to/flightrl 
# change line 20 in setup.py to
packages=['rpg_baselines','rpg_baselines.ppo','rpg_baselines.common','rpg_baselines.envs']
```
Generally the observed errors can be fixed by methods above, but some of my teammates meet the problem that the 
flightrender is black. There is no clue to fix it.

After installing the flightmare, you need to build flightrl
```bash
conda activate ENVNAME
cd /path/to/flightmare/flightrl
pip install .
```
Train neural network controller using PPO
```bash
cd examples
python3 run_drone_control.py --train 1
```
Test a pre-trained neural network controller
```bash
cd examples
python3 run_drone_control.py --train 0
```
With Unity Rendering
To use unity rendering, you need first download the binary from Releases and extract it into the flightrender folder. To enable unity for visualization, double click the extracted executable file RPG_Flightmare.x84-64 and then test a pre-trained controller
```bash
cd examples
python3 run_drone_control.py --train 0 --render 1
```

# If you are using my project, you can directly use the following command to train the model
```bash
cd /path/to/flightrl/On-Policy
python3 On_policy_control.py --train 1 -a PPO
```
default model is PPO, A2C to train the model. 
For off-policy, you can use the following command to train the model
```bash
cd /path/to/flightrl/Off-Policy
python3 Off_policy_control.py --train 1 -a TD3
```
default model is SAC, you can use TD3 or DDPG to train the model.

You can use the following command to test the model
```bash
python3 Off_policy_control.py --train 0 --render 1  -a TD3 -w actor_TD3.pth1005.0
```
or
```bash
python3 On_policy_control.py --train 0 --render 1  -a A2C -w 2024-03-14-23-26-57_Iteration_4319.zip
```
The -w is the filename to the model you want to test.

Caution: The training process for off-policy algorithms is **EXTREMELY** time-consuming. Even I installed CUDA, but there is no significant improvement.
My computer is equipped with following hardware:
- CPU: Intel(R) Core(TM) i9-13900K CPU
- GPU: NVIDIA GeForce RTX 4090
- RAM: 32GB
- OS: Ubuntu 20.04
It takes me about 2 hours for 1500 episodes. There are 2 options for this problem:
- When the return increasing to about 100, manually stop the training process. I have already add the save model function in the code, so you stop the training process anytime you want.
- Uncomment the line 63-65 in rl_utils.py to add a time limit for the training process. The default time limit is 20s,

There are many dependencies for this project, here are some of them:
- Python: 3.6
- OpenCV: 3.4.16-dev
- Tensorflow: 1.14 (1.15 for tensorboard in off-policy)
- Sometimes flightmare requires you to manually install the Zmqpp package
- Sometimes flightmare requires you to manually install the ruamel package
- CUDA: 11.3
- Torch: 1.10.2+cu113



