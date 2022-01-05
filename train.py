import gym

from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
from Contra.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv,VecNormalize
import argparse

env=gym.make('Contra-v0')
env=JoypadSpace(env,SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)


SAVE_PATH = './model/contra_ppn' 



def train(lr=0.00001,timesteps=1000000):
    model = PPO('CnnPolicy', env, verbose=0,clip_range=0.2,learning_rate=lr,n_steps=256) 
    model.learn(total_timesteps=timesteps)
    model.save(SAVE_PATH)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description="Contra PPO RL trainer")
    parser.add_argument("-lr","--learning_rate",help="Learning Rate",dest="learning_rate",type=float)
    parser.add_argument("-t","--time_steps",help="Time Steps",dest="time_steps",type=int)
    args=parser.parse_args()

    train(lr=float(args.learning_rate),timesteps=int(args.time_steps))
