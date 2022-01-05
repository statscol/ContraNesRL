import gym

from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
from Contra.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv,VecNormalize
import argparse
from train import SAVE_PATH,env


if __name__=="__main__":
    model = PPO.load(SAVE_PATH)

    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()