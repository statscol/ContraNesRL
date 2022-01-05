import gym
import os
from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
from Contra.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv,VecNormalize
import argparse
from train import SAVE_PATH,env
from gym.wrappers.monitoring.video_recorder import VideoRecorder
#path_project = os.path.abspath(os.path.join(__file__, ".."))
#path_of_video_with_name = os.path.join(path_project, "contra_ppo_model.mp4")

if __name__=="__main__":
    model = PPO.load(SAVE_PATH)
 #   video_recorder = None
 #   video_recorder = VideoRecorder(env, path_of_video_with_name, enabled=True)
    obs = env.reset()
    game=True
    while game:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
  #      video_recorder.capture_frame()
        if done:
            obs = env.reset()
            game=False
   #         video_recorder.close()
   #         video_recorder.enabled = False
            env.close()
