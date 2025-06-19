# train_car_racing.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return gym.make("CarRacing-v3", render_mode=None)

env = DummyVecEnv([make_env])

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

model.save("car_racing_ppo_v2")
env.close()
