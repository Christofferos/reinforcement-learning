import gymnasium as gym
from gymnasium.envs.registration import register
from cheetah_logic import HalfCheetahEnv
import time

register(
    id="CustomCheetah-v1",
    entry_point="cheetah_logic:HalfCheetahEnv",
    max_episode_steps=1000,
)
env = gym.make("CustomCheetah-v1", render_mode="human", width=1200, height=800)

observation = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # take random actions
    observation, reward, terminated, truncated, info = env.step(action)
    # time.sleep(0.02)
    if terminated or truncated:
        observation = env.reset()

env.close()
