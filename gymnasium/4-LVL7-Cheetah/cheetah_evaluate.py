import gymnasium as gym
from cheetah_logic import HalfCheetahEnv
from stable_baselines3 import PPO
import time

env = HalfCheetahEnv(render_mode="human", frame_skip=1, width=1200, height=800)
model = PPO.load("custom_half_cheetah_v5", env=env)

NUM_EPISODES = 3

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # time.sleep(0.2)  # Slow down the rendering
        total_reward += reward

    print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}")

env.close()
