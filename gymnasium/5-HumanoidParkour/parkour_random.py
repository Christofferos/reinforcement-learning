import gymnasium as gym
from parkour_logic import HumanoidParkourEnv
import time

env = HumanoidParkourEnv(render_mode="human", width=1200, height=800)

NUM_EPISODES = 20

for ep in range(NUM_EPISODES):
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0.0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        time.sleep(0.02)
        # time.sleep(0.1)  # Slow down the rendering
        done = terminated or truncated
    print(f"✅ Episode finished after {step_count} steps with total reward: {total_reward:.2f}")

env.close()
