import gymnasium as gym
from car_racing_logic import CarRacing
import numpy as np

# Create environment
env = CarRacing(render_mode="human")
obs, info = env.reset()

done = False
step_count = 0
total_reward = 0.0

while not done:
    # Sample a random action from the action space
    action = env.action_space.sample()

    # Step through the environment
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    step_count += 1

    done = terminated or truncated

print(f"✅ Episode finished after {step_count} steps with total reward: {total_reward:.2f}")
env.close()
