import gymnasium as gym
from stable_baselines3 import PPO
from car_racing_logic import CarRacing

env = CarRacing(render_mode="human", domain_randomize=False, continuous=False) # render mode "state_pixels"?? or "human"
model = PPO.load("custom_car_racing_model_v4", env=env)

NUM_EPISODES = 3

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=False) # For demo "deterministic=True"
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}")

env.close()
