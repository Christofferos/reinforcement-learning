import gymnasium as gym
from custom_cartpole import CustomCartPoleEnv
from stable_baselines3 import PPO

env = CustomCartPoleEnv(render_mode=None)

# Model training
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=40_000)  # train for X steps
model.save("ppo_custom_cartpole_test") # Save model

env.close()
