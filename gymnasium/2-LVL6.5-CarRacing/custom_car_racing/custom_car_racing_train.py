import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from car_racing_logic import CarRacing 

def make_env():
    return CarRacing(render_mode="human", domain_randomize=False, continuous=False) # state_pixels - to make AI see 96x96 pixels around the car

# Vectorized environment for stable-baselines
env = make_vec_env(make_env, n_envs=1)

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=200_000) # 200k timesteps is around 100 iterations == 1 hours
model.save("custom_car_racing_model_v5")

env.close()
