import gymnasium as gym
from cheetah_logic import HalfCheetahEnv
from gymnasium.envs.registration import register
from stable_baselines3 import PPO

register(
    id="CustomCheetah-v1",
    entry_point="cheetah_logic:HalfCheetahEnv",
    max_episode_steps=1000,
)
env = gym.make("CustomCheetah-v1", render_mode=None, 
               forward_reward_weight=1.0,
               ctrl_cost_weight=0.1, 
               reset_noise_scale=0.1,
               exclude_current_positions_from_observation=True,
               width=1200, height=800)

print(env)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=5_000_000)
model.save("custom_half_cheetah_v6")

env.close()
