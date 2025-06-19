from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from spider_logic import SpiderEnv

def make_env():
    return SpiderEnv(render_mode=None)

env = make_vec_env(make_env, n_envs=4)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
)

model.learn(total_timesteps=3_000_000) # 2M timesteps is a good start for training a model on Spider
model.save("spider_trained_v4")
