from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from standup_logic import HumanoidStandupEnv

def make_env():
    return HumanoidStandupEnv(render_mode=None)

env = make_vec_env(make_env, n_envs=4)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0002,
)

model.learn(total_timesteps=3_000_000) # Converges at 3M timesteps
model.save("v14_standup_trained")
