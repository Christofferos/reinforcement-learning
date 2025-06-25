from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import TQC
from stable_baselines3.common.env_util import make_vec_env
from parkour_logic import HumanoidParkourEnv
import os
import argparse

model_dir = "models/TQC"
log_dir = "logs"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train(env, sb3_algorithm, path_to_model=None):
    if path_to_model:
        if not os.path.isfile(path_to_model):
            raise FileNotFoundError(f"Model file {path_to_model} does not exist.")
        print("Pre-trained model used: ", args.train)
        match sb3_algorithm:
            case "TQC":
                model = TQC.load(path_to_model, env=env)
            case "PPO":
                model = PPO.load(path_to_model, env=env)
            case "SAC":
                model = SAC.load(path_to_model, env=env)
            case _:
                raise ValueError(f"Unsupported algorithm: {sb3_algorithm}")
    else:
        print("New model")
        match sb3_algorithm:
            case "PPO":
                model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
            case "SAC":
                model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
            case "TQC":
                model = TQC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
            case _:
                raise ValueError(f"Unsupported algorithm: {sb3_algorithm}")
    TIMESTEPS = 200_000
    iterations = 0
    while True:
        iterations += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{sb3_algorithm}_{TIMESTEPS*iterations}")

def make_env():
    return HumanoidParkourEnv(render_mode=None)

if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    train_env = make_vec_env(make_env, n_envs=32) # 1 = Slow. 4 = Fast. 8 = Fastest --- (Mac M1 has 8 cores = 8 environments)
    # See stable_baselines3 supported algorithms: https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3') 
    parser.add_argument('-t', '--train', nargs='?', const=None, metavar='path_to_model')
    args = parser.parse_args()
    train(train_env, args.sb3_algo, path_to_model=args.train)

# python .\parkour_train.py TQC -t .\models\