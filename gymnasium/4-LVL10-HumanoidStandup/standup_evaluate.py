import gymnasium as gym
from standup_logic import HumanoidStandupEnv
from stable_baselines3 import PPO
import time
import argparse  # Import the argparse module

# 1. Set up argument parsing
parser = argparse.ArgumentParser(description="Evaluate a trained PPO model for HumanoidStandup.")
parser.add_argument("model_path", type=str,
                    help="Path to the trained PPO model (e.g., 'v3_standup_trained_fail')")
parser.add_argument("--episodes", type=int, default=3,
                    help="Number of episodes to run for evaluation (default: 3)")
args = parser.parse_args()

env = HumanoidStandupEnv(render_mode="human", frame_skip=1, width=1200, height=800) # width=1200, height=800 # 600 400
model = PPO.load(args.model_path, env=env)

NUM_EPISODES = args.episodes

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
