import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordEpisodeStatistics

# Load the environment with rendering enabled
env = gym.make("CarRacing-v3", render_mode="human")
env = RecordEpisodeStatistics(env)

# Load the trained PPO model
model = PPO.load("car_racing_ppo_v1")

NUM_EVAL_EPISODES = 5

for episode in range(NUM_EVAL_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Episode {episode + 1} reward: {total_reward:.2f}")

env.close()
