import gymnasium as gym
from custom_cartpole import CustomCartPoleEnv
from stable_baselines3 import PPO

env = CustomCartPoleEnv(render_mode="human")
model = PPO.load("ppo_custom_cartpole_model", env=env)

NUM_EVAL_EPISODES = 5

for ep in range(NUM_EVAL_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        total_reward += reward
        done = terminated or truncated
    print(f"Episode {ep + 1} reward: {total_reward}")

env.close()
