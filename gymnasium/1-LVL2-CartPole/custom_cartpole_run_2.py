import gymnasium as gym
from custom_cartpole import CustomCartPoleEnv

env = CustomCartPoleEnv(render_mode="human")
num_episodes = 200

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  # random action for demo
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()
    
    print(f"Episode {episode + 1} finished with total reward: {total_reward}")

env.close()
