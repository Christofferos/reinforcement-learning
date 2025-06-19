import gymnasium as gym
from car_racing_logic import CarRacing

class CustomCarRacing(CarRacing):
    def __init__(self, render_mode="human", **kwargs):
        super().__init__(render_mode=render_mode, **kwargs)
        print("Custom Car Racing Environment Initialized")

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.slow_counter = 0
        return super().reset(seed=seed, options=options)

    def render(self):
        return super().render()
