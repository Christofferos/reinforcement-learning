import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class CustomCartPoleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.current_step = 0

        # ✅ Custom angle range
        self.theta_threshold_radians = 90 * math.pi / 180  # default 12 deg, custom 90 deg
        self.x_threshold = 5.0  # default 2.4

        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max,
        ], dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.state = None
        self.steps_beyond_terminated = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # START: position, velocity, angle, angular velocity
        self.state = self.np_random.uniform(low=-0.75, high=0.75, size=(4,)) 
        self.steps_beyond_terminated = None
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Combining force, pole’s inertia, and current pole velocity.
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass

        # Angular acceleration of the pole using Newton’s laws (nonlinear pendulum dynamics with a moving base)
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        
        # Acceleration of the cart — reaction force from the pole's movement affects the cart.
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration to numerically simulate the motion of a physical system
        x += self.tau * x_dot
        x_dot += self.tau * xacc
        theta += self.tau * theta_dot
        theta_dot += self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        terminated = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        max_episode_steps = 2500
        truncated = self.current_step >= max_episode_steps

        self.current_step += 1

        if not terminated:
            # Upright pole
            normalized_angle = abs(theta) / self.theta_threshold_radians  # normalized [0, 1]
            uprightness_reward = (1 - normalized_angle)

            # 🔄 Angular velocity — reward pole movement
            pole_swing_reward = min(abs(theta_dot) / 4.0, 1.0) # normalize to [0, 1]

            # 💨 Cart velocity — encourage swinging (but cap it)
            movement_reward = min(abs(x_dot) / 2.0, 1.0) # normalize to [0, 1]

            # ⚡️ Kinetic energy
            cart_ke = 0.5 * self.masscart * x_dot ** 2
            pole_ke = 0.5 * self.masspole * (self.length * theta_dot) ** 2
            kinetic_energy_reward = (cart_ke + pole_ke) * 0.01  # scale down

            swing_reward = 1.0 - abs(np.sign(x_dot) - np.sign(theta_dot)) * 0.5

            # Encouragement to swing upright from extreme angles
            if abs(theta) > math.radians(20):
                direction_reward = -np.sign(theta) * theta_dot  # +ve if correcting
                swing_upright_reward = max(direction_reward, 0.0)
            else:
                swing_upright_reward = 0.0

            # --- Reward cart for being near center ---
            position_penalty = abs(x) / self.x_threshold
            position_reward = 1.0 - position_penalty

            # 🎯 Combine them
            reward = (
                0.1 * uprightness_reward +     # balance
                0.1 * swing_upright_reward + 
                0.3 * position_reward +
                0.2 * pole_swing_reward +      # swing
                0.2 * swing_reward + 
                0.05 * movement_reward +        # cart movement
                0.05 * kinetic_energy_reward   # total motion
            )
            
            # max, min to ensure reward is in [0, 1]
            reward = max(reward, 0.0)
        else:
            reward = 0.0
            self.current_step = 0

        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return
    
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 300  # Top of the cart
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        x = self.state[0]
        cartx = int(x * scale + screen_width / 2.0)

        # Draw cart
        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            pygame.Rect(cartx - cartwidth / 2, carty - cartheight / 2, cartwidth, cartheight),
        )

        # Draw pole
        theta = self.state[2]
        pole_x = cartx
        pole_y = carty - cartheight / 2
        end_x = pole_x + polelen * math.sin(theta)
        end_y = pole_y - polelen * math.cos(theta)
        pygame.draw.line(
            self.screen, (204, 77, 77), (pole_x, pole_y), (end_x, end_y), int(polewidth)
        )

        # Handle pygame window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
