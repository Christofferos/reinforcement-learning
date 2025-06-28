__credits__ = ["Kallinteris-Andreas"]

import numpy as np
import mujoco

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import os

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((1.60, 0.40, 0.3925)),
    "elevation": -40.0,
}


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:3].copy()


class HumanoidParkourEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "parkour.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.25,
        ctrl_cost_weight: float = 0.1,
        contact_cost_weight: float = 5e-7,
        contact_cost_range: tuple[float, float] = (-np.inf, 10.0),
        healthy_reward: float = 0.1,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple[float, float] = (-2, 5.0),
        reset_noise_scale: float = 2e-2,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        xml_path = os.path.join(os.path.dirname(__file__), "parkour.xml") # "parkour.xml" # "parkour_dojo.xml"
        utils.EzPickle.__init__(self, xml_path, frame_skip, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self.total_reward = 0.0

        # === Predefined Parkour Path ===
        self.parkour_path = [
            np.array([2.5, 0.0, 1.2]),
            np.array([3.9, 0.0, 1.2]),      # platform_jump_1
            np.array([6.5, 0.0, 1.2]),      # platform_jump_2
            np.array([8.9, 0.0, 1.2]),      # platform_jump_3
            np.array([12.8, 0.0, 1.6]),     # platform_incline
            np.array([15, 0.0, 2.15]),     # platform_incline top
            np.array([16, 0.0, 2.15]),     # platform_decline start 
            np.array([18.5, 0.0, 1.0]),     # platform_decline
            np.array([20.5, 0.0, 0.8]),
            np.array([23.7, 0.0, 0.5]),     # platform_landing
            np.array([26, 0.0, 0.65]),
            np.array([27.0, 0.0, 0.80]),
            np.array([28.5, 0.0, 0.95]),     # platform_stair_3 (end of stairs)
            np.array([32.0, 1.0, 1.5]),     # ramp
            np.array([35.5, 0.0, 1.95]),     # high_platform
            np.array([35.5, -4.5, 1.60]),    # high_platform2
            np.array([33.0, -4.5, 1.60]),    # balance_beam
            np.array([30.5, -4.5, 1.60]),   # vault_wall_platform
            np.array([29.0, -4.5, 1.85]),
            np.array([27.0, -4.5, 1.50]),    # jump_1 (second set)
            np.array([25.8, -3.5, 1.4]),    # jump_2 (second set)
            np.array([24.6, -4.5, 1.3]),    # jump_3 (second set)
            np.array([23.4, -5.5, 1.2]),    # jump_4 (second set)
            np.array([21.0, -5.0, 1.1]),    # finish_platform
        ]
        self.current_target_index = self.parkour_path.__len__() - 1
        self.target_point_xyz = self.parkour_path[self.current_target_index]
        self.previous_distance_to_target = 0.0
        self.target_points_reached = 1

        self.next_target_point_xyz = self.parkour_path[self.current_target_index]

        self.truncation_timer = 0
        self.idle_counter = 0

        # 10-Ray Downward Array Setup
        self.num_rays = 10
        self.ray_length = 5.0
        # Spacing of the array in front of the agent
        self.line_x_spacing = 0.2
        # Offset the array from the torso
        self.line_x_offset = -0.3  # How far the first ray is in front
        self.line_z_offset = 0.0 # **FIX: Start rays from torso's vertical center**
        
        # Generate names for the visualization geoms
        self.ray_viz_geom_names = [f"ray_viz_{i}" for i in range(self.num_rays)]


        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        
        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            + self.data.cinert[1:].size
            + self.data.cvel[1:].size
            + self.data.qfrc_actuator.size
            + self.data.cfrc_ext[1:].size
        )
        if self._exclude_current_positions_from_observation:
            obs_size -= 2
            
        obs_size += self.num_rays
        obs_size += self.num_rays * 3 # NEW: Add space for the 3D normal vector of each ray
        obs_size += 3

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )
        
    def _get_obs(self, horizontal_velocity):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        com_inertia = self.data.cinert[1:].flatten()
        com_velocity = self.data.cvel[1:].flatten()
        actuator_forces = self.data.qfrc_actuator.flatten()
        external_contact_forces = self.data.cfrc_ext[1:].flatten()
        
        current_agent_xyz = self.data.qpos[0:3]
        ray_distances, _, _, ray_normals = self._get_ray_data(horizontal_velocity)
        vector_to_target_1 = self.target_point_xyz - current_agent_xyz

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        # length of new observational values === 29
        # total_len = vector_to_target_1.__len__() + vector_to_target_2.__len__() + self.previous_action.__len__() + forward_vec.__len__() + up_vec.__len__()
        # print(total_len)

        return np.concatenate(
            [
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
                ray_distances,
                ray_normals.flatten(),
                vector_to_target_1,
            ]
        )
    
    def _generate_new_practice_target(self, baseline_point = None):
        """Generates a new random target point for the agent to reach."""
        min_radius = 0.85 # 3
        max_radius = 3.5 # 3.1
        
        radius = self.np_random.uniform(low=min_radius, high=max_radius)
        angle = self.np_random.uniform(low=-np.pi, high=np.pi)
        
        # Set the new target point
        current_agent_xyz = self.data.qpos[0:3]
        if baseline_point is not None:
            current_agent_xyz = baseline_point

        # Calculate the offset from the agent's current position
        offset_x = radius * np.cos(angle)
        offset_y = radius * np.sin(angle)

        return [current_agent_xyz[0] + offset_x, current_agent_xyz[1] + offset_y, np.random.uniform(0.6, 1.4)]
    
    def _generate_new_target(self):
        """Sets the next target point from the predefined parkour path."""
        # Move to the next point in the path
        self.current_target_index += 1
        
        # Make sure we don't go out of bounds. If we're at the end, just keep the last point as the target.
        if self.current_target_index >= len(self.parkour_path):
            self.current_target_index = len(self.parkour_path) - 1
    
        self.target_point_xyz = self.parkour_path[self.current_target_index]
        self.next_target_point_xyz = self.parkour_path[min(self.current_target_index + 1, len(self.parkour_path) - 1)]

    def _get_ray_data(self, horizontal_vel):
        distances = np.full(self.num_rays, self.ray_length)
        normals = np.tile(np.array([0., 0., 1.]), (self.num_rays, 1))

        ray_start_positions = np.zeros((self.num_rays, 3))
        pelvis_pos = self.data.body('pelvis').xpos
        speed = np.linalg.norm(horizontal_vel)
        velocity_threshold = 0.1
 
        orientation_matrix = self.data.body('pelvis').xmat.reshape(3, 3) # --- Fallback Case: Agent is not moving ---
        if speed >= velocity_threshold:
            forward_dir = horizontal_vel / speed
            up_dir = np.array([0., 0., 1.])
            side_dir = np.cross(up_dir, forward_dir)
            orientation_matrix = np.column_stack([forward_dir, side_dir, up_dir])
        
        ray_dir_down = np.array([0, 0, -self.ray_length])
        normal_calc_offset = 0.05 

        ENVIRONMENT_GROUP = 1
        geomgroup_flags = np.zeros(6, dtype=np.uint8)
        geomgroup_flags[ENVIRONMENT_GROUP] = 1

        for i in range(self.num_rays):
            # --- Step 1: Find the central hit point (as before) ---
            base_offset = np.array([self.line_x_offset + i * self.line_x_spacing, 0, self.line_z_offset])
            world_offset = orientation_matrix @ base_offset
            ray_start = pelvis_pos + world_offset
            ray_start_positions[i] = ray_start # For visualization

            dist = mujoco.mj_ray(self.model, self.data, ray_start, ray_dir_down, geomgroup_flags, True, -1, np.array([-1], dtype=np.int32))

            if dist != -1 and dist < 1.0: # Only calculate normal if ground is reasonably close
                distances[i] = dist * self.ray_length
                
                # --- Step 2: Find two more points nearby to define a plane ---
                p0 = ray_start + ray_dir_down * dist # Central hit point

                # Point 1: slightly forward
                offset_p1 = np.array([normal_calc_offset, 0, 0])
                world_offset_p1 = orientation_matrix @ offset_p1
                ray_start_p1 = ray_start + world_offset_p1
                dist_p1 = mujoco.mj_ray(self.model, self.data, ray_start_p1, ray_dir_down, geomgroup_flags, True, -1, np.array([-1], dtype=np.int32))
                
                # Point 2: slightly to the side
                offset_p2 = np.array([0, normal_calc_offset, 0])
                world_offset_p2 = orientation_matrix @ offset_p2
                ray_start_p2 = ray_start + world_offset_p2
                dist_p2 = mujoco.mj_ray(self.model, self.data, ray_start_p2, ray_dir_down, geomgroup_flags, True, -1, np.array([-1], dtype=np.int32))

                if dist_p1 != -1 and dist_p2 != -1:
                    # We have three points, now calculate the normal
                    p1 = ray_start_p1 + ray_dir_down * dist_p1
                    p2 = ray_start_p2 + ray_dir_down * dist_p2

                    # Create two vectors on the plane
                    v1 = p1 - p0
                    v2 = p2 - p0

                    # The cross product gives the normal vector
                    cross_product = np.cross(v1, v2)
                    norm_magnitude = np.linalg.norm(cross_product)
                    
                    if norm_magnitude > 1e-6: # Avoid division by zero
                        calculated_normal = cross_product / norm_magnitude
                        # Ensure normal points upwards
                        if calculated_normal[2] < 0:
                            calculated_normal = -calculated_normal
                        normals[i] = calculated_normal
                    
        return distances, ray_start_positions, ray_dir_down, normals
    
    def _reward_upright(self):
        """Reward for keeping the torso upright."""
        torso_up_vector = self.data.xmat[self.model.body('torso').id].reshape(3, 3)[:, 2]
        pelvis_up_vector = self.data.xmat[self.model.body('pelvis').id].reshape(3, 3)[:, 2]
        world_up_vector = np.array([0, 0, 1])
        alignment = np.dot(torso_up_vector, world_up_vector)
        alignment2 = np.dot(pelvis_up_vector, world_up_vector)
        upright_scale = 0.5
        return upright_scale * ((max(0, alignment)**3) + (max(0, alignment2)**3)) / 2
    
    def center_of_mass_offset_penalty(self, xy_position_after):
        left_foot_xy = self.data.body('left_foot').xpos[0:2]
        right_foot_xy = self.data.body('right_foot').xpos[0:2]
        support_center_xy = (left_foot_xy + right_foot_xy) / 2.0
        com_error = np.linalg.norm(xy_position_after - support_center_xy)
        reward_com_stability = -10 * (com_error ** 2)
        return reward_com_stability

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z
        torso_orientation_matrix = self.data.body("torso").xmat.reshape(3, 3)
        local_up_vector = np.array([0, 0, 1])
        world_up_vector = torso_orientation_matrix @ local_up_vector
        upright_z_value = world_up_vector[2]
        min_upright_z = 0.25
        is_orientation_healthy = upright_z_value > min_upright_z
        pelvis_pos = self.data.body("pelvis").xpos
        torso_pos = self.data.body("torso").xpos
        is_laydown = (abs(torso_pos[2]) - abs(pelvis_pos[2])) < 0
        return is_healthy and is_laydown == False # and is_orientation_healthy

    def step(self, action):
        xyz_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xyz_position_after = mass_center(self.model, self.data)

        xyz_velocity = (xyz_position_after - xyz_position_before) / self.dt
        x_velocity, y_velocity, z_velocity = xyz_velocity

        progress_reward_scale = 150.0
        reach_bonus_reward = 10.0
        reach_threshold = 0.75
        current_distance_to_target = np.linalg.norm(self.target_point_xyz - xyz_position_after)
        progress_reward = progress_reward_scale * (self.previous_distance_to_target - current_distance_to_target)
        target_reached_bonus = 0.0
        if current_distance_to_target < reach_threshold:
            target_reached_bonus = reach_bonus_reward
            self.target_points_reached += 1
            # self.target_point_xyz = self.next_target_point_xyz # PRACTICE
            # self.next_target_point_xyz = self._generate_new_practice_target(self.target_point_xyz) # PRACTICE
            self._generate_new_target() # REAL DEAL
            
        self.previous_distance_to_target = np.linalg.norm(self.target_point_xyz - xyz_position_after) # Update
        
        horizontal_velocity = [x_velocity, y_velocity, 0]
        ray_distances, ray_start_positions, ray_dir, ray_normals = self._get_ray_data(horizontal_velocity)
        
        alive_bonus = self._healthy_reward if self.is_healthy else 0.0
        upright_reward = self._reward_upright()
        center_of_mass_offset_penalty = self.center_of_mass_offset_penalty(xyz_position_after[0:2])
        # centre_position_reward = calc y and z distance and reward walking in line and height of targets

        control_cost = -self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))

        fall_facing_penalty = -0.1 if ray_distances[9] > 3.5 else 0

        idle_penalty = 0
        idle_speed_threshold = 0.25
        idle_max_tolerance = 250
        if abs(x_velocity) <= idle_speed_threshold and abs(y_velocity) <= idle_speed_threshold:
            idle_penalty = -0.1
            self.idle_counter += 1
        else:
            self.idle_counter = 0
        
        fall_penalty = 0.0
        if not self.is_healthy or self.idle_counter >= idle_max_tolerance:
            fall_penalty = -200.0 / self.target_points_reached
        
        step_reward = (
            target_reached_bonus +
            progress_reward +
            upright_reward +
            center_of_mass_offset_penalty +
            idle_penalty
        )

        """ print("REWARDS", 
                target_reached_bonus,
                progress_reward,
                upright_reward,
                center_of_mass_offset_penalty,
                idle_penalty) """

        
        if self.render_mode == "human":
            for i in range(self.num_rays):
                start_point = ray_start_positions[i]
                distance = ray_distances[i]
                
                end_point = start_point + ray_dir * distance
                geom_name = self.ray_viz_geom_names[i]
                geom_id = self.model.geom(geom_name).id

                self.data.geom_xpos[geom_id] = (start_point + end_point) / 2
                self.model.geom_size[geom_id][1] = distance / 2
                
                # Since rays are straight down, rotation matrix is identity
                self.data.geom_xmat[geom_id] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
            self.data.site_xpos[self.model.site('target_marker').id] = self.target_point_xyz
            self.data.site_xpos[self.model.site('target_marker_next').id] = self.next_target_point_xyz
            self.render()

        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        self.total_reward += step_reward
        truncation = False
        self.truncation_timer += 1
        if self.idle_counter >= idle_max_tolerance:
            terminated = True
            print("IDLING FAIL.", "TARGETS: ", self.target_points_reached, "REWARD:", np.trunc(self.total_reward))
            self.total_reward = 0
        if self.truncation_timer >= (15000 / self.frame_skip):
            print("Truncated. ", "TARGETS: ", self.target_points_reached, "VEL:", x_velocity if x_velocity > 0.01 else 0, y_velocity if y_velocity > 0.01 else 0, "REWARD: ", np.trunc(self.total_reward))
            truncation = True
            self.total_reward = 0
            self.truncation_timer = 0
        if terminated:
            print("Fall.", "TARGETS: ", self.target_points_reached, "REWARD:", np.trunc(self.total_reward))
            self.total_reward = 0
        info = {
            "x_position": xyz_position_after[0],
            "y_position": xyz_position_after[1],
            "distance_from_origin": np.linalg.norm(xyz_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
        }

        observation = self._get_obs(horizontal_velocity)
        self.previous_action = action
        return observation, step_reward, terminated, truncation, info

    def reset_model(self):
        self.target_points_reached = 1
        self.total_reward = 0.0
        self.truncation_timer = 0
        self.idle_counter = 0

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        # --- START: Random Orientation Logic --- PRACTICE WALKING
        """ random_yaw_angle = self.np_random.uniform(low=-np.pi, high=np.pi)
        random_orientation_quat = np.array([np.cos(random_yaw_angle / 2), 0, 0, np.sin(random_yaw_angle / 2)])
        qpos[3:7] = random_orientation_quat """
        # --- END: Random Orientation Logic ---

        self.current_target_index = -1 # -1
        self._generate_new_target() # REAL DEAL
        # self.target_point_xyz = self._generate_new_practice_target() # PRACTICE WALKING
        # self.next_target_point_xyz = self._generate_new_practice_target(self.target_point_xyz) # PRACTICE WALKING

        current_agent_xyz = self.data.qpos[0:3]
        self.previous_distance_to_target = np.linalg.norm(self.target_point_xyz - current_agent_xyz)

        self.set_state(qpos, qvel)

        observation = self._get_obs([0,0,0])
        return observation