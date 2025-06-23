__credits__ = ["Kallinteris-Andreas"]

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import os

""" DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
} """

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((1.60, 0.40, 0.3925)),
    "elevation": -40.0,
}


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class HumanoidWalkEnv(MujocoEnv, utils.EzPickle):
    r"""
    ## Description
    This environment is based on the environment introduced by Tassa, Erez and Todorov in ["Synthesis and stabilization of complex behaviors through online trajectory optimization"](https://ieeexplore.ieee.org/document/6386025).
    The 3D bipedal robot is designed to simulate a human.
    It has a torso (abdomen) with a pair of legs and arms, and a pair of tendons connecting the hips to the knees.
    The legs each consist of three body parts (thigh, shin, foot), and the arms consist of two body parts (upper arm, forearm).
    The goal of the environment is to walk forward as fast as possible without falling over.


    ## Action Space
    ```{figure} action_space_figures/humanoid.png
    :name: humanoid
    ```

    The action space is a `Box(-0.4, 0.4, (17,), float32)`. An action represents the torques applied at the hinge joints.

    | Num | Action                                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
    | --- | ---------------------------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
    | 0   | Torque applied on the hinge in the y-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_y                        | hinge | torque (N m) |
    | 1   | Torque applied on the hinge in the z-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_z                        | hinge | torque (N m) |
    | 2   | Torque applied on the hinge in the x-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_x                        | hinge | torque (N m) |
    | 3   | Torque applied on the rotor between torso/abdomen and the right hip (x-coordinate) | -0.4        | 0.4         | right_hip_x (right_thigh)        | hinge | torque (N m) |
    | 4   | Torque applied on the rotor between torso/abdomen and the right hip (z-coordinate) | -0.4        | 0.4         | right_hip_z (right_thigh)        | hinge | torque (N m) |
    | 5   | Torque applied on the rotor between torso/abdomen and the right hip (y-coordinate) | -0.4        | 0.4         | right_hip_y (right_thigh)        | hinge | torque (N m) |
    | 6   | Torque applied on the rotor between the right hip/thigh and the right shin         | -0.4        | 0.4         | right_knee                       | hinge | torque (N m) |
    | 7   | Torque applied on the rotor between torso/abdomen and the left hip (x-coordinate)  | -0.4        | 0.4         | left_hip_x (left_thigh)          | hinge | torque (N m) |
    | 8   | Torque applied on the rotor between torso/abdomen and the left hip (z-coordinate)  | -0.4        | 0.4         | left_hip_z (left_thigh)          | hinge | torque (N m) |
    | 9   | Torque applied on the rotor between torso/abdomen and the left hip (y-coordinate)  | -0.4        | 0.4         | left_hip_y (left_thigh)          | hinge | torque (N m) |
    | 10  | Torque applied on the rotor between the left hip/thigh and the left shin           | -0.4        | 0.4         | left_knee                        | hinge | torque (N m) |
    | 11  | Torque applied on the rotor between the torso and right upper arm (coordinate -1)  | -0.4        | 0.4         | right_shoulder1                  | hinge | torque (N m) |
    | 12  | Torque applied on the rotor between the torso and right upper arm (coordinate -2)  | -0.4        | 0.4         | right_shoulder2                  | hinge | torque (N m) |
    | 13  | Torque applied on the rotor between the right upper arm and right lower arm        | -0.4        | 0.4         | right_elbow                      | hinge | torque (N m) |
    | 14  | Torque applied on the rotor between the torso and left upper arm (coordinate -1)   | -0.4        | 0.4         | left_shoulder1                   | hinge | torque (N m) |
    | 15  | Torque applied on the rotor between the torso and left upper arm (coordinate -2)   | -0.4        | 0.4         | left_shoulder2                   | hinge | torque (N m) |
    | 16  | Torque applied on the rotor between the left upper arm and left lower arm          | -0.4        | 0.4         | left_elbow                       | hinge | torque (N m) |


    ## Observation Space
    The observation space consists of the following parts (in order)

    - *qpos (22 elements by default):* The position values of the robot's body parts.
    - *qvel (23 elements):* The velocities of these individual body parts (their derivatives).
    - *cinert (130 elements):* Mass and inertia of the rigid body parts relative to the center of mass,
    (this is an intermediate result of the transition).
    It has shape 13*10 (*nbody * 10*).
    (cinert - inertia matrix and body mass offset and body mass)
    - *cvel (78 elements):* Center of mass based velocity.
    It has shape 13 * 6 (*nbody * 6*).
    (com velocity - velocity x, y, z and angular velocity x, y, z)
    - *qfrc_actuator (17 elements):* Constraint force generated as the actuator force at each joint.
    This has shape `(17,)`  *(nv * 1)*.
    - *cfrc_ext (78 elements):* This is the center of mass based external force on the body parts.
    It has shape 13 * 6 (*nbody * 6*) and thus adds another 78 elements to the observation space.
    (external forces - force x, y, z and torque x, y, z)

    where *nbody* is the number of bodies in the robot,
    and *nv* is the number of degrees of freedom (*= dim(qvel)*).

    By default, the observation does not include the x- and y-coordinates of the torso.
    These can be included by passing `exclude_current_positions_from_observation=False` during construction.
    In this case, the observation space will be a `Box(-Inf, Inf, (350,), float64)`, where the first two observations are the x- and y-coordinates of the torso.
    Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.

    By default, however, the observation space is a `Box(-Inf, Inf, (348,), float64)`, where the position and velocity elements are as follows:

    | Num | Observation                                                                                                     | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)                |
    | --- | --------------------------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | -------------------------- |
    | 0   | z-coordinate of the torso (centre)                                                                              | -Inf | Inf | root                             | free  | position (m)               |
    | 1   | w-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
    | 2   | x-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
    | 3   | y-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
    | 4   | z-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
    | 5   | z-angle of the abdomen (in lower_waist)                                                                         | -Inf | Inf | abdomen_z                        | hinge | angle (rad)                |
    | 6   | y-angle of the abdomen (in lower_waist)                                                                         | -Inf | Inf | abdomen_y                        | hinge | angle (rad)                |
    | 7   | x-angle of the abdomen (in pelvis)                                                                              | -Inf | Inf | abdomen_x                        | hinge | angle (rad)                |
    | 8   | x-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_x                      | hinge | angle (rad)                |
    | 9   | z-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_z                      | hinge | angle (rad)                |
    | 10  | y-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_y                      | hinge | angle (rad)                |
    | 11  | angle between right hip and the right shin (in right_knee)                                                      | -Inf | Inf | right_knee                       | hinge | angle (rad)                |
    | 12  | x-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_x                       | hinge | angle (rad)                |
    | 13  | z-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_z                       | hinge | angle (rad)                |
    | 14  | y-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_y                       | hinge | angle (rad)                |
    | 15  | angle between left hip and the left shin (in left_knee)                                                         | -Inf | Inf | left_knee                        | hinge | angle (rad)                |
    | 16  | coordinate-1 (multi-axis) angle between torso and right arm (in right_upper_arm)                                | -Inf | Inf | right_shoulder1                  | hinge | angle (rad)                |
    | 17  | coordinate-2 (multi-axis) angle between torso and right arm (in right_upper_arm)                                | -Inf | Inf | right_shoulder2                  | hinge | angle (rad)                |
    | 18  | angle between right upper arm and right_lower_arm                                                               | -Inf | Inf | right_elbow                      | hinge | angle (rad)                |
    | 19  | coordinate-1 (multi-axis) angle between torso and left arm (in left_upper_arm)                                  | -Inf | Inf | left_shoulder1                   | hinge | angle (rad)                |
    | 20  | coordinate-2 (multi-axis) angle between torso and left arm (in left_upper_arm)                                  | -Inf | Inf | left_shoulder2                   | hinge | angle (rad)                |
    | 21  | angle between left upper arm and left_lower_arm                                                                 | -Inf | Inf | left_elbow                       | hinge | angle (rad)                |
    | 22  | x-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
    | 23  | y-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
    | 24  | z-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
    | 25  | x-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | angular velocity (rad/s)   |
    | 26  | y-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | angular velocity (rad/s)   |
    | 27  | z-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | angular velocity (rad/s)   |
    | 28  | z-coordinate of angular velocity of the abdomen (in lower_waist)                                                | -Inf | Inf | abdomen_z                        | hinge | angular velocity (rad/s)   |
    | 29  | y-coordinate of angular velocity of the abdomen (in lower_waist)                                                | -Inf | Inf | abdomen_y                        | hinge | angular velocity (rad/s)   |
    | 30  | x-coordinate of angular velocity of the abdomen (in pelvis)                                                     | -Inf | Inf | abdomen_x                        | hinge | angular velocity (rad/s)   |
    | 31  | x-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_x                      | hinge | angular velocity (rad/s)   |
    | 32  | z-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_z                      | hinge | angular velocity (rad/s)   |
    | 33  | y-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_y                      | hinge | angular velocity (rad/s)   |
    | 34  | angular velocity of the angle between right hip and the right shin (in right_knee)                              | -Inf | Inf | right_knee                       | hinge | angular velocity (rad/s)   |
    | 35  | x-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_x                       | hinge | angular velocity (rad/s)   |
    | 36  | z-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_z                       | hinge | angular velocity (rad/s)   |
    | 37  | y-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_y                       | hinge | angular velocity (rad/s)   |
    | 38  | angular velocity of the angle between left hip and the left shin (in left_knee)                                 | -Inf | Inf | left_knee                        | hinge | angular velocity (rad/s)   |
    | 39  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) | -Inf | Inf | right_shoulder1                  | hinge | angular velocity (rad/s)   |
    | 40  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) | -Inf | Inf | right_shoulder2                  | hinge | angular velocity (rad/s)   |
    | 41  | angular velocity of the angle between right upper arm and right_lower_arm                                       | -Inf | Inf | right_elbow                      | hinge | angular velocity (rad/s)   |
    | 42  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm)   | -Inf | Inf | left_shoulder1                   | hinge | angular velocity (rad/s)   |
    | 43  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm)   | -Inf | Inf | left_shoulder2                   | hinge | angular velocity (rad/s)   |
    | 44  | angular velocity of the angle between left upper arm and left_lower_arm                                         | -Inf | Inf | left_elbow                       | hinge | angular velocity (rad/s)   |
    | excluded | x-coordinate of the torso (centre)                                                                         | -Inf | Inf | root                             | free  | position (m)               |
    | excluded | y-coordinate of the torso (centre)                                                                         | -Inf | Inf | root                             | free  | position (m)               |

    The body parts are:

    | body part       | id (for `v2`, `v3`, `v4)` | id (for `v5`) |
    |  -------------  |  ---   |  ---  |
    | worldbody (note: all values are constant 0) | 0  |excluded|
    | torso           |1  | 0      |
    | lwaist          |2  | 1      |
    | pelvis          |3  | 2      |
    | right_thigh     |4  | 3      |
    | right_sin       |5  | 4      |
    | right_foot      |6  | 5      |
    | left_thigh      |7  | 6      |
    | left_sin        |8  | 7      |
    | left_foot       |9  | 8      |
    | right_upper_arm |10 | 9      |
    | right_lower_arm |11 | 10     |
    | left_upper_arm  |12 | 11     |
    | left_lower_arm  |13 | 12     |

    The joints are:

    | joint           | id (for `v2`, `v3`, `v4)` | id (for `v5`) |
    |  -------------  |  ---   |  ---  |
    | root (note: all values are constant 0) | 0  |excluded|
    | root (note: all values are constant 0) | 1  |excluded|
    | root (note: all values are constant 0) | 2  |excluded|
    | root (note: all values are constant 0) | 3  |excluded|
    | root (note: all values are constant 0) | 4  |excluded|
    | root (note: all values are constant 0) | 5  |excluded|
    | abdomen_z       | 6  | 0      |
    | abdomen_y       | 7  | 1      |
    | abdomen_x       | 8  | 2      |
    | right_hip_x     | 9  | 3      |
    | right_hip_z     | 10 | 4      |
    | right_hip_y     | 11 | 5      |
    | right_knee      | 12 | 6      |
    | left_hip_x      | 13 | 7      |
    | left_hiz_z      | 14 | 8      |
    | left_hip_y      | 15 | 9      |
    | left_knee       | 16 | 10     |
    | right_shoulder1 | 17 | 11     |
    | right_shoulder2 | 18 | 12     |
    | right_elbow     | 19 | 13     |
    | left_shoulder1  | 20 | 14     |
    | left_shoulder2  | 21 | 15     |
    | left_elfbow     | 22 | 16     |

    The (x,y,z) coordinates are translational DOFs, while the orientations are rotational DOFs expressed as quaternions.
    One can read more about free joints in the [MuJoCo documentation](https://mujoco.readthedocs.io/en/latest/XMLreference.html).

    **Note:**
    When using Humanoid-v3 or earlier versions, problems have been reported when using a `mujoco-py` version > 2.0, resulting in  contact forces always being 0.
    Therefore, it is recommended to use a `mujoco-py` version < 2.0 when using the Humanoid environment if you want to report results with contact forces (if contact forces are not used in your experiments, you can use version > 2.0).


    ## Rewards
    The total reward is: ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost - contact_cost*.

    - *healthy_reward*:
    Every timestep that the Humanoid is alive (see definition in section "Episode End"),
    it gets a reward of fixed value `healthy_reward` (default is $5$).
    - *forward_reward*:
    A reward for moving forward,
    this reward would be positive if the Humanoid moves forward (in the positive $x$ direction / in the right direction).
    $w_{forward} \times \frac{dx}{dt}$, where
    $dx$ is the displacement of the center of mass ($x_{after-action} - x_{before-action}$),
    $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $5$),
    and `frametime` which is $0.001$ - so the default is $dt = 5 \times 0.003 = 0.015$,
    $w_{forward}$ is the `forward_reward_weight` (default is $1.25$).
    - *ctrl_cost*:
    A negative reward to penalize the Humanoid for taking actions that are too large.
    $w_{control} \times \|action\|_2^2$,
    where $w_{control}$ is `ctrl_cost_weight` (default is $0.1$).
    - *contact_cost*:
    A negative reward to penalize the Humanoid if the external contact forces are too large.
    $w_{contact} \times clamp(contact\_cost\_range, \|F_{contact}\|_2^2)$, where
    $w_{contact}$ is `contact_cost_weight` (default is $5\times10^{-7}$),
    $F_{contact}$ are the external contact forces (see `cfrc_ext` section on observation).

    `info` contains the individual reward terms.

    **Note:** There is a bug in the `Humanoid-v4` environment that causes *contact_cost* to always be 0.


    ## Starting State
    The initial position state is $[0.0, 0.0, 1.4, 1.0, 0.0, ... 0.0] + \mathcal{U}_{[-reset\_noise\_scale \times I_{24}, reset\_noise\_scale \times I_{24}]}$.
    The initial velocity state is $\mathcal{U}_{[-reset\_noise\_scale \times I_{23}, reset\_noise\_scale \times I_{23}]}$.

    where $\mathcal{U}$ is the multivariate uniform continuous distribution.

    Note that the z- and x-coordinates are non-zero so that the humanoid can immediately stand up and face forward (x-axis).


    ## Episode End
    ### Termination
    If `terminate_when_unhealthy is True` (the default), the environment terminates when the Humanoid is unhealthy.
    The Humanoid is said to be unhealthy if any of the following happens:

    1. The z-coordinate of the torso (the height) is **not** in the closed interval given by the `healthy_z_range` argument (default is $[1.0, 2.0]$).

    ### Truncation
    The default duration of an episode is 1000 timesteps.


    ## Arguments
    Humanoid provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
    These parameters can be applied during `gymnasium.make` in the following way:

    ```python
    import gymnasium as gym
    env = gym.make('Humanoid-v5', ctrl_cost_weight=0.1, ....)
    ```

    | Parameter                                    | Type      | Default          | Description                                                                                                                                                                                                 |
    | -------------------------------------------- | --------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                                   | **str**   | `"humanoid.xml"` | Path to a MuJoCo model                                                                                                                                                                                      |
    | `forward_reward_weight`                      | **float** | `1.25`           | Weight for _forward_reward_ term (see `Rewards` section)                                                                                                                                                    |
    | `ctrl_cost_weight`                           | **float** | `0.1`            | Weight for _ctrl_cost_ term (see `Rewards` section)                                                                                                                                                         |
    | `contact_cost_weight`                        | **float** | `5e-7`           | Weight for _contact_cost_ term (see `Rewards` section)                                                                                                                                                      |
    | `contact_cost_range`                         | **float** | `(-np.inf, 10.0)`| Clamps the _contact_cost_ term (see `Rewards` section)                                                                                                                                                      |
    | `healthy_reward`                             | **float** | `5.0`            | Weight for _healthy_reward_ term (see `Rewards` section)                                                                                                                                                    |
    | `terminate_when_unhealthy`                   | **bool**  | `True`           | If `True`, issue a `terminated` signal is unhealthy (see `Episode End` section)                                                                                                                                |
    | `healthy_z_range`                            | **tuple** | `(1.0, 2.0)`     | The humanoid is considered healthy if the z-coordinate of the torso is in this range (see `Episode End` section)                                                                                            |
    | `reset_noise_scale`                          | **float** | `1e-2`           | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                               |
    | `exclude_current_positions_from_observation` | **bool**  | `True`           | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies (see `Observation State` section) |
    | `include_cinert_in_observation`              | **bool**  | `True`           | Whether to include *cinert* elements in the observations (see `Observation State` section)                                                                                                                  |
    | `include_cvel_in_observation`                | **bool**  | `True`           | Whether to include *cvel* elements in the observations (see `Observation State` section)                                                                                                                    |
    | `include_qfrc_actuator_in_observation`       | **bool**  | `True`           | Whether to include *qfrc_actuator* elements in the observations (see `Observation State` section)                                                                                                           |
    | `include_cfrc_ext_in_observation`            | **bool**  | `True`           | Whether to include *cfrc_ext* elements in the observations (see `Observation State` section)                                                                                                                |

    ## Version History
    * v5:
        - Minimum `mujoco` version is now 2.3.3.
        - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
        - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
        - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the MuJoCo environments.
        - Return a non-empty `info` with `reset()`, previously an empty dictionary was returned, the new keys are the same state information as `step()`.
        - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
        - Fixed bug: `healthy_reward` was given on every step (even if the Humanoid was unhealthy), now it is only given when the Humanoid is healthy. The `info["reward_survive"]` is updated with this change (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/526)).
        - Restored `contact_cost` and the corresponding `contact_cost_weight` and `contact_cost_range` arguments, with the same defaults as in `Humanoid-v3` (was removed in `v4`) (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/504)).
        - Excluded the `cinert` & `cvel` & `cfrc_ext` of `worldbody` and `root`/`freejoint` `qfrc_actuator` from the observation space, as it was always 0 and thus provided no useful information to the agent, resulting in slightly faster training (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/204)).
        - Restored the `xml_file` argument (was removed in `v4`).
        - Added `include_cinert_in_observation`, `include_cvel_in_observation`, `include_qfrc_actuator_in_observation`, `include_cfrc_ext_in_observation` arguments to allow for the exclusion of observation elements from the observation space.
        - Fixed `info["x_position"]` & `info["y_position"]` & `info["distance_from_origin"]` returning `xpos` instead of `qpos` based observations (`xpos` observations are behind 1 `mj_step()` more [here](https://github.com/deepmind/mujoco/issues/889#issuecomment-1568896388)) (related [GitHub issue #1](https://github.com/Farama-Foundation/Gymnasium/issues/521) & [GitHub issue #2](https://github.com/Farama-Foundation/Gymnasium/issues/539)).
        - Added `info["tendon_length"]` and `info["tendon_velocity"]` containing observations of the Humanoid's 2 tendons connecting the hips to the knees.
        - Renamed `info["reward_alive"]` to `info["reward_survive"]` to be consistent with the other environments.
        - Renamed `info["reward_linvel"]` to `info["reward_forward"]` to be consistent with the other environments.
        - Renamed `info["reward_quadctrl"]` to `info["reward_ctrl"]` to be consistent with the other environments.
        - Removed `info["forward_reward"]` as it is equivalent to `info["reward_forward"]`.
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3
    * v3: Support for `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc. rgb rendering comes from tracking camera (so agent does not run away from screen). Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
        - Note: the environment robot model was slightly changed at `gym==0.21.0` and training results are not comparable with `gym<0.21` and `gym>=0.21` (related [GitHub PR](https://github.com/openai/gym/pull/932/files))
    * v2: All continuous control environments now use mujoco-py >= 1.50. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
        - Note: the environment robot model was slightly changed at `gym==0.21.0` and training results are not comparable with `gym<0.21` and `gym>=0.21` (related [GitHub PR](https://github.com/openai/gym/pull/932/files))
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "humanoid.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.25,
        ctrl_cost_weight: float = 0.1,
        contact_cost_weight: float = 5e-7,
        contact_cost_range: tuple[float, float] = (-np.inf, 10.0),
        healthy_reward: float = 5.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple[float, float] = (0.85, 2.0), # TODO: TRAIN ON (1.0, 2.0) -- DEMO ON (0.20, 2.0)
        reset_noise_scale: float = 2e-2, # 1e-2
        exclude_current_positions_from_observation: bool = True,
        include_cinert_in_observation: bool = True,
        include_cvel_in_observation: bool = True,
        include_qfrc_actuator_in_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        **kwargs,
    ):
        xml_path = os.path.join(os.path.dirname(__file__), "walk.xml")
        utils.EzPickle.__init__(
            self,
            xml_path or xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            contact_cost_range,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            include_cinert_in_observation,
            include_cvel_in_observation,
            include_qfrc_actuator_in_observation,
            include_cfrc_ext_in_observation,
            **kwargs,
        )

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

        self._include_cinert_in_observation = include_cinert_in_observation
        self._include_cvel_in_observation = include_cvel_in_observation
        self._include_qfrc_actuator_in_observation = (
            include_qfrc_actuator_in_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        self.truncation_timer = 0
        self.last_milestone_x = 0.01
        self.milestone_interval = 0.05
        self.previous_foot_potential = 0.0 
        
        self.possible_speeds = [5.0] # 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5
        self.target_velocity = np.random.choice(self.possible_speeds)

        self.off_track_counter = 0
        self.idle_counter = 0
        self.avg_velocity = 0

        self.possible_directions = [
            np.array([1.0, 0.0]),    # Forward
        ]
        """ np.array([0.707, 0.707]),  # Forward-Left (45 degrees)
        np.array([0.0, 1.0]),    # Left
        np.array([-0.707, 0.707]), # Backward-Left (135 degrees)
        np.array([-1.0, 0.0]),   # Backward
        np.array([-0.707, -0.707]),# Backward-Right (225 degrees)
        np.array([0.0, -1.0]),   # Right
        np.array([0.707, -0.707])  # Forward-Right (315 degrees) """
        direction_index = self.np_random.integers(len(self.possible_directions))
        self.target_direction = self.possible_directions[direction_index]

        self.total_reward = 0.0

        MujocoEnv.__init__(
            self,
            xml_path or xml_file,
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
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += self.data.cinert[1:].size * include_cinert_in_observation
        obs_size += self.data.cvel[1:].size * include_cvel_in_observation
        obs_size += (self.data.qvel.size - 6) * include_qfrc_actuator_in_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation

        obs_size += 3

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
            "cinert": self.data.cinert[1:].size * include_cinert_in_observation,
            "cvel": self.data.cvel[1:].size * include_cvel_in_observation,
            "qfrc_actuator": (self.data.qvel.size - 6)
            * include_qfrc_actuator_in_observation,
            "cfrc_ext": self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation,
            "ten_length": 0,
            "ten_velocity": 0,
            "target_velocity": 1,
            "target_direction": 2,
        }

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    @property
    def contact_cost(self):
        contact_forces = self.data.cfrc_ext
        contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        contact_cost = np.clip(contact_cost, min_cost, max_cost)
        return contact_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._include_cinert_in_observation is True:
            com_inertia = self.data.cinert[1:].flatten()
        else:
            com_inertia = np.array([])
        if self._include_cvel_in_observation is True:
            com_velocity = self.data.cvel[1:].flatten()
        else:
            com_velocity = np.array([])

        if self._include_qfrc_actuator_in_observation is True:
            actuator_forces = self.data.qfrc_actuator[6:].flatten()
        else:
            actuator_forces = np.array([])
        if self._include_cfrc_ext_in_observation is True:
            external_contact_forces = self.data.cfrc_ext[1:].flatten()
        else:
            external_contact_forces = np.array([])

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        standard_observation = np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )
        goal_observation = np.concatenate(
            [
                [self.target_velocity],    # The target speed
                self.target_direction      # The target direction vector [x, y]
            ]
        )

        return np.concatenate([standard_observation, goal_observation])
    
    def quaternion_to_rotation_matrix(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
            [2*x*y + 2*z*w,           1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,           2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
        ])
    
    def reward_leg_swing(self):
        swing_reward_scale = 0.7  
        gamma = 0.99              
        torso_x_position = self.data.qpos[0]
        right_foot_x_position = self.data.body('right_foot').xpos[0]
        left_foot_x_position = self.data.body('left_foot').xpos[0]
        current_foot_potential = max(
            right_foot_x_position - torso_x_position,
            left_foot_x_position - torso_x_position
        )
        reward_leg_swing = swing_reward_scale * (gamma * current_foot_potential - self.previous_foot_potential)
        # Update the potential for the next step's calculation
        self.previous_foot_potential = current_foot_potential
        return np.clip(reward_leg_swing, -1.0, 1.0)
    
    def center_of_mass_offset_penalty(self, xy_position_after):
        left_foot_xy = self.data.body('left_foot').xpos[0:2]
        right_foot_xy = self.data.body('right_foot').xpos[0:2]
        support_center_xy = (left_foot_xy + right_foot_xy) / 2.0 # midpoint between feet

        # 4. Calculate the error (how far the CoM is from the support center)
        com_error = np.linalg.norm(xy_position_after - support_center_xy)

        # 5. Create the reward. This is a PENALTY, so it's negative.
        #    The agent is rewarded for keeping this error small.
        reward_com_stability = -1.5 * com_error
        return reward_com_stability
    
    def feet_push_reward(self):
        reward_foot_push = 0.0
        foot_push_scale = 0.025 # Tuneable weight
        foot_on_ground_max_z = 0.1 # Threshold for foot being on ground

        # Right Foot Push
        id_geom_right_foot = self.model.geom('right_foot').id
        id_body_right_foot = self.model.body('right_foot').id # Assuming a body named 'right_foot'

        right_foot_z_position = self.data.geom_xpos[id_geom_right_foot][2]

        if right_foot_z_position < foot_on_ground_max_z:
            vertical_force_on_right_foot = self.data.cfrc_ext[id_body_right_foot][2]
            reward_foot_push += max(0, vertical_force_on_right_foot)

        # Left Foot Push
        id_geom_left_foot = self.model.geom('left_foot').id
        id_body_left_foot = self.model.body('left_foot').id # Assuming a body named 'left_foot'

        left_foot_z_position = self.data.geom_xpos[id_geom_left_foot][2]

        if left_foot_z_position < foot_on_ground_max_z:
            vertical_force_on_left_foot = self.data.cfrc_ext[id_body_left_foot][2]
            reward_foot_push += max(0, vertical_force_on_left_foot)

        final_reward_foot_push = foot_push_scale * reward_foot_push
        return np.clip(final_reward_foot_push, 0, 1)

    def _reward_upright(self):
        """Reward for keeping the torso upright."""
        # Get the torso's local Z-axis in world frame
        torso_up_vector = self.data.xmat[self.model.body('torso').id].reshape(3, 3)[:, 2]
        pelvis_up_vector = self.data.xmat[self.model.body('pelvis').id].reshape(3, 3)[:, 2]
        world_up_vector = np.array([0, 0, 1])
        # Calculate alignment, clip at 0, and square it to sharpen the reward peak
        alignment = np.dot(torso_up_vector, world_up_vector)
        alignment2 = np.dot(pelvis_up_vector, world_up_vector)
        upright_scale = 0.5
        return upright_scale * ((max(0, alignment)**3) + (max(0, alignment2)**3)) / 2
    
    def _is_geom_in_contact(self, geom_id_1, geom_id_2):
        """Checks if two geoms are currently in contact."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # Check if the two specified geoms are the ones in contact
            # The order (geom1 vs geom2) can vary
            if (contact.geom1 == geom_id_1 and contact.geom2 == geom_id_2) or \
            (contact.geom1 == geom_id_2 and contact.geom2 == geom_id_1):
                return True
        return False

    def sliding_penalty(self, left_foot_z, right_foot_z):
        """
        Calculates a penalty for feet sliding on the ground.
        Returns a negative value proportional to the sliding speed.
        """
        sliding_penalty_scale = -0.1
        sliding_speed_threshold = 0.05 

        total_sliding_speed = 0.0

        """ 
        geom_left_foot = self.model.geom('left_foot').id
        run_lane = self.model.geom('lane_3').id
        left_foot_on_ground = self._is_geom_in_contact(geom_left_foot, run_lane)
        geom_right_foot = self.model.geom('right_foot').id
        right_foot_on_ground = self._is_geom_in_contact(geom_right_foot, run_lane)
        """

        if left_foot_z < 0.1:
            left_foot_horizontal_velocity = self.data.cvel[self.model.body('left_foot').id][0:2]
            left_foot_sliding_speed = np.linalg.norm(left_foot_horizontal_velocity)
            if left_foot_sliding_speed > sliding_speed_threshold:
                total_sliding_speed += left_foot_sliding_speed

        if right_foot_z < 0.1:
            right_foot_horizontal_velocity = self.data.cvel[self.model.body('right_foot').id][0:2]
            right_foot_sliding_speed = np.linalg.norm(right_foot_horizontal_velocity)
            if right_foot_sliding_speed > sliding_speed_threshold:
                total_sliding_speed += right_foot_sliding_speed
        
        return np.clip(sliding_penalty_scale * total_sliding_speed, -1.0, 0.0)
    
    def _get_rew(self, x_velocity: float, action, torso_z):
        forward_reward = 1.5 * x_velocity
        healthy_reward = 10 if torso_z > 1.0 else 0
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        return reward
    
    """
    Calculates a reward for matching target velocity.
    """
    def track_velocity_reward(self, x_velocity: float):
        velocity_reward_tolerance = 2.0 # Controls how tolerant the reward is. Higher value = stricter requirement.
        speed_error_sq = (x_velocity - self.target_velocity)**2
        forward_reward_weight = 2.5
        forward_reward = forward_reward_weight * np.exp(-velocity_reward_tolerance * speed_error_sq)
        return forward_reward if forward_reward > 1e-4 else 0.0
    
    def healthy_state_reward(self, torso_z):
        return -0.1 if torso_z > 0.9 else -5 # TODO: Reduce healthy reward in phase 2
    
    """
    Calculates a reward for facing the target direction.
    """
    def track_direction_reward(self):
        facing_reward_scale = 0.25
        sharpness = 2.0 # Controls how tolerant the reward is. Higher value = stricter requirement.

        torso_quat = self.data.qpos[3:7]
        R_torso = self.quaternion_to_rotation_matrix(torso_quat)
        torso_forward_vector = R_torso[:, 0]
        torso_forward_xy = torso_forward_vector[0:2]
        norm = np.linalg.norm(torso_forward_xy)
        if norm < 1e-5:
            return 0.0
        current_facing_direction = torso_forward_xy / norm
        error_sq = np.sum((current_facing_direction - self.target_direction)**2)
        # Use the exponential function to create a reward that peaks at 1 when error is 0
        torso_reward = np.exp(-sharpness * error_sq)

        R_pelvis = self.data.body('pelvis').xmat.reshape(3, 3)
        pelvis_forward_vector = R_pelvis[:, 0]  # Assuming pelvis X-axis is also "forward"
        pelvis_forward_xy = pelvis_forward_vector[0:2]
        norm_pelvis = np.linalg.norm(pelvis_forward_xy)
        if norm_pelvis < 1e-5:
            return 0.0
        current_pelvis_facing = pelvis_forward_xy / norm_pelvis
        error_sq_pelvis = np.sum((current_pelvis_facing - self.target_direction)**2)
        pelvis_reward = np.exp(-sharpness * error_sq_pelvis)

        combined_reward = (torso_reward + pelvis_reward) / 2.0
        return facing_reward_scale * combined_reward, True if combined_reward > 0.5 else False

    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)
        torso_z = self.data.qpos[2]

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        x_position_after, y_position_after = xy_position_after

        idle_speed_threshold = 0.5
        idle_penalty = 0.0
        if x_velocity <= idle_speed_threshold:
            idle_penalty = -0.1
            self.idle_counter += 1
        true_speed_reward = 0.0
        if x_velocity > idle_speed_threshold:
            self.idle_counter = 0
            true_speed_reward = np.log1p(x_velocity)
        centre_position_reward = 0.0
        if abs(y_position_after) <= 1:
            normalized = max(0, 0.75 - y_position_after)
            centre_position_reward = normalized

        center_of_mass_offset_penalty = self.center_of_mass_offset_penalty(xy_position_after)

        left_foot_x, left_foot_y, left_foot_z = self.data.geom('left_foot').xpos[:]
        right_foot_x, right_foot_y, right_foot_z = self.data.geom('right_foot').xpos[:]
        foot_separation_x_reward = np.clip(abs(left_foot_x - right_foot_x), 0, 0.2) if torso_z > 0.9 else 0.0
        foot_separation_y_reward = 0.1 if abs(left_foot_y - right_foot_y) < 0.5 and torso_z > 0.9 else 0.0
        feet_lift_reward = 0.0
        if left_foot_z > 0.1 or right_foot_z > 0.1 and torso_z > 0.9:
            feet_lift_reward = np.clip(np.max([left_foot_z, right_foot_z]) * 3, 0.2, 1)

        milestone_reward = 0.0
        if left_foot_x - self.last_milestone_x >= self.milestone_interval or right_foot_x - self.last_milestone_x >= self.milestone_interval:
            milestone_reward = 1 # 0.5 in phase 1
            self.last_milestone_x += self.milestone_interval

        # time_penalty = -0.0005

        off_track_penalty = 0.0
        if abs(y_position_after) > 1:
            off_track_penalty = -1
            self.off_track_counter += 1
        
        self.left_foot_id = self.model.body('left_foot').id
        self.right_foot_id = self.model.body('right_foot').id
        left_foot_force_z = self.data.cfrc_ext[self.left_foot_id][2]
        right_foot_force_z = self.data.cfrc_ext[self.right_foot_id][2]
        feet_lift_force_difference = abs(left_foot_force_z - right_foot_force_z)
        penalty_asymmetry = -0.05 * feet_lift_force_difference

        left_hip_vy = self.data.qvel[15] # Index for left_hip_y velocity
        right_hip_vy = self.data.qvel[11] # Index for right_hip_y velocity
        left_knee_v = self.data.qvel[16]
        right_knee_v = self.data.qvel[12]
        movement_bonus = (abs(left_hip_vy) + abs(right_hip_vy) + abs(left_knee_v) + abs(right_knee_v)) / 100

        observation = self._get_obs()

        fall_penalty = 0.0
        if not self.is_healthy:
            fall_penalty = -1000.0 / self.truncation_timer # Phase 2 - turn into less fall penalty for longer milestone reached.

        torso_height_reward = torso_z / 4
        low_height_penalty = 0
        if torso_z < 0.9:
            low_height_penalty = -2.0 * (1.2 - torso_z)
            torso_height_reward = 0.0

        feet_push_reward = self.feet_push_reward() # disabled in phase 1
        bending_knee_reward = self.target_knee_flexion_reward()
        
        sliding_penalty = self.sliding_penalty(left_foot_z, right_foot_z)
        upright_reward = self._reward_upright()
        healthy_state_reward = self.healthy_state_reward(torso_z)
        target_direction_reward, is_correct_orientation = self.track_direction_reward()
        target_velocity_reward = self.track_velocity_reward(x_velocity) if is_correct_orientation else 0.0

        """ print("RESULTS: ",  
              "_:", self.target_direction_KEY, self.target_velocity, x_velocity,
              "A:", target_velocity_reward, 
              "B:", target_direction_reward,
              "C:", center_of_mass_offset_penalty, 
              "D:", upright_reward,
              "E:", feet_lift_reward,
              "I", sliding_penalty,
              "J:", low_height_penalty,
              "K:", fall_penalty,
              "L", healthy_state_reward) """

        # deprecated in Phase 2: torso_height_reward, bending_knee_reward, foot_separation_x_reward, foot_separation_y_reward
        # PHASE 2: off_track_penalty, milestone_reward
        step_reward = (
            centre_position_reward +
            target_velocity_reward +
            target_direction_reward +
            true_speed_reward +
            center_of_mass_offset_penalty +
            upright_reward +
            feet_lift_reward +
            sliding_penalty +
            low_height_penalty +
            fall_penalty +
            healthy_state_reward +
            off_track_penalty +
            idle_penalty +
            milestone_reward
        )
        self.total_reward += step_reward

        if self.render_mode == "human":
            self.render()

        truncation = False
        self.truncation_timer += 1
        self.avg_velocity += x_velocity
        if self.truncation_timer >= (4000 / self.frame_skip): # TODO: DEMO 4000 TRAINING 1500
            print("Truncated. REWARD:", np.trunc(self.total_reward), "Distance: ", np.trunc(x_position_after), "Avg Velocity: ", np.trunc(self.avg_velocity / self.truncation_timer))
            truncation = True
            self.total_reward = 0
            self.truncation_timer = 0
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        if self.off_track_counter >= 500:
            print("Off-track. REWARD:", np.trunc(self.total_reward), "Distance: ", np.trunc(x_position_after), "Avg Velocity: ", np.trunc(self.avg_velocity / self.truncation_timer))
            terminated = True
            self.total_reward = 0
            self.off_track_counter = 0
        if self.idle_counter >= 1000:
            print("Idle. REWARD:", np.trunc(self.total_reward), "Distance: ", np.trunc(x_position_after), "Avg Velocity: ", np.trunc(self.avg_velocity / self.truncation_timer))
            terminated = True
            self.total_reward = 0
            self.idle_counter = 0
        if fall_penalty < 0:
            print("Fall. REWARD:", np.trunc(self.total_reward), "Distance: ", np.trunc(x_position_after), "Avg Velocity: ", np.trunc(self.avg_velocity / self.truncation_timer))
            self.total_reward = 0
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
        }

        return observation, step_reward, terminated, truncation, info
    
    def target_knee_flexion_reward(self):
        """
        Calculates a reward that peaks when the knees are bent
        to a specific target angle.
        Returns a value between 0 and 1.
        """
        # --- 1. Define Your Target ---
        # Your desired amount of bend from the fully straight position, in radians.
        target_flexion_rad = 0.22

        # The 'straightest' angle is -2 degrees (~ -0.035 radians).
        # The target angle is the straight position minus the desired flexion,
        # because more flexion results in a more negative angle.
        target_knee_angle = -0.035 - target_flexion_rad

        # This parameter controls how "sharp" the reward peak is.
        # A higher value means the agent must be more precise to get the reward.
        sharpness = 30.0

        # --- 2. Get Current Knee Angles ---
        right_knee_angle = self.data.qpos[13]
        left_knee_angle = self.data.qpos[17]

        # --- 3. Calculate Reward for Each Knee ---
        # Calculate the squared error (how far the current angle is from the target)
        right_error_sq = (right_knee_angle - target_knee_angle)**2
        left_error_sq = (left_knee_angle - target_knee_angle)**2

        # The exponential function converts the error into a reward between 0 and 1.
        # The reward is 1.0 when the error is 0, and decays towards 0 as error increases.
        right_reward = np.exp(-sharpness * right_error_sq)
        left_reward = np.exp(-sharpness * left_error_sq)

        # --- 4. Return the Average Reward ---
        # We average the rewards from both knees to get a final score.
        return (right_reward + left_reward) / 2

    def reset_model(self):
        self.truncation_timer = 0
        self.last_milestone_x = 0
        self.total_reward = 0
        self.off_track_counter = 0
        self.idle_counter = 0
        self.avg_velocity = 0
        self.target_velocity = np.random.choice(self.possible_speeds)
        direction_index = self.np_random.integers(len(self.possible_directions))
        self.target_direction = self.possible_directions[direction_index]

        match direction_index:
            case 0:  # Forward
                self.target_direction_KEY = "forward"
            case 1:  # Forward-Left
                self.target_direction_KEY = "forward_left"
            case 2:  # Left
                self.target_direction_KEY = "left"
            case 3:  # Backward-Left
                self.target_direction_KEY = "backward_left"
            case 4:  # Backward
                self.target_direction_KEY = "backward"
            case 5:  # Backward-Right
                self.target_direction_KEY = "backward_right"
            case 6:  # Right
                self.target_direction_KEY = "right"
            case 7:  # Forward-Right
                self.target_direction_KEY = "forward_right"

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        slight_bend_angle_rad = -0.30
        # The qpos indices for the knee joints are 13 (right) and 17 (left)
        qpos[13] = slight_bend_angle_rad
        qpos[17] = slight_bend_angle_rad
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }
