__credits__ = ["Kallinteris-Andreas"]

from typing import Dict, Tuple, Union

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import os

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.5, 0.5, 0.5925)),
    "elevation": -40.0,
}


class HumanoidStandupEnv(MujocoEnv, utils.EzPickle):
    r"""
    ## Description
    This environment is based on the environment introduced by Tassa, Erez and Todorov in ["Synthesis and stabilization of complex behaviors through online trajectory optimization"](https://ieeexplore.ieee.org/document/6386025).
    The 3D bipedal robot is designed to simulate a human.
    It has a torso (abdomen) with a pair of legs and arms, and a pair of tendons connecting the hips to the knees.
    The legs each consist of three body parts (thigh, shin, foot), and the arms consist of two body parts (upper arm, forearm).
    The environment starts with the humanoid laying on the ground, and then the goal of the environment is to make the humanoid stand up and then keep it standing by applying torques to the various hinges.


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
    When using HumanoidStandup-v3 or earlier versions, problems have been reported when using a `mujoco-py` version > 2.0, resulting in  contact forces always being 0.
    Therefore, it is recommended to use a `mujoco-py` version < 2.0 when using the HumanoidStandup environment if you want to report results with contact forces (if contact forces are not used in your experiments, you can use version > 2.0).


    ## Rewards
    The total reward is: ***reward*** *=* *uph_cost + 1 - quad_ctrl_cost - quad_impact_cost*.

    - *uph_cost*:
    A reward for moving up (trying to stand up).
    This is not a relative reward, measuring how far up the robot has moved since the last timestep,
    but an absolute reward measuring how far up the Humanoid has moved up in total.
    It is measured as $w_{uph} \times \frac{z_{after\_action} - 0}{dt}$,
    where $z_{after\_action}$ is the z coordinate of the torso after taking an action,
    and $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $5$),
    and `frametime`, which is $0.01$ - so the default is $dt = 5 \times 0.01 = 0.05$,
    and $w_{uph}$ is `uph_cost_weight` (default is $1$).
    - *quad_ctrl_cost*:
    A negative reward to penalize the Humanoid for taking actions that are too large.
    $w_{quad\_control} \times \|action\|_2^2$,
    where $w_{quad\_control}$ is `ctrl_cost_weight` (default is $0.1$).
    - *impact_cost*:
    A negative reward to penalize the Humanoid if the external contact forces are too large.
    $w_{impact} \times clamp(impact\_cost\_range, \|F_{contact}\|_2^2)$, where
    $w_{impact}$ is `impact_cost_weight` (default is $5\times10^{-7}$),
    $F_{contact}$ are the external contact forces (see `cfrc_ext` section on Observation Space).

    `info` contains the individual reward terms.


    ## Starting State
    The initial position state is $[0.0, 0.0, 1.4, 1.0, 0.0, ... 0.0] + \mathcal{U}_{[-reset\_noise\_scale \times I_{24}, reset\_noise\_scale \times I_{24}]}$.
    The initial velocity state is $\mathcal{U}_{[-reset\_noise\_scale \times I_{23}, reset\_noise\_scale \times I_{23}]}$.

    where $\mathcal{U}$ is the multivariate uniform continuous distribution.

    Note that the z- and x-coordinates are non-zero so that the humanoid immediately lies down and faces forward (x-axis).


    ## Episode End
    ### Termination
    The Humanoid never terminates.

    ### Truncation
    The default duration of an episode is 1000 timesteps.


    ## Arguments
    HumanoidStandup provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
    These parameters can be applied during `gymnasium.make` in the following way:

    ```python
    import gymnasium as gym
    env = gym.make('HumanoidStandup-v5', impact_cost_weight=0.5e-6, ....)
    ```

    | Parameter                                    | Type      | Default               | Description                                                                                                                                                                                                 |
    | -------------------------------------------- | --------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                                   | **str**   |`"humanoidstandup.xml"`| Path to a MuJoCo model                                                                                                                                                                                      |
    | `uph_cost_weight`                            | **float** | `1`                   | Weight for _uph_cost_ term (see `Rewards` section)                                                                                                                                                          |
    | `ctrl_cost_weight`                           | **float** | `0.1`                 | Weight for _quad_ctrl_cost_ term (see `Rewards` section)                                                                                                                                                    |
    | `impact_cost_weight`                         | **float** | `0.5e-6`              | Weight for _impact_cost_ term (see `Rewards` section)                                                                                                                                                       |
    | `impact_cost_range`                          | **float** | `(-np.inf, 10.0)`     | Clamps the _impact_cost_ (see `Rewards` section)                                                                                                                                                            |
    | `reset_noise_scale`                          | **float** | `1e-2`                | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                               |
    | `exclude_current_positions_from_observation` | **bool**  | `True`                | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies (see `Observation Space` section) |
    | `include_cinert_in_observation`              | **bool**  | `True`                | Whether to include *cinert* elements in the observations (see `Observation Space` section)                                                                                                                  |
    | `include_cvel_in_observation`                | **bool**  | `True`                | Whether to include *cvel* elements in the observations (see `Observation Space` section)                                                                                                                    |
    | `include_qfrc_actuator_in_observation`       | **bool**  | `True`                | Whether to include *qfrc_actuator* elements in the observations (see `Observation Space` section)                                                                                                           |
    | `include_cfrc_ext_in_observation`            | **bool**  | `True`                | Whether to include *cfrc_ext* elements in the observations (see `Observation Space` section)                                                                                                                |

    ## Version History
    * v5:
        - Minimum `mujoco` version is now 2.3.3.
        - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
        - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
        - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the MuJoCo environments.
        - Return a non-empty `info` with `reset()`, previously an empty dictionary was returned, the new keys are the same state information as `step()`.
        - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
        - Excluded the `cinert` & `cvel` & `cfrc_ext` of `worldbody` and `root`/`freejoint` `qfrc_actuator` from the observation space, as it was always 0, and thus provided no useful information to the agent, resulting in slightly faster training (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/204)).
        - Restored the `xml_file` argument (was removed in `v4`).
        - Added `xml_file` argument.
        - Added `uph_cost_weight`, `ctrl_cost_weight`, `impact_cost_weight`, `impact_cost_range` arguments to configure the reward function (defaults are effectively the same as in `v4`).
        - Added `reset_noise_scale` argument to set the range of initial states.
        - Added `include_cinert_in_observation`, `include_cvel_in_observation`, `include_qfrc_actuator_in_observation`, `include_cfrc_ext_in_observation` arguments to allow for the exclusion of observation elements from the observation space.
        - Added `info["tendon_length"]` and `info["tendon_velocity"]` containing observations of the Humanoid's 2 tendons connecting the hips to the knees.
        - Added `info["x_position"]` & `info["y_position"]` which contain the observations excluded when `exclude_current_positions_from_observation == True`.
        - Added `info["z_distance_from_origin"]` which is the vertical distance of the "torso" body from its initial position.
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3.
    * v3: This environment does not have a v3 release.
    * v2: All continuous control environments now use mujoco-py >= 1.50.
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release.
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
        xml_file: str = "humanoidstandup.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        uph_cost_weight: float = 1,
        ctrl_cost_weight: float = 0.1,
        impact_cost_weight: float = 0.5e-6,
        impact_cost_range: Tuple[float, float] = (-np.inf, 10.0),
        reset_noise_scale: float = 1e-2,
        exclude_current_positions_from_observation: bool = True,
        include_cinert_in_observation: bool = True,
        include_cvel_in_observation: bool = True,
        include_qfrc_actuator_in_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        **kwargs,
    ):
        xml_path = os.path.join(os.path.dirname(__file__), "standup.xml")
        utils.EzPickle.__init__(
            self,
            xml_path or xml_file,
            frame_skip,
            default_camera_config,
            uph_cost_weight,
            ctrl_cost_weight,
            impact_cost_weight,
            impact_cost_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            include_cinert_in_observation,
            include_cvel_in_observation,
            include_qfrc_actuator_in_observation,
            include_cfrc_ext_in_observation,
            **kwargs,
        )

        self._uph_cost_weight = uph_cost_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._impact_cost_weight = impact_cost_weight
        self._impact_cost_range = impact_cost_range
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
        self.last_milestone_z = 0.05  # reset on every episode
        self.milestone_interval = 0.025
        self.initial_pelvis_z_pos = 0.0

        obs_size = 47
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += 130 * include_cinert_in_observation
        obs_size += 78 * include_cvel_in_observation
        obs_size += 17 * include_qfrc_actuator_in_observation
        obs_size += 78 * include_cfrc_ext_in_observation

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
        }

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

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

    def step(self, action):
        pos_before_z = self.data.qpos[2]
        pelvis_z_pos_before = self.data.body("pelvis").xpos[2]
        
        self.do_simulation(action, self.frame_skip)

        pos_after_z = self.data.qpos[2] 
        if self.initial_pelvis_z_pos == 0.0:
            self.initial_pelvis_z_pos = pos_after_z

        # --- Stage 0. General rewards ---
        milestone_reward = 0.0
        if pos_after_z - self.last_milestone_z >= self.milestone_interval:
            milestone_reward = 10.0 # 1
            self.last_milestone_z += self.milestone_interval
        torso_upright_reward = 2 * np.clip(self.uprightness_reward(self.data.qpos), -1, 1.5)
        hand_push_reward = np.clip(self.hand_push_reward(), 0, 1)
        reward, reward_info = self._get_rew(pos_after_z, action)
        pelvis_lift_velocity_reward = self.pelvis_lift_velocity_reward()
        feet_push_reward = self.feet_push_reward()
        pelvis_height_reward = 0
        pelvis_z_pos = self.data.body("pelvis").xpos[2] # Get Z-coordinate of pelvis
        if torso_upright_reward > 2 * 0.75:
            pelvis_height_scale = 2
            pelvis_height_reward = pelvis_height_scale * pelvis_z_pos 

        # --- Stage 1. Rewards for moving feet close to pelvis ---
        dist_pelvis_right_foot = np.linalg.norm(self.data.body("pelvis").xpos[:2] - self.data.body("right_foot").xpos[:2])
        dist_pelvis_left_foot = np.linalg.norm(self.data.body("pelvis").xpos[:2] - self.data.body("left_foot").xpos[:2])
        feet_butt_proximity_reward = 4 * np.clip(1 - ((dist_pelvis_right_foot + dist_pelvis_left_foot) / 2), 0, 1)

        # --- Stage 2. Rewards for leaning up/forward when in stable sitting position ---
        delta_z = pos_after_z - pos_before_z
        forward_lean_uprising_reward = 0.0
        upward_movement_threshold = 0.0005
        if delta_z > upward_movement_threshold: # Threshold to avoid rewarding noise
            torso_quat = self.data.qpos[3:7] # Root joint orientation (w,x,y,z)
            R_torso_to_world = self.quaternion_to_rotation_matrix(torso_quat)
            # Torso's local Z-axis (upwards vector from torso) in world coordinates
            torso_local_up_in_world = R_torso_to_world[:, 2]
            # Component of torso's local Z-axis along world's X-axis
            # Positive value means torso's "up" is tilted towards world's "forward"
            forward_tilt_component = torso_local_up_in_world[0]
            # Reward if tilted forward - no penalty for leaning back
            lean_uprising_scale = 1.5
            forward_lean_uprising_reward = lean_uprising_scale * max(0, forward_tilt_component)

        # --- Stage 3. Rewards for activating lift from stable sitting posture ---
        pelvis_upward_movement_reward = 0.0
        hip_extension_reward = 0.0
        knee_extension_reward = 0.0
        knee_extension_effort = 0.0
        height_threshold_for_leg_rewards = 0.40
        if pos_after_z > height_threshold_for_leg_rewards and feet_butt_proximity_reward > (0.5 * 4):
            feet_butt_proximity_reward = 0.0
            knee_extension_reward, hip_extension_reward, knee_extension_effort = self.hip_knee_extension_reward()
            pelvis_upward_movement_reward = np.clip((pelvis_z_pos - pelvis_z_pos_before) * 700, 0, 1)

        # --- Stage 4. Rewards for maintaining stable upright posture ---
        torso_linear_stability_at_standup_penalty = 0.0
        torso_angular_stability_at_standup_penalty = 0.0
        height_threshold_for_stable_standup_posture = 0.95
        if pos_after_z > height_threshold_for_stable_standup_posture:
            # --- Torso Linear Stability Reward ---
            torso_x_velocity = self.data.qvel[0]
            torso_y_velocity = self.data.qvel[1]
            # Penalize squared velocity; higher penalty for more speed
            # Using a penalty form (negative reward)
            horizontal_velocity_penalty_scale = 0.4 # Tuneable
            penalty_torso_horizontal_velocity = -horizontal_velocity_penalty_scale * (
                torso_x_velocity**2 + torso_y_velocity**2
            )
            torso_linear_stability_at_standup_penalty = 3 * (1 + np.clip(penalty_torso_horizontal_velocity, -1, 0))

            torso_angular_velocity_world = self.data.qvel[3:6] # wx, wy, wz for root/torso
            angular_velocity_penalty_scale = 0.025 # Tuneable
            # Penalize squared norm of angular velocity
            penalty_torso_angular_velocity = -angular_velocity_penalty_scale * np.linalg.norm(
                torso_angular_velocity_world
            )**2
            torso_angular_stability_at_standup_penalty = 3 * (1 + np.clip(penalty_torso_angular_velocity, -1, 0))

        # --- Stage 5. Rewards for having torse above height 1meter ---
        standing_reward = 0.0
        if pos_after_z > 0.90:
            standing_reward = 10.0 

        """ print("REWARDS: ",  
              "A:", milestone_reward, 
              "B:", forward_lean_uprising_reward, 
              "C:", pelvis_height_reward, 
              "D:", feet_butt_proximity_reward, 
              "E:", torso_upright_reward,
              "F:", pelvis_upward_movement_reward, 
              "G:", hand_push_reward,
              "H:", feet_push_reward,
              "J:", reward,
              "K:", hip_extension_reward,
              "L:", knee_extension_reward,
              "M:", knee_extension_effort,
              "N:", pelvis_lift_velocity_reward,
              "O:", standing_reward,
              "P:", torso_linear_stability_at_standup_penalty,
              "Q:", torso_angular_stability_at_standup_penalty) """

        step_reward = (milestone_reward +
              reward +
              forward_lean_uprising_reward +
              torso_upright_reward +
              knee_extension_effort + 
              knee_extension_reward +
              feet_push_reward +
              feet_butt_proximity_reward +
              standing_reward +
              hand_push_reward
              )
        """ pelvis_height_reward +
              pelvis_upward_movement_reward +
              pelvis_lift_velocity_reward +
              hip_extension_reward +
              torso_linear_stability_at_standup_penalty +
              torso_angular_stability_at_standup_penalty """

        if self.render_mode == "human":
            self.render()

        info = {"x_position": self.data.qpos[0], 
                "y_position": self.data.qpos[1], 
                "z_distance_from_origin": self.data.qpos[2] - self.init_qpos[2], 
                "tendon_length": self.data.ten_length, 
                "tendon_velocity": self.data.ten_velocity, **reward_info,}

        truncation = False
        self.truncation_timer += 1
        if self.truncation_timer >= (700 / self.frame_skip): # 800
            truncation = True
            self.truncation_timer = 0
            info["TimeLimit.truncated"] = True

        return self._get_obs(), step_reward, False, truncation, info
    
    def hip_knee_extension_reward(self):
        right_knee_angle = self.data.qpos[13]
        left_knee_angle = self.data.qpos[17]
        target_extended_knee_angle = -0.035  # radians (approx -2 degrees)

        knee_extension_error_scale = 0.5 # Tuneable weight
        # Quadratic penalty for deviation, negated to be a reward
        reward_knee_extension = -knee_extension_error_scale * (
            (right_knee_angle - target_extended_knee_angle)**2 +
            (left_knee_angle - target_extended_knee_angle)**2
        )

        right_hip_y_angle = self.data.qpos[12]
        left_hip_y_angle = self.data.qpos[16]
        target_extended_hip_y_angle = 0.0  # radians (0 degrees)

        hip_extension_error_scale = 0.4 # Tuneable weight
        reward_hip_y_extension = -hip_extension_error_scale * (
            (right_hip_y_angle - target_extended_hip_y_angle)**2 +
            (left_hip_y_angle - target_extended_hip_y_angle)**2
        )

        reward_knee_extension_effort = 0.0
        # --- Right Knee Extension Effort ---
        id_actuator_right_knee = self.model.actuator('right_knee').id
        right_knee_actuator_torque = self.data.actuator_force[id_actuator_right_knee]

        # Assuming positive torque from the actuator leads to extension (angle value increasing)
        if right_knee_actuator_torque > 0:
            reward_knee_extension_effort += right_knee_actuator_torque

        # --- Left Knee Extension Effort ---
        id_actuator_left_knee = self.model.actuator('left_knee').id
        left_knee_actuator_torque = self.data.actuator_force[id_actuator_left_knee]

        # Assuming positive torque from the actuator leads to extension
        if left_knee_actuator_torque > 0:
            reward_knee_extension_effort += left_knee_actuator_torque
        knee_extension_effort = reward_knee_extension_effort

        return np.clip(reward_knee_extension / 7, -1, 0), np.clip(reward_hip_y_extension / 5, -1, 0), knee_extension_effort * 2
    
    def pelvis_lift_velocity_reward(self):
        id_body_pelvis = self.model.body('pelvis').id
        pelvis_vertical_velocity = self.data.cvel[id_body_pelvis][2] # Z-component of CoM-based velocity
        pelvis_lift_velocity_scale = 0.3 # Tuneable weight
        reward_pelvis_lift_velocity = pelvis_lift_velocity_scale * max(0, pelvis_vertical_velocity)
        return np.clip(reward_pelvis_lift_velocity, 0, 1)
    
    def feet_push_reward(self):
        reward_foot_push = 0.0
        foot_push_scale = 0.025 # Tuneable weight
        foot_on_ground_max_z = 0.1 # Threshold for foot being on ground

        try:
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
        except KeyError as e:
            # print(f"Warning: Could not calculate foot push reward. Model element name not found: {e}")
            pass

        final_reward_foot_push = foot_push_scale * reward_foot_push
        return np.clip(final_reward_foot_push, 0, 1)
    
    def hand_push_reward(self):
        reward_hand_push = 0.0
        hand_push_scale = 0.025  # Tuneable weight for this reward component
        torso_z_position = self.data.qpos[2]

        # Define thresholds for when this reward is active
        # Only try to reward hand pushing when the torso is relatively low
        pushing_phase_max_torso_z = 0.6  # e.g., if torso is below 0.6m
        # Consider hands to be "on ground" if their Z-coordinate is very low
        hand_on_ground_max_z = 0.05     # e.g., if hand is below 0.05m from origin

        if torso_z_position < pushing_phase_max_torso_z:
            try:
                # --- Right Hand Push ---
                id_geom_right_hand = self.model.geom('right_hand').id
                id_body_right_lower_arm = self.model.body('right_lower_arm').id

                right_hand_z_position = self.data.geom_xpos[id_geom_right_hand][2]

                if right_hand_z_position < hand_on_ground_max_z:
                    vertical_force_on_right_forearm = self.data.cfrc_ext[id_body_right_lower_arm][2]
                    reward_hand_push += max(0, vertical_force_on_right_forearm)

                # --- Left Hand Push ---
                id_geom_left_hand = self.model.geom('left_hand').id
                id_body_left_lower_arm = self.model.body('left_lower_arm').id
                
                left_hand_z_position = self.data.geom_xpos[id_geom_left_hand][2]

                if left_hand_z_position < hand_on_ground_max_z:
                    vertical_force_on_left_forearm = self.data.cfrc_ext[id_body_left_lower_arm][2]
                    reward_hand_push += max(0, vertical_force_on_left_forearm)
                    
            except Exception as e:
                # This can happen if model elements are not found (e.g., if using a different XML)
                # print(f"Warning: Could not calculate hand push reward. Missing model elements: {e}")
                pass # Keep reward_hand_push as 0

        final_reward_hand_push = hand_push_scale * reward_hand_push
        return final_reward_hand_push
    
    def quaternion_to_rotation_matrix(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
            [2*x*y + 2*z*w,           1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,           2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
        ])
    
    def uprightness_reward(self, qpos):
        torso_quat = qpos[3:7]
        R = self.quaternion_to_rotation_matrix(torso_quat)
        torso_up = -R[:, 0]  # x axis dot is 'up' vector
        world_up = np.array([0, 0, 1])
        
        # Alignment ranges from -1 (upside down) to +1 (perfectly upright)
        alignment = np.dot(torso_up, world_up)
        reward = max(0.0, alignment)  # optionally ignore upside-down rewards
        
        return reward

    def _get_rew(self, pos_after_z: float, action):
        uph_cost = (pos_after_z - 0) / self.model.opt.timestep

        quad_ctrl_cost = self._ctrl_cost_weight * np.square(self.data.ctrl).sum()

        quad_impact_cost = (
            self._impact_cost_weight * np.square(self.data.cfrc_ext).sum()
        )
        min_impact_cost, max_impact_cost = self._impact_cost_range
        quad_impact_cost = np.clip(quad_impact_cost, min_impact_cost, max_impact_cost)

        scale = 1.0 / 100
        reward = scale * (uph_cost - quad_ctrl_cost - quad_impact_cost + 1)

        reward_info = {
            "reward_linup": uph_cost,
            "reward_quadctrl": -quad_ctrl_cost,
            "reward_impact": -quad_impact_cost,
        }

        return np.clip(reward, 0, 1), reward_info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        self.last_milestone_z = 0.05
        self.initial_pelvis_z_pos = 0.0

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_distance_from_origin": self.data.qpos[2] - self.init_qpos[2],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
        }
