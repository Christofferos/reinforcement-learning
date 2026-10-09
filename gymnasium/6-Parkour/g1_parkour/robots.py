"""Robot specifications used to build parkour scenes.

The default robot is the same textured humanoid used in ``5-HumanoidParkour/parkour.xml``
(same body tree, joint ranges, tendons, actuators and materials), so policies and
visuals stay comparable between the two projects. Two joint values are changed to make it
exactly left/right symmetric, which the mirror maps in :mod:`g1_parkour.symmetry` rely on:
the right ``hip_y`` armature is 0.01 like the left (0.008 there), and the left knee has no
spring, like the right (stiffness 1 there).

A Unitree G1 spec is also provided.  It is only usable when the MuJoCo Menagerie
``unitree_g1`` model is available locally (see :func:`g1_model_dir`).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


@dataclass(frozen=True)
class RobotSpec:
    """Everything the scene builder needs to insert a robot into a terrain."""

    name: str
    asset_xml: str
    body_xml: str
    extra_xml: str
    default_xml: str
    root_body: str
    spawn_height: float
    """Height of the root body above the terrain surface at spawn."""
    healthy_height_range: tuple[float, float]
    """Allowed root height relative to the terrain surface below the robot."""
    foot_bodies: tuple[str, ...]
    action_scale: float = 0.4
    includes: tuple[str, ...] = field(default_factory=tuple)
    """Extra MJCF files to ``<include>`` (used by the Menagerie-based G1 spec)."""
    default_joint_pos: dict[str, float] = field(default_factory=dict)
    """Standing pose in radians by joint name; unlisted hinge joints use the model's qpos0.
    Used as the reset pose and as the zero-action target in position-control mode."""
    pd_stiffness_per_torque: float = 2.0
    """Position-control kp per actuator = this factor x the actuator's torque limit (Nm/rad)."""
    pd_damping_ratio: float = 0.05
    """Position-control kd = this ratio x kp (Nm s/rad)."""


# --------------------------------------------------------------------------------------
# Textured humanoid (5-HumanoidParkour/parkour.xml, made exactly left/right symmetric)
# --------------------------------------------------------------------------------------

_HUMANOID_ASSETS = f"""
    <texture name="shirt_texture" type="2d" file="{ASSETS_DIR / 'shirtPattern.png'}"/>
    <material name="material_shirt" texture="shirt_texture" specular="0.3" shininess="0.5" texrepeat="0.5 0.5"/>
    <texture name="trunks_texture" type="2d" file="{ASSETS_DIR / 'shorts.png'}"/>
    <material name="material_trunks" texture="trunks_texture" specular="0.3" shininess="0.5" texrepeat="2 2"/>
    <texture name="lucas_head_texture" type="2d" file="{ASSETS_DIR / 'lucas.png'}"/>
    <material name="lucas_material" texture="lucas_head_texture" specular="0.3" shininess="0.5" texrepeat="1 1"/>
    <texture name="skin_texture" type="2d" file="{ASSETS_DIR / 'skintone.png'}"/>
    <material name="material_skin" texture="skin_texture" specular="0.3" shininess="0.5" texrepeat="1 1"/>
"""

_HUMANOID_BODY = """
    <body name="torso" pos="{x} {y} {z}">
        <camera name="side_view" mode="trackcom" pos="1 -3 0.5" xyaxes="1 0 0 0 0.4 0.9"/>
        <camera name="chase_view" mode="trackcom" pos="-3 0 0.5" xyaxes="0 -1 0 0.342 0 0.940"/>
        <camera name="front_view" mode="trackcom" pos="2.5 0 0.25" xyaxes="0 1 0 -0.342 0 0.940"/>
        <camera name="egocentric" mode="fixed" pos="0.2 0 0.15" xyaxes="0 -1 0 0.3 0 0.95"/>
        <joint armature="0" damping="0" limited="false" name="root" pos="0 0 0" stiffness="0" type="free"/>
        <geom fromto="0 -.07 0 0 .07 0" name="torso1" size="0.07" type="capsule" material="material_shirt"/>
        <geom name="head" pos="0 0 .19" size=".09" type="sphere" material="lucas_material" quat="0.707 0 0.507 0"/>
        <geom fromto="-.01 -.06 -.12 -.01 .06 -.12" name="uwaist" size="0.06" type="capsule" material="material_shirt"/>
        <body name="lwaist" pos="-.01 0 -0.260" quat="1.000 0 -0.002 0">
            <geom fromto="0 -.06 0 0 .06 0" name="lwaist" size="0.06" type="capsule" material="material_shirt"/>
            <joint armature="0.02" axis="0 0 1" damping="5" name="abdomen_z" pos="0 0 0.065" range="-45 45" stiffness="20" type="hinge"/>
            <joint armature="0.02" axis="0 1 0" damping="5" name="abdomen_y" pos="0 0 0.065" range="-75 30" stiffness="10" type="hinge"/>
            <body name="pelvis" pos="0 0 -0.165" quat="1.000 0 -0.002 0">
                <joint armature="0.02" axis="1 0 0" damping="5" name="abdomen_x" pos="0 0 0.1" range="-35 35" stiffness="10" type="hinge"/>
                <geom fromto="-.02 -.07 0 -.02 .07 0" name="butt" size="0.09" type="capsule" material="material_trunks"/>
                <body name="right_thigh" pos="0 -0.1 -0.04">
                    <joint armature="0.01" axis="1 0 0" damping="5" name="right_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
                    <joint armature="0.01" axis="0 0 1" damping="5" name="right_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
                    <joint armature="0.01" axis="0 1 0" damping="5" name="right_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
                    <geom fromto="0 0 0 0 0.01 -.34" name="right_thigh1" size="0.06" type="capsule" material="material_skin"/>
                    <body name="right_shin" pos="0 0.01 -0.403">
                        <joint armature="0.0060" axis="0 -1 0" name="right_knee" pos="0 0 .02" range="-160 -2" type="hinge"/>
                        <geom fromto="0 0 0 0 0 -.3" name="right_shin1" size="0.049" type="capsule" material="material_skin"/>
                        <body name="right_foot" pos="0 0 -0.45">
                            <geom name="right_foot" pos="0 0 0.1" size="0.075" type="sphere" user="0" material="material_skin"/>
                        </body>
                    </body>
                </body>
                <body name="left_thigh" pos="0 0.1 -0.04">
                    <joint armature="0.01" axis="-1 0 0" damping="5" name="left_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
                    <joint armature="0.01" axis="0 0 -1" damping="5" name="left_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
                    <joint armature="0.01" axis="0 1 0" damping="5" name="left_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
                    <geom fromto="0 0 0 0 -0.01 -.34" name="left_thigh1" size="0.06" type="capsule" material="material_skin"/>
                    <body name="left_shin" pos="0 -0.01 -0.403">
                        <joint armature="0.0060" axis="0 -1 0" name="left_knee" pos="0 0 .02" range="-160 -2" type="hinge"/>
                        <geom fromto="0 0 0 0 0 -.3" name="left_shin1" size="0.049" type="capsule" material="material_skin"/>
                        <body name="left_foot" pos="0 0 -0.45">
                            <geom name="left_foot" type="sphere" size="0.075" pos="0 0 0.1" user="0" material="material_skin"/>
                        </body>
                    </body>
                </body>
            </body>
        </body>
        <body name="right_upper_arm" pos="0 -0.17 0.06">
            <joint armature="0.0068" axis="2 1 1" name="right_shoulder1" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
            <joint armature="0.0051" axis="0 -1 1" name="right_shoulder2" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
            <geom fromto="0 0 0 .16 -.16 -.16" name="right_uarm1" size="0.04 0.16" type="capsule" material="material_skin"/>
            <body name="right_lower_arm" pos=".18 -.18 -.18">
                <joint armature="0.0028" axis="0 -1 1" name="right_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
                <geom fromto="0.01 0.01 0.01 .17 .17 .17" name="right_larm" size="0.031" type="capsule" material="material_skin"/>
                <geom name="right_hand" pos=".18 .18 .18" size="0.04" type="sphere" material="material_skin"/>
            </body>
        </body>
        <body name="left_upper_arm" pos="0 0.17 0.06">
            <joint armature="0.0068" axis="2 -1 1" name="left_shoulder1" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
            <joint armature="0.0051" axis="0 1 1" name="left_shoulder2" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
            <geom fromto="0 0 0 .16 .16 -.16" name="left_uarm1" size="0.04 0.16" type="capsule" material="material_skin"/>
            <body name="left_lower_arm" pos=".18 .18 -.18">
                <joint armature="0.0028" axis="0 -1 -1" name="left_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
                <geom fromto="0.01 -0.01 0.01 .17 -.17 .17" name="left_larm" size="0.031" type="capsule" material="material_skin"/>
                <geom name="left_hand" pos=".18 -.18 .18" size="0.04" type="sphere" material="material_skin"/>
            </body>
        </body>
    </body>
"""

_HUMANOID_EXTRA = """
  <tendon>
    <fixed name="left_hipknee">
      <joint coef="-1" joint="left_hip_y"/>
      <joint coef="1" joint="left_knee"/>
    </fixed>
    <fixed name="right_hipknee">
      <joint coef="-1" joint="right_hip_y"/>
      <joint coef="1" joint="right_knee"/>
    </fixed>
  </tendon>
  <actuator>
    <motor gear="100" joint="abdomen_y" name="abdomen_y"/>
    <motor gear="100" joint="abdomen_z" name="abdomen_z"/>
    <motor gear="100" joint="abdomen_x" name="abdomen_x"/>
    <motor gear="100" joint="right_hip_x" name="right_hip_x"/>
    <motor gear="100" joint="right_hip_z" name="right_hip_z"/>
    <motor gear="300" joint="right_hip_y" name="right_hip_y"/>
    <motor gear="200" joint="right_knee" name="right_knee"/>
    <motor gear="100" joint="left_hip_x" name="left_hip_x"/>
    <motor gear="100" joint="left_hip_z" name="left_hip_z"/>
    <motor gear="300" joint="left_hip_y" name="left_hip_y"/>
    <motor gear="200" joint="left_knee" name="left_knee"/>
    <motor gear="25" joint="right_shoulder1" name="right_shoulder1"/>
    <motor gear="25" joint="right_shoulder2" name="right_shoulder2"/>
    <motor gear="25" joint="right_elbow" name="right_elbow"/>
    <motor gear="25" joint="left_shoulder1" name="left_shoulder1"/>
    <motor gear="25" joint="left_shoulder2" name="left_shoulder2"/>
    <motor gear="25" joint="left_elbow" name="left_elbow"/>
  </actuator>
"""

_HUMANOID_DEFAULT = """
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="1" condim="1" contype="1" margin="0.001" material="geom"/>
    <motor ctrllimited="true" ctrlrange="-.4 .4"/>
  </default>
"""

HUMANOID = RobotSpec(
    name="humanoid",
    asset_xml=_HUMANOID_ASSETS,
    body_xml=_HUMANOID_BODY,
    extra_xml=_HUMANOID_EXTRA,
    default_xml=_HUMANOID_DEFAULT,
    root_body="torso",
    spawn_height=1.4,
    healthy_height_range=(0.95, 2.2),
    foot_bodies=("left_foot", "right_foot"),
    # Flexed hips/knees chosen so the whole-body CoM sits over the feet (the model has no
    # ankles, so balance must come from the hips); also keeps qpos inside the knee range.
    default_joint_pos={
        "right_hip_y": math.radians(-18.5), "left_hip_y": math.radians(-18.5),
        "right_knee": math.radians(-30.0), "left_knee": math.radians(-30.0),
    },
)


# --------------------------------------------------------------------------------------
# Unitree G1 (optional, requires MuJoCo Menagerie)
# --------------------------------------------------------------------------------------

def g1_model_dir() -> Path | None:
    """Locate a MuJoCo Menagerie ``unitree_g1`` directory, or ``None`` if unavailable.

    Searched in order: ``$MENAGERIE_PATH/unitree_g1``, ``$G1_MODEL_DIR``, ``assets/unitree_g1``.
    """
    candidates = []
    if os.environ.get("MENAGERIE_PATH"):
        candidates.append(Path(os.environ["MENAGERIE_PATH"]) / "unitree_g1")
    if os.environ.get("G1_MODEL_DIR"):
        candidates.append(Path(os.environ["G1_MODEL_DIR"]))
    candidates.append(ASSETS_DIR / "unitree_g1")
    for candidate in candidates:
        if (candidate / "g1.xml").is_file():
            return candidate
    return None


def make_g1_spec() -> RobotSpec:
    model_dir = g1_model_dir()
    if model_dir is None:
        raise FileNotFoundError(
            "Unitree G1 model not found. Clone mujoco_menagerie and either set "
            "MENAGERIE_PATH=<menagerie root>, set G1_MODEL_DIR=<path to unitree_g1>, "
            "or copy the unitree_g1 folder into 6-Parkour/assets/."
        )
    return RobotSpec(
        name="g1",
        asset_xml="",
        body_xml="",
        extra_xml="",
        default_xml="",
        root_body="pelvis",
        spawn_height=0.79,
        healthy_height_range=(0.4, 1.4),
        foot_bodies=("left_ankle_roll_link", "right_ankle_roll_link"),
        action_scale=0.25,
        includes=(str(model_dir / "g1.xml"),),
    )


ROBOTS = {"humanoid": lambda: HUMANOID, "g1": make_g1_spec}


def get_robot(name: str) -> RobotSpec:
    if name not in ROBOTS:
        raise KeyError(f"Unknown robot '{name}'. Available: {sorted(ROBOTS)}")
    return ROBOTS[name]()
