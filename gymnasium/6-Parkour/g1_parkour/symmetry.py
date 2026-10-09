"""Left/right mirror maps for symmetric policy learning.

The mirror reflects the world in the vertical plane through the robot along its heading:
body-frame vectors flip their lateral (y) component, angular velocity (a pseudovector) flips
roll and yaw rates, left and right joints swap places, and the yaw-aligned height scan swaps
its lateral columns. :class:`g1_parkour.symmetric_ppo.SymmetricPPO` uses these maps to train
policies that act the same on both sides (Yu et al. 2018; Abdolhosseini et al. 2019;
Mittal et al. 2024).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

REFLECTION = np.diag([1.0, -1.0, 1.0])
"""Reflection y -> -y, in the world at a zero-yaw reference pose and in the body frame."""


@dataclass(frozen=True, eq=False)
class Mirror:
    """A signed permutation: ``mirror(x)[..., k] = sign[k] * x[..., perm[k]]``."""

    perm: np.ndarray
    sign: np.ndarray

    def __post_init__(self) -> None:
        perm = np.asarray(self.perm, dtype=np.int64)
        sign = np.asarray(self.sign, dtype=np.float64)
        object.__setattr__(self, "perm", perm)
        object.__setattr__(self, "sign", sign)
        size = len(perm)
        if perm.shape != (size,) or sign.shape != (size,) or sorted(perm) != list(range(size)):
            raise ValueError("a mirror needs a permutation and one sign per element")
        if not np.all(np.isin(sign, (-1.0, 1.0))):
            raise ValueError("mirror signs must be +1 or -1")
        # Mirroring twice must give back the original vector.
        if not np.array_equal(perm[perm], np.arange(size)) or not np.all(sign * sign[perm] == 1.0):
            raise ValueError("a mirror must be its own inverse")

    def __len__(self) -> int:
        return len(self.perm)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)[..., self.perm] * self.sign

    @classmethod
    def identity(cls, size: int) -> "Mirror":
        return cls(np.arange(size), np.ones(size))

    @classmethod
    def diagonal(cls, signs) -> "Mirror":
        return cls(np.arange(len(signs)), np.asarray(signs, dtype=np.float64))

    @classmethod
    def concat(cls, parts: list["Mirror"]) -> "Mirror":
        offsets = np.cumsum([0] + [len(part) for part in parts[:-1]])
        perm = np.concatenate([part.perm + offset for part, offset in zip(parts, offsets)])
        return cls(perm, np.concatenate([part.sign for part in parts]))


VECTOR = Mirror.diagonal([1.0, -1.0, 1.0])
"""A body-frame vector (velocity, gravity, offset to a waypoint)."""
PSEUDOVECTOR = Mirror.diagonal([-1.0, 1.0, -1.0])
"""A body-frame angular velocity: roll and yaw rates change sign, pitch rate does not."""


def partner_name(name: str) -> str:
    """The mirrored counterpart of a joint or body name (``left_*`` <-> ``right_*``)."""
    if "left" in name:
        return name.replace("left", "right")
    if "right" in name:
        return name.replace("right", "left")
    return name


def joint_mirror(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Partner joint id and sign for every joint: mirrored angle of ``j`` = sign[j] * angle of partner[j].

    Joint axes are compared in the world at the model's reference pose (zero joint angles,
    upright, zero yaw), where the body frame equals the world frame. A rotation by ``t`` about
    axis ``a`` mirrors to a rotation by ``-t`` about ``REFLECTION @ a``, so the partner moves by
    ``+t`` when its axis is ``-REFLECTION @ a`` and by ``-t`` when it is ``REFLECTION @ a``.
    Raises if a joint's axis or range is not the mirror image of its partner's.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    for joint in range(model.njnt):
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            adr = model.jnt_qposadr[joint]
            data.qpos[adr + 3: adr + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_kinematics(model, data)

    partner = np.arange(model.njnt)
    sign = np.ones(model.njnt)
    for joint in range(model.njnt):
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = model.joint(joint).name
        other = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, partner_name(name))
        if other < 0:
            raise ValueError(f"joint '{name}' has no mirrored partner '{partner_name(name)}'")
        alignment = float(data.xaxis[other] @ REFLECTION @ data.xaxis[joint])
        if abs(abs(alignment) - 1.0) > 1e-6:
            raise ValueError(f"joints '{name}' and '{model.joint(other).name}' do not have mirrored axes")
        partner[joint], sign[joint] = other, -np.sign(alignment)
        if model.jnt_limited[joint]:
            mirrored_range = np.sort(sign[joint] * model.jnt_range[joint])
            if not np.allclose(mirrored_range, model.jnt_range[other], atol=1e-6):
                raise ValueError(f"joint ranges of '{name}' and '{model.joint(other).name}' are not mirrored")
    return partner, sign


def hinge_mirror(model: mujoco.MjModel) -> Mirror:
    """Mirror of the joint block of the observation (``qpos[7:]`` and ``qvel[6:]``)."""
    hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    if model.nq - 7 != len(hinges) or model.nv - 6 != len(hinges):
        raise ValueError("joint mirroring expects one free root joint followed by hinge joints")
    partner, sign = joint_mirror(model)
    slot = {joint: index for index, joint in enumerate(hinges)}
    return Mirror([slot[partner[j]] for j in hinges], [sign[j] for j in hinges])


def actuator_mirror(model: mujoco.MjModel) -> Mirror:
    """Mirror of the action vector: actuators of partner joints swap, with the joint's sign.

    Raises unless partner actuators have equal gear and control ranges symmetric about zero,
    because only then does a signed swap of actions give the mirrored joint commands.
    """
    partner, sign = joint_mirror(model)
    joints = model.actuator_trnid[:, 0]
    actuator_of = {int(joint): index for index, joint in enumerate(joints)}
    perm = []
    for index, joint in enumerate(joints):
        other = actuator_of.get(int(partner[joint]))
        if other is None:
            raise ValueError(f"actuator '{model.actuator(index).name}' has no mirrored partner")
        low, high = model.actuator_ctrlrange[index]
        if not np.isclose(low, -high) or not np.allclose(model.actuator_ctrlrange[other], (low, high)):
            raise ValueError("mirrored actuators need equal control ranges symmetric about zero")
        if not np.allclose(model.actuator_gear[index], model.actuator_gear[other]):
            raise ValueError(f"actuators '{model.actuator(index).name}' and its partner differ in gear")
        perm.append(other)
    return Mirror(perm, sign[joints])


def lateral_mirror(points: np.ndarray) -> Mirror:
    """Mirror of a scan sampled at heading-frame ``points``: point (x, y) takes the value of (x, -y)."""
    perm = []
    for x, y in points:
        match = np.flatnonzero(np.all(np.isclose(points, (x, -y), atol=1e-9), axis=1))
        if len(match) != 1:
            raise ValueError("scan points must be symmetric about the heading axis")
        perm.append(int(match[0]))
    return Mirror(perm, np.ones(len(points)))
