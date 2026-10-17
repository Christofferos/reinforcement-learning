"""Runtime behaviour of terrain modules whose parts move on a schedule or react to the robot.

A module that needs more than static geometry or passive physics (a hinge the robot's weight
turns) returns actors with its geometry. The environment calls each actor's

* ``bind(env)`` after compiling a course, to resolve body, joint and geom ids;
* ``reset(env)`` at every reset, after the robot's state is set and before the first forward pass;
* ``before_step(env)`` once per control step, before physics;
* ``holds_stall(env)``, true while waiting is part of the task, which pauses stall termination;
* ``markers(env)`` when rendering, for viewer-only decoration.

Every actor returns at once while the robot is away from its module, so a course costs no
more than its static geometry until the robot gets there.
"""

from __future__ import annotations

import mujoco
import numpy as np

GRAVITY = 9.81


class Actor:
    """Base: a module spanning ``x_range`` along the course."""

    NEAR_MARGIN = 3.0
    """How far before or after its module an actor keeps running."""

    def __init__(self, x_range: tuple[float, float]):
        self.x_range = x_range

    def near(self, env, margin: float | None = None) -> bool:
        margin = self.NEAR_MARGIN if margin is None else margin
        x = float(env.data.qpos[0])
        return self.x_range[0] - margin <= x <= self.x_range[1] + margin

    def bind(self, env) -> None:
        pass

    def reset(self, env) -> None:
        pass

    def before_step(self, env) -> None:
        pass

    def holds_stall(self, env) -> bool:
        return False

    def terminates(self, env) -> str:
        """A termination reason this module imposes now (like ``"too_slow"``), or ``""``."""
        return ""

    def markers(self, env) -> list[dict]:
        return []


class ElevatorActor(Actor):
    """Moves a slide-jointed platform back and forth across a pit: it waits ``dwell`` seconds at
    each end, then eases across in ``travel / speed`` seconds.

    The platform is heavy and its joint position and velocity are set every control step, so
    it moves on schedule whatever stands on it, and its velocity is real to the contact solver:
    friction carries a robot riding it. It only runs while the robot is within
    ``NEAR_MARGIN`` of the module, from a random point of its cycle. While the robot stands at
    the boarding edge with the platform away, ``holds_stall`` pauses stall termination: a
    full cycle is longer than the stall limit.
    """

    def __init__(self, x_range, joint: str, travel: float, speed: float, dwell: float,
                 boarding: tuple[float, float]):
        super().__init__(x_range)
        self.joint = joint
        self.travel = travel
        self.move_time = travel / speed
        self.dwell = dwell
        self.period = 2.0 * (self.move_time + dwell)
        self.boarding = boarding
        self.time = 0.0

    def bind(self, env) -> None:
        joint = env.model.joint(self.joint).id
        self.qpos_adr = int(env.model.jnt_qposadr[joint])
        self.dof_adr = int(env.model.jnt_dofadr[joint])

    def state(self, time: float) -> tuple[float, float]:
        """Platform offset along its slide (0 at the boarding edge) and its velocity."""
        t = time % self.period
        half = self.move_time + self.dwell
        outbound = t < half
        t = t if outbound else t - half
        if t < self.dwell:
            offset, velocity = 0.0, 0.0
        else:
            phase = (t - self.dwell) / self.move_time
            offset = 0.5 * self.travel * (1.0 - np.cos(np.pi * phase))
            velocity = 0.5 * self.travel * np.pi / self.move_time * np.sin(np.pi * phase)
        if not outbound:
            offset, velocity = self.travel - offset, -velocity
        return offset, velocity

    def reset(self, env) -> None:
        self.time = float(env.np_random.uniform(0.0, self.period))
        env.data.qpos[self.qpos_adr], env.data.qvel[self.dof_adr] = self.state(self.time)

    def before_step(self, env) -> None:
        if self.near(env):
            self.time += env.dt
            env.data.qpos[self.qpos_adr], env.data.qvel[self.dof_adr] = self.state(self.time)
        else:
            env.data.qvel[self.dof_adr] = 0.0

    def holds_stall(self, env) -> bool:
        x = float(env.data.qpos[0])
        return self.boarding[0] <= x <= self.boarding[1] and env.data.qpos[self.qpos_adr] > 0.05


class CrumbleActor(Actor):
    """Drops each bridge plank ``delay`` seconds after the robot first touches it.

    Planks are mocap bodies: they hold still until their delay runs out, then fall freely,
    each drifting and tumbling its own way (drawn anew every reset), and are parked far below
    once out of sight.
    """

    FALL_TIME = 2.5

    def __init__(self, x_range, planks: list[str], delay: float):
        super().__init__(x_range)
        self.planks = planks
        self.delay = delay

    def bind(self, env) -> None:
        model = env.model
        self.body_ids = [model.body(name).id for name in self.planks]
        self.mocap_ids = [int(model.body_mocapid[body]) for body in self.body_ids]
        self.home_pos = np.array([model.body_pos[body] for body in self.body_ids])
        self.home_quat = np.array([model.body_quat[body] for body in self.body_ids])
        self.plank_of_geom = {
            geom: index
            for index, body in enumerate(self.body_ids)
            for geom in range(model.ngeom)
            if model.geom_bodyid[geom] == body and model.geom_contype[geom]
        }
        self.robot_geoms = env._robot_geom_ids
        self.touched = np.full(len(self.planks), np.inf)

    def reset(self, env) -> None:
        self.touched[:] = np.inf
        env.data.mocap_pos[self.mocap_ids] = self.home_pos
        env.data.mocap_quat[self.mocap_ids] = self.home_quat
        rng, count = env.np_random, len(self.planks)
        # Per plank: a horizontal drift (mostly sideways, off either edge), and a tumble about
        # a random horizontal axis at a random rate in either direction.
        self.drift = np.column_stack([rng.uniform(-0.3, 0.3, count), rng.uniform(-0.8, 0.8, count), np.zeros(count)])
        heading = rng.uniform(0.0, 2.0 * np.pi, count)
        self.tumble_axis = np.column_stack([np.cos(heading), np.sin(heading), np.zeros(count)])
        self.tumble_rate = rng.uniform(1.0, 4.0, count) * rng.choice((-1.0, 1.0), count)

    def before_step(self, env) -> None:
        if not self.near(env):
            return
        data, now = env.data, float(env.data.time)
        for contact in data.contact[: data.ncon]:
            for plank_geom, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                index = self.plank_of_geom.get(int(plank_geom))
                if index is not None and int(other) in self.robot_geoms and self.touched[index] == np.inf:
                    self.touched[index] = now
        falling = np.flatnonzero(now - self.touched >= self.delay)
        for index in falling:
            t = now - self.touched[index] - self.delay
            mocap = self.mocap_ids[index]
            if t > self.FALL_TIME:
                data.mocap_pos[mocap] = self.home_pos[index] - np.array([0.0, 0.0, 30.0])
                continue
            data.mocap_pos[mocap] = (
                self.home_pos[index] + self.drift[index] * t - np.array([0.0, 0.0, 0.5 * GRAVITY * t * t])
            )
            quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, self.tumble_axis[index], self.tumble_rate[index] * t)
            result = np.zeros(4)
            mujoco.mju_mulQuat(result, quat, self.home_quat[index])
            data.mocap_quat[mocap] = result


class WindActor(Actor):
    """A wind cannon at the side of the course: while the robot is on the module it fires a
    ball every ``interval`` seconds (±40%) that kicks the robot's base sideways by up to
    ``strength`` m/s, the same kick the random pushes give, aimed from the cannon's mouth, and
    blows a steady ``steady`` newtons on the base away from the cannon. The steady wind goes
    through ``qfrc_applied`` on the base's free joint, so water forces (``xfrc_applied``) never
    overwrite it.
    """

    def __init__(self, x_range, mouth: tuple[float, float, float], side: float, interval: float,
                 strength: float, steady: float = 0.0):
        super().__init__(x_range)
        self.steady = steady
        self.blowing = False
        self.mouth = np.asarray(mouth, dtype=float)
        self.side = side
        """+1 when the cannon stands on the course's left (+y), -1 on its right."""
        self.interval = interval
        self.strength = strength
        self.next_push_step = -1
        self.pending = np.zeros(2)
        self.last_push: tuple[int, np.ndarray] | None = None

    def reset(self, env) -> None:
        self.next_push_step = -1
        self.pending = np.zeros(2)
        self.last_push = None
        self.blowing = False

    def _schedule(self, env) -> None:
        rng = env.np_random
        delay = max(1, int(round(self.interval * rng.uniform(0.6, 1.4) / env.dt)))
        self.next_push_step = env._step_count + delay
        away = env.data.qpos[:2] - self.mouth[:2]
        away /= max(np.linalg.norm(away), 1e-9)
        self.pending = rng.uniform(0.5, 1.0) * self.strength * away

    def before_step(self, env) -> None:
        if not self.near(env, margin=0.0):
            self.next_push_step = -1
            if self.blowing:
                env.data.qfrc_applied[1] = 0.0
                self.blowing = False
            return
        if self.steady:
            env.data.qfrc_applied[1] = -self.side * self.steady  # the base's world-frame y force
            self.blowing = True
        if self.next_push_step < 0:
            self._schedule(env)
        elif env._step_count == self.next_push_step:
            from ..mdp import push

            push(env.data, self.pending)
            self.last_push = (env._step_count, self.pending.copy())
            self._schedule(env)

    def markers(self, env) -> list[dict]:
        markers = []
        if self.next_push_step >= 0 or self.last_push is not None:
            markers = env._push_markers((self.next_push_step, self.pending, self.last_push), origin=self.mouth)
        # Turbine blades spinning in the cannon's mouth, faster while it is armed.
        speed = 12.0 if self.next_push_step >= 0 else 3.0
        for blade in range(3):
            angle = speed * float(env.data.time) + blade * 2.0 * np.pi / 3.0
            radial = np.array([np.cos(angle), 0.0, np.sin(angle)])
            axis_y = np.array([0.0, 1.0, 0.0])
            markers.append(dict(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=self.mouth + 0.17 * radial - np.array([0.0, 0.03 * self.side, 0.0]),  # just out of the mouth
                size=np.array([0.17, 0.01, 0.04]),
                mat=np.column_stack([radial, axis_y, np.cross(radial, axis_y)]).ravel(),
                rgba=(0.85, 0.85, 0.9, 1.0), emission=0.2, label="", group=2,
            ))
        return markers


class TrampolineActor(Actor):
    """Launches the robot off a trampoline bed. A landing that hits the bed falling faster
    than ``ARM_SPEED`` along the bed's ``normal`` arms it; when the bed then starts back up,
    the robot's base gets ``GAIN`` times its landing speed along the normal, at most ``launch``
    m/s: a velocity kick like the random pushes'. A bed tilted forward throws forward too. Drops from the deck (2.2-2.6 m/s) get 3.1-3.5 m/s; settling onto the
    sagging bed (about 0.9 m/s) does not arm it, so the robot can stand on it. The spring alone
    returned about half a landing's energy, the robot's knees soaking up the rest, and pushing
    the bed harder only bent the knees further.
    """

    ARM_SPEED = 1.5
    GAIN = 1.4

    def __init__(self, x_range, joint: str, bed_geom: str, launch: float,
                 normal: tuple[float, float, float] = (0.0, 0.0, 1.0)):
        super().__init__(x_range)
        self.joint = joint
        self.bed_geom = bed_geom
        self.launch = launch
        self.normal = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
        self.armed = False
        self.landing_speed = 0.0
        self.launches = 0

    def bind(self, env) -> None:
        joint = env.model.joint(self.joint).id
        self.qpos_adr = int(env.model.jnt_qposadr[joint])
        self.dof_adr = int(env.model.jnt_dofadr[joint])
        self.bed = env.model.geom(self.bed_geom).id
        self.robot_geoms = env._robot_geom_ids

    def reset(self, env) -> None:
        self.armed = False
        self.launches = 0

    def _on_bed(self, data) -> bool:
        return any(
            (c.geom1 == self.bed and c.geom2 in self.robot_geoms) or (c.geom2 == self.bed and c.geom1 in self.robot_geoms)
            for c in data.contact[: data.ncon]
        )

    def before_step(self, env) -> None:
        if not self.near(env, margin=0.5):
            self.armed = False
            return
        data = env.data
        speed = float(data.qvel[0:3] @ self.normal)  # the free joint's linear velocity is world-frame
        if not self.armed:
            self.armed = speed < -self.ARM_SPEED and self._on_bed(data)
            self.landing_speed = -speed
        elif data.qvel[self.dof_adr] > 0.0:  # the bed is on its way back up
            self.armed = False
            if self._on_bed(data):
                data.qvel[0:3] += max(0.0, min(self.launch, self.GAIN * self.landing_speed) - speed) * self.normal
                self.launches += 1


# Seven-segment digits: which of the segments a (top), b (top right), c (bottom right), d (bottom),
# e (bottom left), f (top left) and g (middle) are lit.
SEVEN_SEGMENTS = {
    "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg", "5": "acdfg", "6": "acdefg",
    "7": "abc", "8": "abcdefg", "9": "abcdfg",
}


class RunClockActor(Actor):
    """A sprint's clock: the robot must cross the module fast enough, or the episode ends ``"too_slow"``.

    The clock starts as the robot enters. The limit is the time the run's waypoint speeds ask
    for (each stretch at its waypoint's speed, sprint boost included), divided by
    ``required_share`` of that speed, plus ``GRACE`` seconds to speed up from the previous
    module. Zero-command episodes have no clock. When rendering, a black panel over the middle
    of the module counts the remaining seconds down in red.
    """

    GRACE = 0.3
    PANEL_HEIGHT = 2.2
    """Above the lane, clear of the 1.53 m standing head top."""

    def __init__(self, x_range, y: float, z: float, required_share: float):
        super().__init__(x_range)
        self.y, self.z = y, z
        self.required_share = required_share
        self.start_step: int | None = None
        self.limit: float | None = None
        self.finished = False

    def reset(self, env) -> None:
        self.start_step, self.limit, self.finished = None, None, False

    def allowed_time(self, env) -> float | None:
        """Seconds allowed to cross, from the module's waypoint speeds; None without a command."""
        x0, x1 = self.x_range
        points = [(x, i) for i, x in enumerate(env._waypoints[:, 0]) if x0 < x <= x1 + 1e-6]
        speeds = [float(env._waypoint_speeds[i]) for _, i in points]
        if not points or min(speeds) <= 0.0:
            return None
        edges = [x0] + [x for x, _ in points]
        expected = sum((b - a) / v for a, b, v in zip(edges[:-1], edges[1:], speeds))
        return expected / self.required_share + self.GRACE

    def elapsed(self, env) -> float:
        return 0.0 if self.start_step is None else (env._step_count - self.start_step) * env.dt

    def before_step(self, env) -> None:
        if self.finished:
            return
        x = float(env.data.qpos[0])
        if self.start_step is None and x >= self.x_range[0]:
            self.start_step = env._step_count
            self.limit = self.allowed_time(env)
        if self.start_step is not None and x >= self.x_range[1]:
            self.finished = True

    def terminates(self, env) -> str:
        if self.finished or self.limit is None or self.start_step is None:
            return ""
        return "too_slow" if self.elapsed(env) > self.limit else ""

    def remaining(self, env) -> float | None:
        limit = self.limit if self.limit is not None else self.allowed_time(env)
        return None if limit is None else max(0.0, limit - self.elapsed(env))

    def markers(self, env) -> list[dict]:
        if not self.near(env, margin=6.0):
            return []
        remaining = self.remaining(env)
        if remaining is None:
            return []
        x = (self.x_range[0] + self.x_range[1]) / 2.0
        centre = np.array([x, self.y, self.z + self.PANEL_HEIGHT])
        half = np.array([0.24, 0.24, 0.13])
        markers = [dict(
            type=mujoco.mjtGeom.mjGEOM_BOX, pos=centre, size=half,
            rgba=(0.02, 0.02, 0.02, 1.0), emission=0.0, label="", group=2,
        )]
        # The same reading on all four sides, so the robot's own camera and the side-on default
        # camera both see it: (outward normal, the viewer's right) per side.
        sides = (((-1, 0, 0), (0, -1, 0)), ((1, 0, 0), (0, 1, 0)), ((0, -1, 0), (1, 0, 0)), ((0, 1, 0), (-1, 0, 0)))
        for normal, right in sides:
            normal, right = np.asarray(normal, float), np.asarray(right, float)
            face = centre + normal * (np.abs(normal) @ half + 0.004)
            markers += self._digits(f"{min(remaining, 9.9):.1f}", face, normal, right)
        return markers

    @staticmethod
    def _digits(text: str, face: np.ndarray, normal: np.ndarray, right: np.ndarray) -> list[dict]:
        """Red seven-segment boxes spelling ``text`` (digits and one point) centred on ``face``."""
        red = dict(type=mujoco.mjtGeom.mjGEOM_BOX, rgba=(1.0, 0.05, 0.03, 1.0), emission=1.0, label="", group=2)
        width, height, thick = 0.09, 0.18, 0.016
        pitch = width + 0.05
        segments = {  # centre (right, up) and half sizes (right, up) per segment
            "a": ((0.0, height / 2), (width / 2, thick / 2)), "g": ((0.0, 0.0), (width / 2, thick / 2)),
            "d": ((0.0, -height / 2), (width / 2, thick / 2)),
            "b": ((width / 2, height / 4), (thick / 2, height / 4)), "c": ((width / 2, -height / 4), (thick / 2, height / 4)),
            "f": ((-width / 2, height / 4), (thick / 2, height / 4)), "e": ((-width / 2, -height / 4), (thick / 2, height / 4)),
        }
        up = np.array([0.0, 0.0, 1.0])
        chars = [ch for ch in text if ch != "."]
        first = -pitch * (len(chars) - 1) / 2.0 - 0.02  # leftmost digit, along the viewer's right

        def box(along: float, rise: float, half_along: float, half_rise: float) -> dict:
            size = 0.003 * np.abs(normal) + half_along * np.abs(right) + half_rise * up
            return dict(red, pos=face + right * along + up * rise, size=size)

        markers = []
        for k, ch in enumerate(chars):
            centre = first + k * pitch
            for name in SEVEN_SEGMENTS[ch]:
                (along, rise), (half_along, half_rise) = segments[name]
                markers.append(box(centre + along, rise, half_along, half_rise))
            if k == len(chars) - 2:  # the decimal point, after the whole seconds
                markers.append(box(centre + pitch / 2, -height / 2, thick / 2, thick / 2))
        return markers
