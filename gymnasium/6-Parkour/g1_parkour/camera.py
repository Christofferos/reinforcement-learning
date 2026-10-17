"""A stabilised first-person camera: render-only, nothing in the simulation depends on it.

The model's ``egocentric`` camera is fixed to the torso, so it copies every degree of the
torso's step-to-step pitch, roll and yaw wobble, and a few degrees of rotation swing the whole
picture. ``StableEyeCamera`` keeps a head's steadiness with the viewer's free camera:

* it looks along the course, the direction of the path segment the robot is on, not where the
  torso points. Walking, the humanoid's torso twists ±9° every stride (1.27 Hz) and its path
  weaves left and right (about 0.5 Hz); any heading taken from the torso carries both. The
  course direction changes only where the course turns, smoothed over 0.5 s. Without a course
  heading (``heading=None``) it falls back to the torso's, smoothed.
* forward motion is followed exactly, so the view never lags into the robot's own head;
* the eye's height follows the torso's almost directly, its bounce exaggerated to 1.56x against
  the torso's slow (0.5 s) height so footsteps and landings read clearly, and its sideways
  position follows through a light 0.09 s smoothing; neither may leave the head by more than
  0.15 m up or down or 0.25 m sideways;
* on flat ground it glances 10-15° up every 3-8 s for 0.5-2 s, and a glance once started runs
  to its end before anything else turns the view; it looks 15° down while the ground scan has
  a magenta (clipped) reading, a pit, gap or drop ahead, or ground more than 0.3 m above its
  floor, a hurdle, climb or flight of stairs; and 35° up while the upward scan sees a ceiling
  (that one wins over looking down), easing in and out over 0.3 s;
* pitch is a fixed 30° look-down plus 48% of the torso's pitch, and the picture tilts with
  48% of its sideways lean, as a head rolls with the body, both smoothed over 0.15 s. Walking on
  flat ground the course 2 m ahead moves 29 cm/s across the view (22 of it up and down); a
  camera fixed to the torso made it 616 cm/s, nearly all from turning with the torso's twist,
  and the steadiest version 2.6;
"""

from __future__ import annotations

import numpy as np

STABLE_EGOCENTRIC = "egocentric"
"""The first-person camera's name: a model camera on the ``EYE_MOUNT`` mocap body, so the
viewer's TAB reaches it; the environment moves the mount to the steadied eye every frame."""
EYE_MOUNT = "egocentric_eye"
LEGACY_NAMES = {"egocentric_stable": STABLE_EGOCENTRIC}
"""Older ``--camera`` names for this view."""


def camera_frame(azimuth: float, elevation: float, roll: float = 0.0) -> np.ndarray:
    """World rotation (columns: camera x, y, z) of a camera looking along ``azimuth`` and
    ``elevation``, rolled ``roll`` (degrees, positive tilting the picture's top to the left, as a
    head leaning left). A MuJoCo camera looks down its -z axis, +y up."""
    az, el, tilt = np.radians(azimuth), np.radians(elevation), np.radians(roll)
    look = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    level_up = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])
    right = np.cross(look, level_up)
    up = level_up * np.cos(tilt) - right * np.sin(tilt)
    return np.column_stack([np.cross(up, -look), up, -look])


class StableEyeCamera:
    EYE_OFFSET = np.array([0.02, 0.0, -0.10])
    """Eye position relative to the torso, in the view's heading frame: at the chest (it was
    0.2 m ahead and 0.15 m up, then 0.12 and 0.10, 0.05 and 0.05, 0.05 and -0.10). The body
    never shows: MuJoCo draws nothing nearer a camera than 1% of the scene's size, about 0.6 m."""
    BOB_GAIN = 1.56
    """The eye's bounce against the torso's slow height: 1 follows it, more exaggerates it, so
    each footstep and landing reads clearly as up-and-down."""
    BOB_BASE_TIME_CONSTANT = 0.5
    """Seconds, for the slow height the bounce is measured from."""
    BOB_TIME_CONSTANT = 0.03
    """Seconds of smoothing on the eye's height, only against simulation jitter."""
    MAX_BOB_LAG = 0.15
    """Metres the eye may trail the head up or down."""
    SWAY_TIME_CONSTANT = 0.09
    """Seconds; keeps about 88% of a 1 Hz sway."""
    MAX_SWAY_LAG = 0.25
    """Metres the eye may trail the head sideways."""
    YAW_TIME_CONSTANT = 0.3
    """Seconds, for the view heading: the course's, plus ``TWIST_SHARE`` of the torso's turn."""
    TWIST_SHARE = 0.18
    """Fraction of the torso's turn away from the course direction the view follows: its
    once-a-stride ±9° twist comes through as about ±0.6°. Turning moves the whole picture, so
    a little goes far: a quarter made the course 2 m ahead swing 35 cm/s sideways."""
    TORSO_YAW_TIME_CONSTANT = 1.0
    """Seconds, for the torso heading used without a course heading."""
    LOOK_DOWN = 30.0
    """Degrees below the horizon: the middle of the view meets flat ground about 2.5 m ahead."""
    PITCH_SHARE = 0.48
    """Fraction of the torso's pitch passed on to the view."""
    PITCH_TIME_CONSTANT = 0.15
    ROLL_SHARE = 0.48
    """Fraction of the torso's sideways lean the picture tilts with, as a head rolls with the body."""
    ROLL_TIME_CONSTANT = 0.15
    LOOK_DISTANCE = 2.0
    """The free camera orbits a point; it is put this far ahead, with the camera at the eye."""
    GLANCE_GAP = (3.0, 8.0)
    """Seconds between random glances up."""
    GLANCE_ANGLE = (10.0, 15.0)
    """Degrees a random glance turns the view up."""
    GLANCE_DURATION = (0.5, 2.0)
    """Seconds a random glance holds before the view eases back."""
    CUE_DOWN_ANGLE = 15.0
    """Degrees the view turns down at a magenta ground-scan reading (a pit, gap or drop) or
    ground more than ``LOOK_DOWN_RISE`` above the floor."""
    CUE_UP_ANGLE = 35.0
    """Degrees the view turns up while the upward scan sees a ceiling, and only then; it wins over
    looking down."""
    LOOK_DOWN_RISE = 0.3
    """Metres above the floor the robot stands on that ground in the scan makes it look down:
    a hurdle, a climb, a flight of stairs."""
    FLAT_TOLERANCE = 0.05
    """Metres the scanned ground may vary for the view to count it as flat, where it glances."""
    GAZE_TIME_CONSTANT = 0.3
    """Seconds to ease into and out of a glance or a cue."""

    def __init__(self, seed: int = 0):
        self.time: float | None = None
        self.rng = np.random.default_rng(seed)
        """Its own randomness: drawing on the environment's would let rendering change a run."""
        self.gaze = 0.0
        """Degrees the view is turned up (positive) or down from its usual look-down."""
        self.glance, self.glance_until, self.next_glance = 0.0, -np.inf, np.inf
        """The glance under way (degrees up, until when) and when the next may start."""
        self.roll = 0.0
        """Degrees the picture tilts (``camera_frame``'s ``roll``), updated by ``update``."""

    def update(self, data, root_body_id: int, heading: float | None = None, *,
               look_down: bool = False, look_up: bool = False,
               glance: bool = True) -> tuple[np.ndarray, float, float, float]:
        """Advance to ``data.time`` and return the camera's lookat, distance, azimuth and
        elevation (degrees), looking along ``heading`` (radians; the course's direction) or,
        without one, the torso's. ``look_down`` and ``look_up`` are the scans' cues (see
        ``CUE_DOWN_ANGLE`` and ``CUE_UP_ANGLE``). While ``glance`` holds (the ground scan is
        flat) it glances up now and then; a glance once started runs to its end, and no cue
        turns the view until it has. A first call, or time running backwards (a reset), snaps."""
        position = data.xpos[root_body_id]
        rotation = data.xmat[root_body_id].reshape(3, 3)
        forward = rotation[:, 0]
        torso_pitch = float(np.degrees(np.arcsin(np.clip(-forward[2], -1.0, 1.0))))  # nose down > 0
        torso_roll = float(np.degrees(np.arcsin(np.clip(-rotation[2, 1], -1.0, 1.0))))  # left side down > 0
        torso_yaw = float(np.arctan2(forward[1], forward[0]))
        if heading is None:
            goal = torso_yaw
        else:
            goal = float(heading) + self.TWIST_SHARE * float(np.angle(np.exp(1j * (torso_yaw - heading))))
        yaw_time_constant = self.TORSO_YAW_TIME_CONSTANT if heading is None else self.YAW_TIME_CONSTANT
        now = float(data.time)
        snap = self.time is None or now < self.time
        dt = 0.0 if snap else now - self.time

        def follow(current, target, tau):
            return current + (target - current) * (1.0 - np.exp(-dt / tau))

        def leashed(behind, tau, leash):
            """How far to move towards a point ``behind`` away: smoothed over ``tau``, but never
            left trailing it by more than ``leash``."""
            moved = follow(0.0, behind, tau)
            return behind - float(np.clip(behind - moved, -leash, leash))

        if snap:
            self.gaze, self.glance, self.glance_until = 0.0, 0.0, -np.inf
            self.next_glance = now + self.rng.uniform(*self.GLANCE_GAP)
        elif now >= self.next_glance and glance and not (look_up or look_down):
            self.glance = self.rng.uniform(*self.GLANCE_ANGLE)
            self.glance_until = now + self.rng.uniform(*self.GLANCE_DURATION)
            self.next_glance = self.glance_until + self.rng.uniform(*self.GLANCE_GAP)
        if now < self.glance_until:
            gaze_goal = self.glance  # a glance under way finishes before anything else turns the view
        elif look_up:
            gaze_goal = self.CUE_UP_ANGLE
        elif look_down:
            gaze_goal = -self.CUE_DOWN_ANGLE
        else:
            gaze_goal = 0.0
        if not snap:
            self.gaze = follow(self.gaze, gaze_goal, self.GAZE_TIME_CONSTANT)

        if snap:
            self.yaw, self.pitch, self.roll = goal, torso_pitch, self.ROLL_SHARE * torso_roll
        else:
            self.yaw = follow(self.yaw, self.yaw + float(np.angle(np.exp(1j * (goal - self.yaw)))), yaw_time_constant)
            self.pitch = follow(self.pitch, torso_pitch, self.PITCH_TIME_CONSTANT)
            self.roll = follow(self.roll, self.ROLL_SHARE * torso_roll, self.ROLL_TIME_CONSTANT)
        cos_yaw, sin_yaw = np.cos(self.yaw), np.sin(self.yaw)
        target = position + np.array([[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]]) @ self.EYE_OFFSET
        if snap:
            self.eye = target.copy()
            self.height = self.base = float(target[2])
        else:
            along = np.array([cos_yaw, sin_yaw, 0.0])
            across = np.array([-sin_yaw, cos_yaw, 0.0])
            lag = target - self.eye
            self.eye = (
                self.eye
                + along * (lag @ along)
                + across * leashed(lag @ across, self.SWAY_TIME_CONSTANT, self.MAX_SWAY_LAG)
            )
            self.height = follow(self.height, float(target[2]), self.BOB_TIME_CONSTANT)
            self.base = follow(self.base, float(target[2]), self.BOB_BASE_TIME_CONSTANT)
            bounced = self.base + self.BOB_GAIN * (self.height - self.base)
            self.eye[2] = float(np.clip(bounced, target[2] - self.MAX_BOB_LAG, target[2] + self.MAX_BOB_LAG))
        self.time = now

        azimuth = np.degrees(self.yaw)
        elevation = -(self.LOOK_DOWN + self.PITCH_SHARE * self.pitch) + self.gaze
        az, el = np.radians(azimuth), np.radians(elevation)
        look = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        return self.eye + self.LOOK_DISTANCE * look, self.LOOK_DISTANCE, float(azimuth), float(elevation)
