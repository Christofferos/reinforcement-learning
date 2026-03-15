"""
Hide and Seek MuJoCo Environment
=================================
A modern recreation of OpenAI's "Emergent Tool Use From Multi-Agent Autocurricula"
using gymnasium + mujoco + PettingZoo APIs.

Game Rules:
- Hiders and seekers compete in a walled arena with moveable boxes and ramps.
- Each episode has a preparation phase (seekers frozen) and a play phase.
- Hiders receive +1 if ALL hiders are hidden, -1 otherwise.
- Seekers receive +1 if ANY seeker sees a hider, -1 otherwise.
- Agents can grab and push boxes and ramps to build barricades.
- Line-of-sight is blocked by walls and boxes.

References:
    Baker et al. "Emergent Tool Use From Multi-Agent Autocurricula" (2019)
    https://arxiv.org/abs/1909.07528
"""

from __future__ import annotations

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from functools import lru_cache

# Type hints
from numpy.typing import NDArray

# Procedural world generator
from worldgen import generate_arena

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_DIR)  # hide-n-seek/
_DEFAULT_XML = os.path.join(_PROJECT_ROOT, "assets", "hide_and_seek.xml")

HIDER_NAMES = ["hider_0", "hider_1"]
SEEKER_NAMES = ["seeker_0", "seeker_1"]
AGENT_NAMES = HIDER_NAMES + SEEKER_NAMES

# Max possible objects (used for fixed-size obs/state vectors)
MAX_BOXES = 5
MAX_RAMPS = 2

# Default names (for backward compat with static XML)
BOX_NAMES = ["box_0", "box_1", "box_2"]
RAMP_NAMES = ["ramp_0"]
OBJECT_NAMES = BOX_NAMES + RAMP_NAMES

N_HIDERS = len(HIDER_NAMES)
N_SEEKERS = len(SEEKER_NAMES)
N_AGENTS = N_HIDERS + N_SEEKERS
N_BOXES = len(BOX_NAMES)
N_RAMPS = len(RAMP_NAMES)

# Agent display labels (must match those in worldgen.py / hide_and_seek.xml)
AGENT_LABELS = {
    "hider_0": "Hider Jacob",
    "hider_1": "Hider Lucas",
    "seeker_0": "Seeker Mikael",
    "seeker_1": "Seeker Kriss",
}

# Orange highlight for grab indicator (RGBA)
_GRAB_RGBA = np.array([1.0, 0.55, 0.0, 1.0], dtype=np.float32)

# Bright magenta/pink for "hider is seen" indicator (RGBA)
_SEEN_RGBA = np.array([1.0, 0.2, 0.6, 1.0], dtype=np.float32)

# Seekers frozen & blind during prep phase — dim grey to show inactive
_PREP_RGBA = np.array([0.45, 0.45, 0.50, 0.6], dtype=np.float32)

# ──────────────────────────────────────────────────────────
# Reward shaping coefficients
# ──────────────────────────────────────────────────────────
# Small additive bonuses on top of the base binary reward to provide
# a learning gradient for agents on single-GPU hardware.
# All coefficients are intentionally small so the base game reward
# dominates once agents learn to hide / seek.

SHAPE_PREP_MOVEMENT   = 0.001   # hider bonus per unit speed during prep
SHAPE_GRAB_AND_MOVE   = 0.005   # bonus per step for grabbing + moving an object
SHAPE_HIDER_NEAR_COVER = 0.010  # hider bonus during prep if lidar sees wall ≤ 1.5m
SHAPE_SEEKER_EXPLORE  = 0.002   # seeker bonus per metre moved during play phase
SHAPE_INDIVIDUAL_BLEND = 0.20   # fraction of per-agent reward blended into team reward
_COVER_LIDAR_THRESHOLD = 1.5 / 18.0  # 1.5m / lidar_max_dist (normalised)
_GRAB_MOVE_VEL_THRESH  = 0.05  # min object velocity (m/step) to count as "moving"


# ──────────────────────────────────────────────────────────
# Helper: Ray-casting line-of-sight
# ──────────────────────────────────────────────────────────

# Fixed ray height — cast horizontally at agent centre height so that
# body rotation, Z jitter, and floor-plane hits don't cause false negatives.
_RAY_Z = 0.35  # matches AGENT_Z / AGENT_RADIUS in worldgen.py


def _can_see(model: mujoco.MjModel, data: mujoco.MjData,
             pos_a: NDArray, pos_b: NDArray,
             exclude_geom_ids: set | None = None) -> bool:
    """
    Check line-of-sight between two agents via a **horizontal** MuJoCo raycast.

    Both the origin and target are projected to a fixed Z height (_RAY_Z) so
    that the ray travels perfectly flat.  This avoids false negatives caused by
    the ray clipping the floor when agents are at slightly different heights.

    If the ray hits an excluded geom (agent sphere / ring cylinder), we
    advance past it and re-cast to handle stacked excluded geoms correctly.
    """
    # Flatten to horizontal ray at fixed height
    origin = np.array([pos_a[0], pos_a[1], _RAY_Z], dtype=np.float64)
    target = np.array([pos_b[0], pos_b[1], _RAY_Z], dtype=np.float64)

    direction = target - origin
    total_dist = np.linalg.norm(direction)
    if total_dist < 1e-6:
        return True
    direction = direction / total_dist

    geomid = np.array([-1], dtype=np.int32)
    travelled = 0.0
    max_bounces = 8  # safety limit for re-casts through excluded geoms

    for _ in range(max_bounces):
        pnt = origin + direction * travelled
        hit_dist = mujoco.mj_ray(
            model, data,
            pnt=pnt,
            vec=direction,
            geomgroup=None,
            flg_static=1,
            bodyexclude=-1,
            geomid=geomid,
        )

        if hit_dist < 0:
            # No hit at all — clear line of sight
            return True

        abs_hit = travelled + hit_dist

        # Hit is beyond (or at) the target — nothing blocking
        if abs_hit >= total_dist - 0.05:
            return True

        # Hit an excluded geom (agent sphere/ring) — skip past it
        if exclude_geom_ids and geomid[0] in exclude_geom_ids:
            travelled = abs_hit + 0.02  # advance just past the hit point
            if travelled >= total_dist:
                return True
            continue

        # Hit a real obstacle (wall, box, ramp) before reaching the target
        return False

    # Exhausted bounces — assume blocked (shouldn't happen normally)
    return False


# ──────────────────────────────────────────────────────────
# Lidar helper
# ──────────────────────────────────────────────────────────

def _compute_lidar(model: mujoco.MjModel, data: mujoco.MjData,
                   origin: NDArray, n_rays: int, max_dist: float,
                   bodyexclude: int = -1,
                   exclude_geom_ids: set | None = None) -> NDArray:
    """
    Cast `n_rays` evenly spaced around the agent (in the XY plane) and
    return normalised distances [0-1].  1 = nothing hit within max_dist.

    The lidar senses **obstacles only** (walls, boxes, ramps).  Agent geoms
    are skipped so that other agents don't shadow obstacles behind them.

    Args:
        bodyexclude: MuJoCo body ID to exclude from ray hits (agent's own body).
        exclude_geom_ids: Set of geom IDs to skip through (all agent geoms).
    """
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    readings = np.ones(n_rays, dtype=np.float32)

    for i, angle in enumerate(angles):
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        geomid = np.array([-1], dtype=np.int32)
        travelled = 0.0

        for _bounce in range(6):  # up to 6 re-casts through agent geoms
            pnt = origin.astype(np.float64) + direction * travelled
            hit_dist = mujoco.mj_ray(
                model, data,
                pnt=pnt,
                vec=direction,
                geomgroup=None,
                flg_static=1,
                bodyexclude=bodyexclude,
                geomid=geomid,
            )

            if hit_dist < 0:
                break  # no hit — reading stays 1.0

            abs_hit = travelled + hit_dist

            if abs_hit > max_dist:
                break  # beyond range — reading stays 1.0

            # Skip agent geoms (other agents' spheres/rings)
            if exclude_geom_ids and geomid[0] in exclude_geom_ids:
                travelled = abs_hit + 0.02
                continue

            # Real obstacle hit
            readings[i] = abs_hit / max_dist
            break

    return readings


# ──────────────────────────────────────────────────────────
# Main Environment
# ──────────────────────────────────────────────────────────

class HideAndSeekEnv(gym.Env):
    """
    Multi-agent Hide & Seek environment.

    This exposes a *vectorised* interface: `step()` takes a dict of actions
    keyed by agent name and returns dicts of observations, rewards, etc.
    Compatible with PettingZoo parallel-env wrapping (see `pettingzoo_wrapper.py`).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        xml_path: str = _DEFAULT_XML,
        horizon: int = 240,
        prep_fraction: float = 0.4,
        n_substeps: int = 15,
        reward_type: str = "joint_zero_sum",
        reward_scale: float = 1.0,
        lidar_n_rays: int = 30,
        lidar_max_dist: float = 18.0,  # ≥ arena diagonal (~17m) so rays cover full map
        agent_obs_radius: float = 100.0,  # effectively unlimited — pure line-of-sight (OpenAI style)
        box_obs_radius: float = 6.0,
        grab_radius: float = 0.8,
        movement_scale: float = 1.0,
        render_mode: str | None = None,
        procedural: bool = True,
    ):
        super().__init__()

        # ── Procedural mode flag ──
        self.procedural = procedural

        # ── MuJoCo model ──
        # In procedural mode we generate a fresh XML on every reset.
        # The first model is built now so that spaces/dimensions are available.
        if self.procedural:
            self._worldgen_rng = np.random.default_rng()
            xml, self._last_meta = generate_arena(self._worldgen_rng)
            self.model = mujoco.MjModel.from_xml_string(xml)
            self._cur_n_boxes = self._last_meta["n_boxes"]
            self._cur_n_ramps = self._last_meta["n_ramps"]
        else:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self._last_meta = None
            self._cur_n_boxes = N_BOXES
            self._cur_n_ramps = N_RAMPS

        self.data = mujoco.MjData(self.model)
        self.n_substeps = n_substeps

        # ── Episode ──
        self.horizon = horizon
        self.prep_steps = int(horizon * prep_fraction)
        self.current_step = 0

        # ── Reward ──
        self.reward_type = reward_type
        self.reward_scale = reward_scale

        # ── Sensing ──
        self.lidar_n_rays = lidar_n_rays
        self.lidar_max_dist = lidar_max_dist
        self.agent_obs_radius = agent_obs_radius
        self.box_obs_radius = box_obs_radius

        # ── Manipulation ──
        self.grab_radius = grab_radius
        self.movement_scale = movement_scale

        # ── Object state (use MAX sizes for fixed buffers) ──
        # Which agent grabbed which object (-1 = nobody)
        self.box_grabbed_by = np.full(MAX_BOXES, -1, dtype=np.int32)
        self.ramp_grabbed_by = np.full(MAX_RAMPS, -1, dtype=np.int32)

        # ── Rendering ──
        self.render_mode = render_mode
        self._renderer = None
        self._viewer_handle = None
        if render_mode == "human":
            self._init_renderer()

        # ── Body/geom id caches ──
        self._cache_ids()

        # ── Spaces (use MAX boxes/ramps for fixed obs size) ──
        obs_size = self._obs_size()
        self.observation_space = spaces.Dict({
            name: spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)
            for name in AGENT_NAMES
        })

        # Actions: [move_x, move_y, grab]  (continuous)
        self.action_space = spaces.Dict({
            name: spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
            for name in AGENT_NAMES
        })

        # Global state (for centralised critic)
        self._global_state_size = self._compute_global_state_size()

        self.possible_agents = list(AGENT_NAMES)
        self.agents = list(AGENT_NAMES)

    # ──────────────────────────────────────────────
    # ID caching
    # ──────────────────────────────────────────────

    def _cache_ids(self):
        """Cache MuJoCo body/geom IDs for fast lookup (adapts to current model)."""
        self._agent_body_ids = {}
        self._agent_geom_ids = {}
        for name in AGENT_NAMES:
            self._agent_body_ids[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self._agent_geom_ids[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")

        # Collect ALL geom IDs belonging to agent bodies (sphere + ring cylinder)
        # so raycasts can exclude them all — not just the named sphere geom.
        self._all_agent_geom_ids: set[int] = set()
        agent_body_id_set = set(self._agent_body_ids.values())
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] in agent_body_id_set:
                self._all_agent_geom_ids.add(gid)

        # Boxes — only those present in current model
        self._cur_box_names = [f"box_{i}" for i in range(self._cur_n_boxes)]
        self._box_body_ids = {}
        self._box_geom_ids = {}
        for name in self._cur_box_names:
            self._box_body_ids[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self._box_geom_ids[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")

        # Ramps — only those present in current model
        self._cur_ramp_names = [f"ramp_{i}" for i in range(self._cur_n_ramps)]
        self._ramp_body_ids = {}
        self._ramp_geom_ids = {}
        for name in self._cur_ramp_names:
            self._ramp_body_ids[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self._ramp_geom_ids[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")

        # Actuator IDs (for XY movement)
        self._actuator_ids = {}
        for name in AGENT_NAMES:
            self._actuator_ids[name] = (
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_x"),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_y"),
            )

        # Save original RGBA colours so we can restore after grab highlight
        self._original_rgba = {}
        for name in AGENT_NAMES:
            gid = self._agent_geom_ids[name]
            self._original_rgba[name] = self.model.geom_rgba[gid].copy()
        for bname in self._cur_box_names:
            gid = self._box_geom_ids[bname]
            self._original_rgba[bname] = self.model.geom_rgba[gid].copy()
        for rname in self._cur_ramp_names:
            gid = self._ramp_geom_ids[rname]
            self._original_rgba[rname] = self.model.geom_rgba[gid].copy()

    # ──────────────────────────────────────────────
    # Observation helpers
    # ──────────────────────────────────────────────

    def _obs_size(self) -> int:
        """Calculate observation vector size per agent (fixed, using MAX object counts)."""
        # Self: pos(3) + vel(3) + is_hider(1) + prep_phase(1) + prep_remaining(1) = 9
        self_obs = 9
        # Other agents (N_AGENTS-1): rel_pos(3) + rel_vel(3) + visible(1) + is_hider(1) = 8 each
        other_agents_obs = (N_AGENTS - 1) * 8
        # Boxes (MAX_BOXES): rel_pos(3) + grabbed(1) + exists(1) = 5 each
        box_obs = MAX_BOXES * 5
        # Ramps (MAX_RAMPS): rel_pos(3) + grabbed(1) + exists(1) = 5 each
        ramp_obs = MAX_RAMPS * 5
        # Lidar
        lidar_obs = self.lidar_n_rays
        return self_obs + other_agents_obs + box_obs + ramp_obs + lidar_obs

    def _compute_global_state_size(self) -> int:
        """Size of the global state vector (for MAPPO centralized critic, fixed)."""
        # All agents: pos(3) + vel(3) = 6 each
        # All boxes (MAX): pos(3) + vel(3) + exists(1) = 7 each
        # All ramps (MAX): pos(3) + grabbed(1) + exists(1) = 5 each
        # Phase info: prep(1) + step_frac(1) = 2
        return N_AGENTS * 6 + MAX_BOXES * 7 + MAX_RAMPS * 5 + 2

    def _get_body_pos(self, body_id: int) -> NDArray:
        return self.data.xpos[body_id].copy()

    def _get_body_vel(self, body_id: int) -> NDArray:
        """Get body velocity (linear component of cvel)."""
        # cvel is [angular(3), linear(3)]
        return self.data.cvel[body_id, 3:6].copy()

    def _agent_positions(self) -> dict[str, NDArray]:
        return {name: self._get_body_pos(self._agent_body_ids[name]) for name in AGENT_NAMES}

    def _visibility_matrix(self) -> NDArray:
        """
        Compute N_AGENTS x N_AGENTS boolean matrix.
        vis[i][j] = True if agent i can see agent j (line-of-sight).
        """
        positions = [self._get_body_pos(self._agent_body_ids[n]) for n in AGENT_NAMES]
        vis = np.zeros((N_AGENTS, N_AGENTS), dtype=bool)

        # Exclude ALL agent geom IDs from ray blocking (spheres + ring cylinders)
        exclude = self._all_agent_geom_ids

        for i in range(N_AGENTS):
            for j in range(N_AGENTS):
                if i == j:
                    vis[i, j] = True
                    continue
                dist = np.linalg.norm(positions[i][:2] - positions[j][:2])
                if dist > self.agent_obs_radius:
                    vis[i, j] = False
                else:
                    vis[i, j] = _can_see(
                        self.model, self.data,
                        positions[i], positions[j],
                        exclude_geom_ids=exclude,
                    )
        return vis

    def _get_obs(self, agent_idx: int, vis_matrix: NDArray) -> NDArray:
        """Build observation vector for a single agent."""
        name = AGENT_NAMES[agent_idx]
        body_id = self._agent_body_ids[name]
        pos = self._get_body_pos(body_id)
        vel = self._get_body_vel(body_id)
        is_hider = 1.0 if agent_idx < N_HIDERS else 0.0
        in_prep = 1.0 if self.current_step < self.prep_steps else 0.0
        prep_remaining = max(0.0, (self.prep_steps - self.current_step) / self.prep_steps)

        obs_parts = [pos, vel, [is_hider, in_prep, prep_remaining]]

        # Other agents
        for j in range(N_AGENTS):
            if j == agent_idx:
                continue
            other_body_id = self._agent_body_ids[AGENT_NAMES[j]]
            other_pos = self._get_body_pos(other_body_id)
            other_vel = self._get_body_vel(other_body_id)
            visible = float(vis_matrix[agent_idx, j])
            other_is_hider = 1.0 if j < N_HIDERS else 0.0

            if visible:
                rel_pos = other_pos - pos
                rel_vel = other_vel - vel
            else:
                rel_pos = np.zeros(3)
                rel_vel = np.zeros(3)

            obs_parts.append(rel_pos)
            obs_parts.append(rel_vel)
            obs_parts.append([visible, other_is_hider])

        # Boxes (iterate over MAX_BOXES, zero-pad slots beyond cur count)
        for bi in range(MAX_BOXES):
            if bi < self._cur_n_boxes:
                bname = self._cur_box_names[bi]
                box_body_id = self._box_body_ids[bname]
                box_pos = self._get_body_pos(box_body_id)
                rel_pos = box_pos - pos
                dist = np.linalg.norm(rel_pos[:2])
                if dist <= self.box_obs_radius:
                    obs_parts.append(rel_pos)
                else:
                    obs_parts.append(np.zeros(3))
                obs_parts.append([float(self.box_grabbed_by[bi] == agent_idx),
                                  1.0])  # exists = True
            else:
                obs_parts.append(np.zeros(3))
                obs_parts.append([0.0, 0.0])  # grabbed=0, exists=0

        # Ramps (iterate over MAX_RAMPS, zero-pad slots beyond cur count)
        for ri in range(MAX_RAMPS):
            if ri < self._cur_n_ramps:
                rname = self._cur_ramp_names[ri]
                ramp_body_id = self._ramp_body_ids[rname]
                ramp_pos = self._get_body_pos(ramp_body_id)
                rel_pos = ramp_pos - pos
                dist = np.linalg.norm(rel_pos[:2])
                if dist <= self.box_obs_radius:
                    obs_parts.append(rel_pos)
                else:
                    obs_parts.append(np.zeros(3))
                obs_parts.append([float(self.ramp_grabbed_by[ri] == agent_idx),
                                  1.0])  # exists = True
            else:
                obs_parts.append(np.zeros(3))
                obs_parts.append([0.0, 0.0])  # grabbed=0, exists=0

        # Lidar (exclude own body + all agent geoms → obstacle-only sensing)
        lidar = _compute_lidar(self.model, self.data, pos, self.lidar_n_rays,
                               self.lidar_max_dist, bodyexclude=body_id,
                               exclude_geom_ids=self._all_agent_geom_ids)
        obs_parts.append(lidar)

        return np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in obs_parts])

    def get_global_state(self) -> NDArray:
        """
        Return the full global state vector (used by MAPPO centralised critic).
        Fixed size using MAX_BOXES / MAX_RAMPS with zero-padding + exists flags.
        """
        parts = []
        for name in AGENT_NAMES:
            body_id = self._agent_body_ids[name]
            parts.append(self._get_body_pos(body_id))
            parts.append(self._get_body_vel(body_id))

        for bi in range(MAX_BOXES):
            if bi < self._cur_n_boxes:
                bname = self._cur_box_names[bi]
                body_id = self._box_body_ids[bname]
                parts.append(self._get_body_pos(body_id))
                parts.append(self._get_body_vel(body_id))
                parts.append([1.0])  # exists=1
            else:
                parts.append(np.zeros(3))  # pos
                parts.append(np.zeros(3))  # vel
                parts.append([0.0])        # exists=0

        for ri in range(MAX_RAMPS):
            if ri < self._cur_n_ramps:
                rname = self._cur_ramp_names[ri]
                body_id = self._ramp_body_ids[rname]
                parts.append(self._get_body_pos(body_id))
                parts.append([float(self.ramp_grabbed_by[ri] >= 0),
                              1.0])  # exists=1
            else:
                parts.append(np.zeros(3))  # pos
                parts.append([0.0, 0.0])   # grabbed=0, exists=0

        in_prep = 1.0 if self.current_step < self.prep_steps else 0.0
        step_frac = self.current_step / self.horizon
        parts.append([in_prep, step_frac])

        return np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in parts])

    # ──────────────────────────────────────────────
    # Reward
    # ──────────────────────────────────────────────

    def _compute_rewards(self, vis_matrix: NDArray) -> dict[str, float]:
        """
        Compute per-agent rewards based on visibility + reward shaping.

        Base reward (joint_zero_sum):
            Hiders: +1 if ALL hiders hidden, -1 otherwise.
            Seekers: +1 if ANY seeker sees a hider, -1 otherwise.

        Reward shaping (additive bonuses, always small):
            1. Prep movement:  Hiders get a tiny bonus for moving during prep.
            2. Grab & move:    Any agent gets a bonus for grabbing an object AND
                               moving it (velocity > threshold).
            3. Hider near cover: During prep, hiders get a bonus if any lidar
                               ray detects a wall within 1.5 m.
            4. Seeker explore:  During play, seekers get a bonus proportional
                               to distance moved.
            5. Individual blend: 20% of the reward is per-agent (gives gradient
                               when one teammate hides but the other doesn't).
        """
        rewards = {name: 0.0 for name in AGENT_NAMES}
        in_prep = self.current_step < self.prep_steps

        # ── Visibility analysis (always needed for shaping too) ──
        seeker_sees_hider = np.zeros(N_SEEKERS, dtype=bool)
        hider_is_seen = np.zeros(N_HIDERS, dtype=bool)

        for si in range(N_SEEKERS):
            seeker_idx = N_HIDERS + si
            for hi in range(N_HIDERS):
                if vis_matrix[seeker_idx, hi]:
                    seeker_sees_hider[si] = True
                    hider_is_seen[hi] = True

        # ── Current agent positions + velocities ──
        agent_pos = {}
        agent_speed = {}
        for name in AGENT_NAMES:
            pos = self._get_body_pos(self._agent_body_ids[name])[:2]
            agent_pos[name] = pos
            if name in self._prev_agent_pos:
                displacement = np.linalg.norm(pos - self._prev_agent_pos[name])
            else:
                displacement = 0.0
            agent_speed[name] = displacement

        # ── Current object positions ──
        cur_box_pos = np.zeros((MAX_BOXES, 2), dtype=np.float32)
        box_moved = np.zeros(MAX_BOXES, dtype=np.float32)
        for bi in range(self._cur_n_boxes):
            bname = self._cur_box_names[bi]
            cur_box_pos[bi] = self._get_body_pos(self._box_body_ids[bname])[:2]
            box_moved[bi] = np.linalg.norm(cur_box_pos[bi] - self._prev_box_pos[bi])

        cur_ramp_pos = np.zeros((MAX_RAMPS, 2), dtype=np.float32)
        ramp_moved = np.zeros(MAX_RAMPS, dtype=np.float32)
        for ri in range(self._cur_n_ramps):
            rname = self._cur_ramp_names[ri]
            cur_ramp_pos[ri] = self._get_body_pos(self._ramp_body_ids[rname])[:2]
            ramp_moved[ri] = np.linalg.norm(cur_ramp_pos[ri] - self._prev_ramp_pos[ri])

        # ──────────────────────────────────────────
        # SHAPING 1: Hider movement during prep
        # ──────────────────────────────────────────
        if in_prep:
            for hi in range(N_HIDERS):
                name = HIDER_NAMES[hi]
                rewards[name] += SHAPE_PREP_MOVEMENT * agent_speed[name]

        # ──────────────────────────────────────────
        # SHAPING 2: Grab + move objects (both teams)
        # ──────────────────────────────────────────
        for agent_idx, name in enumerate(AGENT_NAMES):
            bonus = 0.0
            # Check boxes grabbed by this agent
            for bi in range(self._cur_n_boxes):
                if self.box_grabbed_by[bi] == agent_idx and box_moved[bi] > _GRAB_MOVE_VEL_THRESH:
                    bonus += SHAPE_GRAB_AND_MOVE
            # Check ramps grabbed by this agent
            for ri in range(self._cur_n_ramps):
                if self.ramp_grabbed_by[ri] == agent_idx and ramp_moved[ri] > _GRAB_MOVE_VEL_THRESH:
                    bonus += SHAPE_GRAB_AND_MOVE
            rewards[name] += bonus

        # ──────────────────────────────────────────
        # SHAPING 3: Hider near cover during prep
        # ──────────────────────────────────────────
        if in_prep:
            for hi in range(N_HIDERS):
                name = HIDER_NAMES[hi]
                body_id = self._agent_body_ids[name]
                pos3d = self._get_body_pos(body_id)
                lidar = _compute_lidar(
                    self.model, self.data, pos3d,
                    self.lidar_n_rays, self.lidar_max_dist,
                    bodyexclude=body_id,
                    exclude_geom_ids=self._all_agent_geom_ids,
                )
                # Any ray hitting a wall/box within threshold?
                min_reading = float(np.min(lidar))
                if min_reading < _COVER_LIDAR_THRESHOLD:
                    rewards[name] += SHAPE_HIDER_NEAR_COVER

        # ──────────────────────────────────────────
        # SHAPING 4: Seeker exploration during play
        # ──────────────────────────────────────────
        if not in_prep:
            for si in range(N_SEEKERS):
                name = SEEKER_NAMES[si]
                dist = agent_speed[name]
                self._seeker_distance[si] += dist
                rewards[name] += SHAPE_SEEKER_EXPLORE * dist

        # ──────────────────────────────────────────
        # BASE + INDIVIDUAL BLEND (play phase only)
        # ──────────────────────────────────────────
        if not in_prep:
            all_hidden = not np.any(hider_is_seen)
            any_sees = np.any(seeker_sees_hider)

            # Team-level base reward
            team_hider_rew = 1.0 if all_hidden else -1.0
            team_seeker_rew = 1.0 if any_sees else -1.0

            # Per-agent individual reward (gives gradient)
            for hi in range(N_HIDERS):
                indiv = 1.0 if not hider_is_seen[hi] else -1.0
                blended = ((1.0 - SHAPE_INDIVIDUAL_BLEND) * team_hider_rew +
                           SHAPE_INDIVIDUAL_BLEND * indiv)
                rewards[HIDER_NAMES[hi]] += blended * self.reward_scale

            for si in range(N_SEEKERS):
                indiv = 1.0 if seeker_sees_hider[si] else -1.0
                blended = ((1.0 - SHAPE_INDIVIDUAL_BLEND) * team_seeker_rew +
                           SHAPE_INDIVIDUAL_BLEND * indiv)
                rewards[SEEKER_NAMES[si]] += blended * self.reward_scale

        # ── Update tracking state for next step ──
        for name in AGENT_NAMES:
            self._prev_agent_pos[name] = agent_pos[name].copy()
        self._prev_box_pos[:] = cur_box_pos
        self._prev_ramp_pos[:] = cur_ramp_pos

        return rewards

    # ──────────────────────────────────────────────
    # Actions: grab
    # ──────────────────────────────────────────────

    def _apply_actions(self, actions: dict[str, NDArray]):
        """
        Apply agent actions:
          action[0:2] = movement (x, y force)
          action[2]   = grab (>0 = attempt grab nearest box/ramp)
        """
        for agent_idx, name in enumerate(AGENT_NAMES):
            act = actions.get(name, np.zeros(3, dtype=np.float32))

            # ── Movement ──
            # During prep phase, seekers cannot move
            is_seeker = agent_idx >= N_HIDERS
            if is_seeker and self.current_step < self.prep_steps:
                move = np.zeros(2)
            else:
                move = np.clip(act[:2], -1.0, 1.0) * self.movement_scale

            ax_id, ay_id = self._actuator_ids[name]
            self.data.ctrl[ax_id] = move[0]
            self.data.ctrl[ay_id] = move[1]

            # ── Grab ──
            if act[2] > 0:
                self._try_grab(agent_idx)
            else:
                self._release_grab(agent_idx)

    def _try_grab(self, agent_idx: int):
        """Try to grab the nearest ungrabbed box or ramp within grab_radius."""
        name = AGENT_NAMES[agent_idx]
        agent_pos = self._get_body_pos(self._agent_body_ids[name])[:2]

        best_dist = self.grab_radius
        best_type = None   # 'box' or 'ramp'
        best_idx = -1

        for bi, bname in enumerate(self._cur_box_names):
            if self.box_grabbed_by[bi] >= 0 and self.box_grabbed_by[bi] != agent_idx:
                continue  # already grabbed by another
            box_pos = self._get_body_pos(self._box_body_ids[bname])[:2]
            dist = np.linalg.norm(agent_pos - box_pos)
            if dist < best_dist:
                best_dist = dist
                best_type = "box"
                best_idx = bi

        for ri, rname in enumerate(self._cur_ramp_names):
            if self.ramp_grabbed_by[ri] >= 0 and self.ramp_grabbed_by[ri] != agent_idx:
                continue  # already grabbed by another
            ramp_pos = self._get_body_pos(self._ramp_body_ids[rname])[:2]
            dist = np.linalg.norm(agent_pos - ramp_pos)
            if dist < best_dist:
                best_dist = dist
                best_type = "ramp"
                best_idx = ri

        if best_type == "box" and best_idx >= 0:
            # Release any ramp this agent is holding first
            self.ramp_grabbed_by[self.ramp_grabbed_by == agent_idx] = -1
            self.box_grabbed_by[best_idx] = agent_idx
        elif best_type == "ramp" and best_idx >= 0:
            # Release any box this agent is holding first
            self.box_grabbed_by[self.box_grabbed_by == agent_idx] = -1
            self.ramp_grabbed_by[best_idx] = agent_idx

    def _release_grab(self, agent_idx: int):
        """Release any box or ramp this agent is grabbing."""
        mask_box = self.box_grabbed_by == agent_idx
        self.box_grabbed_by[mask_box] = -1
        mask_ramp = self.ramp_grabbed_by == agent_idx
        self.ramp_grabbed_by[mask_ramp] = -1

    def _apply_grab_forces(self):
        """Apply a spring force pulling grabbed boxes/ramps toward the grabbing agent."""
        for bi, bname in enumerate(self._cur_box_names):
            grabber = self.box_grabbed_by[bi]
            if grabber < 0:
                continue
            agent_name = AGENT_NAMES[grabber]
            agent_pos = self._get_body_pos(self._agent_body_ids[agent_name])
            box_body_id = self._box_body_ids[bname]
            box_pos = self._get_body_pos(box_body_id)

            direction = agent_pos - box_pos
            dist = np.linalg.norm(direction)
            if dist > 0.01:
                # Spring force
                force = 5.0 * direction / dist * min(dist, 1.0)
                self.data.xfrc_applied[box_body_id, :3] = force

        for ri, rname in enumerate(self._cur_ramp_names):
            grabber = self.ramp_grabbed_by[ri]
            if grabber < 0:
                continue
            agent_name = AGENT_NAMES[grabber]
            agent_pos = self._get_body_pos(self._agent_body_ids[agent_name])
            ramp_body_id = self._ramp_body_ids[rname]
            ramp_pos = self._get_body_pos(ramp_body_id)

            direction = agent_pos - ramp_pos
            dist = np.linalg.norm(direction)
            if dist > 0.01:
                # Slightly weaker spring for heavier ramps
                force = 3.5 * direction / dist * min(dist, 1.0)
                self.data.xfrc_applied[ramp_body_id, :3] = force

    # ── Agent velocity damping (caps top speed) ──────────────

    _AGENT_DAMPING = 25.0       # N·s/m — opposing force per unit velocity
    _AGENT_MAX_SPEED = 1.5      # m/s   — approximate terminal speed

    def _apply_velocity_damping(self):
        """Apply linear drag to agent XY velocities to create a natural speed cap."""
        for name in AGENT_NAMES:
            body_id = self._agent_body_ids[name]
            vel_xy = self.data.cvel[body_id, 3:5]  # linear vel x, y
            # Opposing force proportional to velocity
            drag = -self._AGENT_DAMPING * vel_xy
            self.data.xfrc_applied[body_id, 0:2] += drag

    # ──────────────────────────────────────────────
    # ──────────────────────────────────────────────
    # Random spawn zones for domain randomisation
    # ──────────────────────────────────────────────

    # Hider spawn zones: rooms in the southern half (SE, SW, center-south)
    _HIDER_SPAWN_ZONES = [
        (2.0, 5.5, -5.5, -3.5),   # SE room
        (-5.5, -2.0, -5.5, -3.5), # SW room
        (-1.0, 1.0, -5.5, -3.5),  # center-south corridor
        (2.0, 5.5, -2.5, -1.0),   # east side, mid-south
    ]
    # Seeker spawn zones: northern half (NW open area, NE room, center-north)
    _SEEKER_SPAWN_ZONES = [
        (-5.5, 0.5, 2.5, 5.5),    # NW open area
        (-5.5, 0.5, 0.5, 2.5),    # west side, mid-north
        (2.0, 5.5, 2.5, 5.5),     # NE room
    ]
    # Box spawn zones: near door areas ± randomisation
    _BOX_SPAWN_ZONES = [
        (0.5, 4.5, -3.0, 0.0),    # near vertical wall doors
        (-3.0, 1.0, -4.5, -1.5),  # near SW door
        (2.0, 5.5, 0.5, 3.5),     # near NE door
    ]
    # Ramp spawn zones: open areas
    _RAMP_SPAWN_ZONES = [
        (-5.0, 0.0, -2.5, 2.0),   # west open corridor
        (-5.0, -1.0, 2.5, 5.0),   # NW open area
    ]

    def _random_pos_in_zone(self, rng, zone):
        """Sample a random (x, y) within an (xmin, xmax, ymin, ymax) rectangle."""
        xmin, xmax, ymin, ymax = zone
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        return x, y

    def _set_body_xy(self, name, x, y):
        """Set the x, y position of a freejoint body in qpos."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        jnt_adr = self.model.body_jntadr[body_id]
        if jnt_adr >= 0:
            qadr = self.model.jnt_qposadr[jnt_adr]
            self.data.qpos[qadr] = x
            self.data.qpos[qadr + 1] = y

    # ──────────────────────────────────────────────
    # Gym interface
    # ──────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        rng = self.np_random

        if self.procedural:
            # ── Generate a fresh arena layout ──
            xml, meta = generate_arena(
                self._worldgen_rng,
                n_hiders=N_HIDERS,
                n_seekers=N_SEEKERS,
            )
            self._last_meta = meta
            self._cur_n_boxes = meta["n_boxes"]
            self._cur_n_ramps = meta["n_ramps"]

            # Close old viewer before replacing model
            if self._viewer_handle is not None:
                self._viewer_handle.close()
                self._viewer_handle = None

            # Replace model + data
            self.model = mujoco.MjModel.from_xml_string(xml)
            self.data = mujoco.MjData(self.model)

            # Re-cache all IDs for the new model
            self._cache_ids()
        else:
            # Static XML mode — keep same model, randomise positions
            mujoco.mj_resetData(self.model, self.data)

            # ── Randomise agent spawn positions ──
            for name in HIDER_NAMES:
                zone = self._HIDER_SPAWN_ZONES[rng.integers(len(self._HIDER_SPAWN_ZONES))]
                x, y = self._random_pos_in_zone(rng, zone)
                self._set_body_xy(name, x, y)

            for name in SEEKER_NAMES:
                zone = self._SEEKER_SPAWN_ZONES[rng.integers(len(self._SEEKER_SPAWN_ZONES))]
                x, y = self._random_pos_in_zone(rng, zone)
                self._set_body_xy(name, x, y)

            for bi, bname in enumerate(self._cur_box_names):
                zone_idx = bi % len(self._BOX_SPAWN_ZONES)
                zone = self._BOX_SPAWN_ZONES[zone_idx]
                x, y = self._random_pos_in_zone(rng, zone)
                self._set_body_xy(bname, x, y)

            for ri, rname in enumerate(self._cur_ramp_names):
                zone = self._RAMP_SPAWN_ZONES[rng.integers(len(self._RAMP_SPAWN_ZONES))]
                x, y = self._random_pos_in_zone(rng, zone)
                self._set_body_xy(rname, x, y)

        # Reset object states
        self.box_grabbed_by[:] = -1
        self.ramp_grabbed_by[:] = -1
        self.current_step = 0

        # ── Reward-shaping tracking state ──
        # Previous agent positions (for movement / distance tracking)
        mujoco.mj_forward(self.model, self.data)
        self._prev_agent_pos = {}
        for name in AGENT_NAMES:
            self._prev_agent_pos[name] = self._get_body_pos(
                self._agent_body_ids[name])[:2].copy()
        # Cumulative distance moved per seeker during play phase
        self._seeker_distance = np.zeros(N_SEEKERS, dtype=np.float32)
        # Previous box/ramp positions (for grab-and-move detection)
        self._prev_box_pos = np.zeros((MAX_BOXES, 2), dtype=np.float32)
        for bi in range(self._cur_n_boxes):
            bname = self._cur_box_names[bi]
            self._prev_box_pos[bi] = self._get_body_pos(
                self._box_body_ids[bname])[:2]
        self._prev_ramp_pos = np.zeros((MAX_RAMPS, 2), dtype=np.float32)
        for ri in range(self._cur_n_ramps):
            rname = self._cur_ramp_names[ri]
            self._prev_ramp_pos[ri] = self._get_body_pos(
                self._ramp_body_ids[rname])[:2]

        # Forward already done above for position tracking

        vis_matrix = self._visibility_matrix()
        obs = {name: self._get_obs(i, vis_matrix) for i, name in enumerate(AGENT_NAMES)}

        info = {
            "global_state": self.get_global_state(),
            "prep_phase": True,
            "step": 0,
        }
        if self.procedural:
            info["layout_type"] = self._last_meta["layout_type"]
            info["mood"] = self._last_meta.get("mood", "")
            info["n_boxes"] = self._cur_n_boxes
            info["n_ramps"] = self._cur_n_ramps
        return obs, info

    def step(self, actions: dict[str, NDArray]):
        # Apply actions (sets ctrl for actuators, handles grab)
        self._apply_actions(actions)

        # Step physics with per-substep force updates
        for _ in range(self.n_substeps):
            # Recompute external forces each substep (grab + drag)
            self.data.xfrc_applied[:] = 0.0
            self._apply_grab_forces()
            self._apply_velocity_damping()
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1

        # Compute visibility & observations
        vis_matrix = self._visibility_matrix()
        self._last_vis_matrix = vis_matrix  # cache for colour indicator
        obs = {name: self._get_obs(i, vis_matrix) for i, name in enumerate(AGENT_NAMES)}
        rewards = self._compute_rewards(vis_matrix)

        terminated = {name: False for name in AGENT_NAMES}
        truncated = {name: self.current_step >= self.horizon for name in AGENT_NAMES}

        info = {
            "global_state": self.get_global_state(),
            "prep_phase": self.current_step < self.prep_steps,
            "step": self.current_step,
            "visibility_matrix": vis_matrix,
        }

        # Update visual indicators every step (cheap numpy writes)
        self._update_agent_colors()

        if self.render_mode == "human":
            self.render()

        return obs, rewards, terminated, truncated, info

    # ──────────────────────────────────────────────
    # Visual state indicators (grab + visibility)
    # ──────────────────────────────────────────────

    def _agent_is_grabbing(self, agent_idx: int) -> bool:
        """Return True if the given agent is currently grabbing any object."""
        if np.any(self.box_grabbed_by == agent_idx):
            return True
        if np.any(self.ramp_grabbed_by == agent_idx):
            return True
        return False

    def _update_agent_colors(self):
        """
        Update geom colours to reflect the current game state:

        Priority (highest → lowest):
          1. Seeker frozen during prep phase → DIM GREY (seekers only)
          2. Hider seen by a seeker (play phase) → MAGENTA
          3. Grabbing → ORANGE
          4. Default → original team colour

        Grabbed objects also turn orange; released objects restore.
        """
        rgba = self.model.geom_rgba
        in_play = self.current_step >= self.prep_steps

        # ── Compute which hiders are currently seen ──
        hider_is_seen = np.zeros(N_HIDERS, dtype=bool)
        if in_play and hasattr(self, '_last_vis_matrix'):
            vm = self._last_vis_matrix
            for si in range(N_SEEKERS):
                seeker_idx = N_HIDERS + si
                for hi in range(N_HIDERS):
                    if vm[seeker_idx, hi]:
                        hider_is_seen[hi] = True

        # ── Agents ──
        for agent_idx, name in enumerate(AGENT_NAMES):
            gid = self._agent_geom_ids[name]
            is_seeker = agent_idx >= N_HIDERS

            if is_seeker and not in_play:
                rgba[gid] = _PREP_RGBA                # highest: frozen seeker (dim)
            elif agent_idx < N_HIDERS and hider_is_seen[agent_idx]:
                rgba[gid] = _SEEN_RGBA                # seen hider: magenta
            elif self._agent_is_grabbing(agent_idx):
                rgba[gid] = _GRAB_RGBA                # grabbing: orange
            else:
                rgba[gid] = self._original_rgba[name]  # default team colour

        # ── Boxes: orange while grabbed, original otherwise ──
        for bi, bname in enumerate(self._cur_box_names):
            gid = self._box_geom_ids[bname]
            if self.box_grabbed_by[bi] >= 0:
                rgba[gid] = _GRAB_RGBA
            else:
                rgba[gid] = self._original_rgba[bname]

        # ── Ramps: orange while grabbed, original otherwise ──
        for ri, rname in enumerate(self._cur_ramp_names):
            gid = self._ramp_geom_ids[rname]
            if self.ramp_grabbed_by[ri] >= 0:
                rgba[gid] = _GRAB_RGBA
            else:
                rgba[gid] = self._original_rgba[rname]

    # ──────────────────────────────────────────────
    # Rendering
    # ──────────────────────────────────────────────

    def _init_renderer(self):
        """Initialise MuJoCo viewer."""
        self._viewer_handle = None  # will be launched on first render
        try:
            import mujoco.viewer  # noqa: F401 – just check availability
        except ImportError:
            print("Warning: mujoco viewer not available. Install mujoco for rendering.")

    def render(self):
        import mujoco.viewer  # ensure sub-module is loaded; also satisfies Python scoping

        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=720, width=1280)
            # Use the zoomed-out "overview" camera defined in the XML
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
            renderer.update_scene(self.data, camera=cam_id)
            pixels = renderer.render()
            renderer.close()
            return pixels
        elif self.render_mode == "human":
            if self._viewer_handle is None:
                self._viewer_handle = mujoco.viewer.launch_passive(
                    self.model, self.data,
                    show_left_ui=False, show_right_ui=False,
                )
                # Zoomed-out overview showing the full arena
                self._viewer_handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self._viewer_handle.cam.trackbodyid = 0  # world
                self._viewer_handle.cam.distance = 16.8
                self._viewer_handle.cam.elevation = -50.0
                self._viewer_handle.cam.azimuth = 90.0
                self._viewer_handle.cam.lookat[:] = [0.0, 0.0, 0.0]
                # Show site labels (only agents have sites → only agents get labelled)
                self._viewer_handle.opt.label = mujoco.mjtLabel.mjLABEL_SITE
            self._viewer_handle.sync()

    def close(self):
        if self._viewer_handle is not None:
            self._viewer_handle.close()
            self._viewer_handle = None
        super().close()

    # ──────────────────────────────────────────────
    # Utility properties
    # ──────────────────────────────────────────────

    @property
    def global_state_size(self) -> int:
        return self._global_state_size

    @property
    def n_hiders(self) -> int:
        return N_HIDERS

    @property
    def n_seekers(self) -> int:
        return N_SEEKERS
