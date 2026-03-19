"""
Procedural World Generator for Hide and Seek
=============================================

Generates randomised MuJoCo XML arenas for each episode, inspired by OpenAI's
mujoco-worldgen but built for MuJoCo 3.x (modern Python bindings).

Each call to `generate_arena()` produces a complete XML string with:
  - Random room layout (1-3 interior rooms with varying sizes)
  - Random door positions and widths
  - Random number and placement of boxes (2-5)
  - Random number and placement of ramps (0-2)
  - Random agent spawn positions (hiders south, seekers north)
  - Randomised lighting / colour mood (night, sunset, noon, winter)
  - Decorative terrain blocks outside the arena walls (OpenAI style)
  - Matte checker-pattern floor

The XML is loaded directly via `mujoco.MjModel.from_xml_string()`.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator


# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

ARENA_HALF = 6.0       # half-size -> 12m x 12m arena
WALL_H_OUTER = 0.75     # outer wall half-height -> 1.5 m total (unscalable)
WALL_H_INNER = 0.400   # inner wall half-height -> 0.80 m total (ramp-scalable)
WALL_THICK = 0.12      # wall thickness (half-size)
AGENT_RADIUS = 0.35
AGENT_Z = 0.35

# Ramp wedge geometry (triangular prism)
RAMP_LENGTH = 2.4       # length of the slope along the ground (metres)
RAMP_WIDTH  = 0.8       # full width of the ramp (metres)
RAMP_HEIGHT = 0.85      # height of the tall back edge (metres) -- above inner walls

# Outer terrain (decorative blocks surrounding the arena)
_TERRAIN_EXTENT = 14.0   # how far out the terrain grid extends from centre
_TERRAIN_STEP   = 1.6    # grid spacing for terrain blocks
_TERRAIN_H_MIN  = 0.05   # minimum block half-height
_TERRAIN_H_MAX  = 0.8    # maximum block half-height
_TERRAIN_HX     = 0.7    # block half-size X
_TERRAIN_HY     = 0.7    # block half-size Y


# ─────────────────────────────────────────────────────────
# Lighting / colour mood presets
# ─────────────────────────────────────────────────────────

# Each mood defines: floor tile colour, grid-line colour, wall tint, skybox
# gradient, headlight settings, directional lights, and terrain block colour.

_MOODS = {
    "night": {
        "floor_rgb":     "0.16 0.18 0.23",
        "grid_rgba":     "0.08 0.09 0.12 1",
        "wall_rgba":     "0.22 0.28 0.36 1",
        "box_rgba":      "0.40 0.42 0.48 1",
        "ramp_rgba":     "0.35 0.55 0.40 1",
        "terrain_rgba":  "0.14 0.17 0.22 1",
        "skybox_rgb1":   "0.06 0.06 0.14",
        "skybox_rgb2":   "0.02 0.02 0.06",
        "headlight":     'ambient="0.15 0.16 0.22" diffuse="0.12 0.13 0.20" specular="0.03 0.03 0.05"',
        "key_light":     'pos="4 -4 14" dir="-0.3 0.3 -1" diffuse="0.30 0.35 0.55" specular="0.10 0.12 0.18" castshadow="true"',
        "fill_light":    'pos="-5 5 10" dir="0.3 -0.3 -1" diffuse="0.10 0.12 0.20" specular="0.02 0.02 0.04"',
        "rim_light":     'pos="0 0 16" dir="0 0 -1" diffuse="0.08 0.10 0.16" specular="0.04 0.04 0.06"',
    },
    "sunset": {
        "floor_rgb":     "0.30 0.26 0.22",
        "grid_rgba":     "0.15 0.12 0.10 1",
        "wall_rgba":     "0.40 0.34 0.30 1",
        "box_rgba":      "0.58 0.50 0.44 1",
        "ramp_rgba":     "0.52 0.68 0.42 1",
        "terrain_rgba":  "0.30 0.24 0.18 1",
        "skybox_rgb1":   "0.45 0.25 0.12",
        "skybox_rgb2":   "0.18 0.10 0.06",
        "headlight":     'ambient="0.30 0.24 0.18" diffuse="0.25 0.20 0.14" specular="0.06 0.05 0.03"',
        "key_light":     'pos="10 -3 6" dir="-0.7 0.2 -0.5" diffuse="1.2 0.70 0.35" specular="0.50 0.35 0.18" castshadow="true"',
        "fill_light":    'pos="-5 5 10" dir="0.3 -0.3 -1" diffuse="0.18 0.15 0.22" specular="0.04 0.03 0.05"',
        "rim_light":     'pos="0 0 16" dir="0 0 -1" diffuse="0.12 0.10 0.08" specular="0.06 0.05 0.04"',
    },
    "noon": {
        "floor_rgb":     "0.32 0.31 0.28",
        "grid_rgba":     "0.16 0.15 0.13 1",
        "wall_rgba":     "0.42 0.40 0.38 1",
        "box_rgba":      "0.60 0.58 0.55 1",
        "ramp_rgba":     "0.55 0.75 0.52 1",
        "terrain_rgba":  "0.30 0.28 0.24 1",
        "skybox_rgb1":   "0.40 0.55 0.75",
        "skybox_rgb2":   "0.15 0.22 0.35",
        "headlight":     'ambient="0.40 0.38 0.34" diffuse="0.35 0.34 0.30" specular="0.10 0.10 0.08"',
        "key_light":     'pos="1 -1 18" dir="-0.05 0.05 -1" diffuse="1.4 1.30 1.05" specular="0.55 0.52 0.45" castshadow="true"',
        "fill_light":    'pos="-5 5 10" dir="0.3 -0.3 -1" diffuse="0.30 0.30 0.35" specular="0.06 0.06 0.07"',
        "rim_light":     'pos="0 0 16" dir="0 0 -1" diffuse="0.20 0.20 0.22" specular="0.10 0.10 0.10"',
    },
    "winter": {
        "floor_rgb":     "0.48 0.50 0.53",
        "grid_rgba":     "0.25 0.26 0.28 1",
        "wall_rgba":     "0.40 0.44 0.50 1",
        "box_rgba":      "0.48 0.50 0.54 1",
        "ramp_rgba":     "0.42 0.58 0.46 1",
        "terrain_rgba":  "0.44 0.46 0.52 1",
        "skybox_rgb1":   "0.55 0.58 0.65",
        "skybox_rgb2":   "0.30 0.34 0.42",
        "headlight":     'ambient="0.38 0.40 0.45" diffuse="0.34 0.36 0.40" specular="0.05 0.05 0.06"',
        "key_light":     'pos="5 -5 14" dir="-0.3 0.3 -1" diffuse="0.75 0.78 0.85" specular="0.15 0.15 0.18" castshadow="true"',
        "fill_light":    'pos="-5 5 10" dir="0.3 -0.3 -1" diffuse="0.25 0.27 0.32" specular="0.04 0.04 0.05"',
        "rim_light":     'pos="0 0 16" dir="0 0 -1" diffuse="0.18 0.20 0.24" specular="0.06 0.06 0.08"',
    },
}

# Default hider / seeker colours (unchanged by mood -- team identity matters)
_HIDER_RGBA  = "0.310 0.765 0.969 1"
_SEEKER_RGBA = "0.937 0.325 0.314 1"


# ─────────────────────────────────────────────────────────
# XML template pieces
# ─────────────────────────────────────────────────────────

_XML_HEADER = """\
<mujoco model="hide_and_seek_procgen">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.02" gravity="0 0 -9.81" integrator="implicitfast"
          solver="Newton" iterations="30" tolerance="1e-8" impratio="10">
    <flag warmstart="enable"/>
  </option>
  <default>
    <geom condim="6" friction="0.4 0.3 0.1" margin="0.001"/>
    <joint damping="0.5" armature="0.01"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0"/>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient"
             rgb1="{skybox_rgb1}" rgb2="{skybox_rgb2}"
             width="512" height="512"/>
    <texture name="groundplane" type="2d" builtin="flat"
             rgb1="{floor_rgb}" rgb2="{floor_rgb}"
             width="64" height="64"/>
    <material name="groundplane" texture="groundplane"
              texuniform="true" reflectance="0" shininess="0"
              specular="0" emission="0"/>
{mesh_assets}
  </asset>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
    <headlight {headlight}/>
  </visual>
  <worldbody>
    <geom name="floor" type="plane" size="200 200 0.1" material="groundplane"
          conaffinity="1" condim="6"/>
    <light name="key_light" {key_light} cutoff="60"/>
    <light name="fill_light" {fill_light} cutoff="80"/>
    <light name="rim_light" {rim_light} cutoff="100"/>
    <camera name="overview" pos="0 -12 20" xyaxes="1 0 0 0 0.55 0.84" fovy="55"/>
"""

_XML_FOOTER = """\
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


# ─────────────────────────────────────────────────────────
# Helper: wall segment XML
# ─────────────────────────────────────────────────────────

_wall_counter = 0

def _wall_xml(cx: float, cy: float, hx: float, hy: float,
              half_height: float | None = None,
              wall_rgba: str = "0.42 0.40 0.38 1") -> str:
    """Generate XML for a wall box centered at (cx, cy) with half-sizes (hx, hy).
    Uses WALL_H_INNER by default (interior walls)."""
    if half_height is None:
        half_height = WALL_H_INNER
    global _wall_counter
    name = f"wall_{_wall_counter}"
    _wall_counter += 1
    return (f'    <body name="{name}" pos="{cx:.3f} {cy:.3f} {half_height}">\n'
            f'      <geom type="box" size="{hx:.3f} {hy:.3f} {half_height}" '
            f'rgba="{wall_rgba}" conaffinity="1" condim="3"/>\n'
            f'    </body>\n')


# ─────────────────────────────────────────────────────────
# Wall generation with doors
# ─────────────────────────────────────────────────────────

def _segments_with_doors(start: float, end: float, door_positions: list[float],
                         door_width: float) -> list[tuple[float, float]]:
    """
    Given a wall running from `start` to `end`, cut doors at the given positions.
    Returns list of (seg_start, seg_end) solid segments.
    """
    cuts = []
    for dp in sorted(door_positions):
        d_lo = dp - door_width / 2
        d_hi = dp + door_width / 2
        cuts.append((d_lo, d_hi))

    # Merge overlapping cuts
    merged = []
    for lo, hi in sorted(cuts):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    segments = []
    prev = start
    for lo, hi in merged:
        lo = max(lo, start)
        hi = min(hi, end)
        if lo > prev + 0.1:  # min segment length
            segments.append((prev, lo))
        prev = hi
    if end > prev + 0.1:
        segments.append((prev, end))

    return segments


def _vertical_wall_xml(x: float, y_start: float, y_end: float,
                       door_positions: list[float], door_width: float,
                       wall_rgba: str = "0.42 0.40 0.38 1") -> str:
    """Generate XML for a vertical wall at x with doors."""
    segments = _segments_with_doors(y_start, y_end, door_positions, door_width)
    xml = ""
    for s, e in segments:
        cy = (s + e) / 2
        hy = (e - s) / 2
        xml += _wall_xml(x, cy, WALL_THICK, hy, wall_rgba=wall_rgba)
    return xml


def _horizontal_wall_xml(y: float, x_start: float, x_end: float,
                         door_positions: list[float], door_width: float,
                         wall_rgba: str = "0.42 0.40 0.38 1") -> str:
    """Generate XML for a horizontal wall at y with doors."""
    segments = _segments_with_doors(x_start, x_end, door_positions, door_width)
    xml = ""
    for s, e in segments:
        cx = (s + e) / 2
        hx = (e - s) / 2
        xml += _wall_xml(cx, y, hx, WALL_THICK, wall_rgba=wall_rgba)
    return xml


# ─────────────────────────────────────────────────────────
# Agent XML
# ─────────────────────────────────────────────────────────

def _agent_xml(name: str, label: str, x: float, y: float,
               rgba: str, ring_rgba: str) -> str:
    return (
        f'    <body name="{name}" pos="{x:.3f} {y:.3f} {AGENT_Z}">\n'
        f'      <freejoint name="{name}_joint"/>\n'
        f'      <geom name="{name}_geom" type="sphere" size="{AGENT_RADIUS}" density="11"\n'
        f'            rgba="{rgba}" conaffinity="1" condim="6" friction="0.3 0.1 0.05"/>\n'
        f'      <site name="{label}" pos="0 0 0.55" size="0.01" rgba="{ring_rgba} 0.01"/>\n'
        f'      <geom type="cylinder" size="0.42 0.02" pos="0 0 0" mass="0" '
        f'rgba="{ring_rgba} 0.10" contype="0" conaffinity="0"/>\n'
        f'    </body>\n'
    )


# ─────────────────────────────────────────────────────────
# Box XML
# ─────────────────────────────────────────────────────────

def _box_xml(idx: int, x: float, y: float,
             hx: float = 0.5, hy: float = 0.5, hz: float = 0.3,
             mass: float = 2.0, box_rgba: str = "0.60 0.58 0.55 1") -> str:
    return (
        f'    <body name="box_{idx}" pos="{x:.3f} {y:.3f} {hz:.3f}">\n'
        f'      <freejoint name="box_{idx}_joint"/>\n'
        f'      <geom name="box_{idx}_geom" type="box" size="{hx:.2f} {hy:.2f} {hz:.2f}" '
        f'mass="{mass:.1f}"\n'
        f'            rgba="{box_rgba}" conaffinity="1" condim="6" '
        f'friction="0.2 0.005 0.0001"/>\n'
        f'    </body>\n'
    )


# ─────────────────────────────────────────────────────────
# Ramp XML
# ─────────────────────────────────────────────────────────

def _ramp_xml(idx: int, x: float, y: float, yaw: float = 0.0,
              ramp_rgba: str = "0.55 0.75 0.52 1") -> str:
    """
    A proper wedge-shaped ramp built from an inline MuJoCo mesh (convex hull).

    The wedge is a triangular prism: flat bottom, vertical back face, sloped
    top surface.  MuJoCo automatically builds the convex hull from the 6
    vertices we supply, producing a solid wedge that is visually obvious and
    physically climbable.

    Dimensions (before yaw rotation):
      - Length  (X):  RAMP_LENGTH   (2.4 m — the slope direction)
      - Width   (Y):  RAMP_WIDTH    (0.8 m)
      - Height  (Z):  RAMP_HEIGHT   (0.85 m — the tall back edge)

    The low (approach) side faces −X; the tall back edge faces +X.
    The mesh is centred at (0, 0, 0) in the body frame so that the freejoint
    position corresponds to the ramp's ground-level centre.
    """
    L  = RAMP_LENGTH   # 2.4
    W  = RAMP_WIDTH    # 0.8
    H  = RAMP_HEIGHT   # 0.85
    hw = W / 2         # half-width

    # 6 vertices of the triangular prism, centred at origin
    # Bottom face (z=0) — full rectangle
    # Top face — only the back edge is at H, the front edge stays at z=0
    # This naturally forms a wedge / ramp shape.
    #
    #  Side view (XZ plane):
    #
    #     back (+X)
    #       │H ╱  slope
    #       │ ╱
    #  ─────┘───────  ground (z=0)
    #     +X   −X  (approach)
    #
    verts = [
        # bottom-front-left, bottom-front-right
        f"{-L/2:.4f} {-hw:.4f} {-H/3:.4f}",
        f"{-L/2:.4f} { hw:.4f} {-H/3:.4f}",
        # bottom-back-left,  bottom-back-right
        f"{ L/2:.4f} {-hw:.4f} {-H/3:.4f}",
        f"{ L/2:.4f} { hw:.4f} {-H/3:.4f}",
        # top-back-left,     top-back-right   (the tall edge)
        f"{ L/2:.4f} {-hw:.4f} { 2*H/3:.4f}",
        f"{ L/2:.4f} { hw:.4f} { 2*H/3:.4f}",
    ]
    vert_str = "  ".join(verts)

    body_z = H / 3 + 0.005  # lift so bottom sits just above ground

    return (
        # Mesh asset (unique per ramp index — placed inside <asset> would be
        # cleaner, but MuJoCo also accepts <asset> items inside <worldbody>
        # siblings if we place them before use.  Instead we define the mesh
        # inline just before the body; the generator concatenates all XML parts
        # and the mesh assets block is added to the header.)
        #
        # Joint setup: slide X + slide Y + hinge Z.
        # This keeps the ramp flat on the ground (no tipping/flipping) while
        # still allowing it to be pushed around and rotated by agents.
        f'    <body name="ramp_{idx}" pos="{x:.3f} {y:.3f} {body_z:.4f}" euler="0 0 {yaw:.4f}">\n'
        f'      <joint name="ramp_{idx}_jx" type="slide" axis="1 0 0" damping="2.0"/>\n'
        f'      <joint name="ramp_{idx}_jy" type="slide" axis="0 1 0" damping="2.0"/>\n'
        f'      <joint name="ramp_{idx}_jz" type="hinge" axis="0 0 1" damping="1.0"/>\n'
        f'      <geom name="ramp_{idx}_geom" type="mesh" mesh="ramp_wedge_{idx}"\n'
        f'            mass="5.0" rgba="{ramp_rgba}"\n'
        f'            conaffinity="1" condim="6" friction="0.8 0.01 0.001"/>\n'
        f'    </body>\n'
    ), vert_str   # return body XML *and* vertex data for the asset block


# ─────────────────────────────────────────────────────────
# Layout configurations
# ─────────────────────────────────────────────────────────

# Minimum gap between an interior wall endpoint and another wall / arena edge.
# This prevents pockets that trap agents.
_WALL_GAP = 1.2

# Perimeter escape gap: interior walls near outer walls get shortened
# to create a corridor along the arena perimeter for fleeing.
_PERIMETER_GAP = 1.5     # metres — gap between interior wall end and outer wall
_PERIMETER_GAP_PROB = 0.7  # probability of adding a gap at each wall end


def _gen_layout(rng: Generator,
                allowed_layouts: list[str] | None = None) -> dict:
    """
    Generate a random room layout.

    Layout families:
      A) divider  — one horizontal wall splitting the arena, optional vertical stub walls
      B) cross    — vertical + horizontal wall that do NOT touch each other
      C) L_shape  — two walls forming an L, but with a guaranteed gap at the corner
      D) rooms    — two perpendicular walls creating 3-4 enclosed areas

    IMPORTANT: interior walls never meet each other.  Every wall either
    - ends at an outer arena edge, OR
    - stops short of the other wall by at least _WALL_GAP metres.
    This guarantees agents always have a path around every wall.

    Args:
        allowed_layouts: subset of layout types to sample from (curriculum control).
                         If None, all types are equally likely.
    """
    all_types = ["divider", "cross", "L_shape", "rooms"]
    if allowed_layouts is not None:
        choices = [t for t in all_types if t in allowed_layouts]
        if not choices:
            choices = all_types  # fallback
    else:
        choices = all_types
    layout_type = rng.choice(choices)

    walls = []  # list of (orientation, pos, start, end, [door_positions], door_width)
    A = ARENA_HALF  # 6.0

    if layout_type == "divider":
        # ── Horizontal wall from west edge to east edge ──
        hy = float(rng.uniform(-1.5, 1.5))
        n_doors = int(rng.integers(1, 3))  # 1-2 doors
        door_width = float(rng.uniform(1.0, 2.0))
        door_xs = rng.uniform(-A + 1.5, A - 1.5, size=n_doors).tolist()
        walls.append(("h", hy, -A, A, door_xs, door_width))

        # Optional vertical STUB on the left side.
        # Runs from south arena edge but STOPS SHORT of the horizontal wall.
        if rng.random() < 0.6:
            vx = float(rng.uniform(-A + 1.5, -0.5))
            stub_end = hy - _WALL_GAP      # guaranteed gap before horizontal wall
            if stub_end > -A + 2.0:         # only if stub is long enough to matter
                n_d = int(rng.integers(1, 2))
                dw = float(rng.uniform(1.0, 1.8))
                door_lo = -A + 1.0
                door_hi = max(door_lo + 0.1, stub_end - 0.5)
                dys = rng.uniform(door_lo, door_hi, size=n_d).tolist()
                walls.append(("v", vx, -A, stub_end, dys, dw))

        # Optional vertical STUB on the right side, above the horizontal wall.
        # Runs from horizontal wall + gap to north arena edge.
        if rng.random() < 0.5:
            vx2 = float(rng.uniform(0.5, A - 1.5))
            stub_start = hy + _WALL_GAP    # guaranteed gap after horizontal wall
            if stub_start < A - 2.0:
                n_d2 = int(rng.integers(1, 2))
                dw2 = float(rng.uniform(1.0, 1.8))
                door_lo2 = stub_start + 0.5
                door_hi2 = max(door_lo2 + 0.1, A - 1.0)
                dys2 = rng.uniform(door_lo2, door_hi2, size=n_d2).tolist()
                walls.append(("v", vx2, stub_start, A, dys2, dw2))

        # Hiders spawn below the wall, seekers above
        hider_zones = [(-A + 0.5, A - 0.5, -A + 0.5, hy - 1.0),
                        (-A + 0.5, -0.5, -A + 0.5, hy - 0.5)]
        seeker_zones = [(-A + 0.5, A - 0.5, hy + 1.0, A - 0.5),
                        (0.5, A - 0.5, hy + 0.5, A - 0.5)]

    elif layout_type == "cross":
        # ── Two walls that NEVER touch ──
        # Vertical wall runs full height (south edge → north edge).
        vx = float(rng.uniform(-0.5, 2.0))
        dw_v = float(rng.uniform(1.0, 2.0))
        door_ys = rng.uniform(-A + 2.0, A - 2.0, size=int(rng.integers(1, 2))).tolist()
        walls.append(("v", vx, -A, A, door_ys, dw_v))

        # Horizontal wall is split into TWO stubs that each stop _WALL_GAP
        # before the vertical wall, leaving a clear corridor across the middle.
        hy = float(rng.uniform(-1.5, 1.5))
        dw_h = float(rng.uniform(1.0, 2.0))

        # Left stub: west edge → (vx - _WALL_GAP)
        left_end = vx - _WALL_GAP
        if left_end > -A + 2.0:
            door_lo = -A + 1.0
            door_hi = max(door_lo + 0.1, left_end - 0.5)
            door_xs_L = rng.uniform(door_lo, door_hi, size=int(rng.integers(1, 2))).tolist()
            walls.append(("h", hy, -A, left_end, door_xs_L, dw_h))

        # Right stub: (vx + _WALL_GAP) → east edge
        right_start = vx + _WALL_GAP
        if right_start < A - 2.0:
            door_lo_R = right_start + 0.5
            door_hi_R = max(door_lo_R + 0.1, A - 1.0)
            door_xs_R = rng.uniform(door_lo_R, door_hi_R, size=int(rng.integers(1, 2))).tolist()
            walls.append(("h", hy, right_start, A, door_xs_R, dw_h))

        hider_zones = [(vx + 1.0, A - 0.5, -A + 0.5, hy - 0.5),
                        (-A + 0.5, vx - 0.5, -A + 0.5, hy - 0.5)]
        seeker_zones = [(-A + 0.5, vx - 0.5, hy + 0.5, A - 0.5),
                        (vx + 1.0, A - 0.5, hy + 0.5, A - 0.5)]

    elif layout_type == "L_shape":
        # ── Two walls forming an L, but with a GAP at the corner ──
        vx = float(rng.uniform(0.5, 3.0))

        # Vertical wall runs from some midpoint up to north edge (not full height).
        vy_start = float(rng.uniform(-2.0, 0.0))  # starts in mid-arena
        dw_v = float(rng.uniform(1.0, 1.8))
        door_ys = [float(rng.uniform(vy_start + 1.5, A - 1.5))]
        walls.append(("v", vx, vy_start, A, door_ys, dw_v))

        # Horizontal wall runs from west edge but STOPS SHORT of vx by _WALL_GAP.
        hy = float(rng.uniform(vy_start - 1.5, vy_start + 0.5))
        h_end = vx - _WALL_GAP             # guaranteed gap before vertical wall
        if h_end > -A + 2.0:
            dw_h = float(rng.uniform(1.0, 1.8))
            door_lo = -A + 1.0
            door_hi = max(door_lo + 0.1, h_end - 0.5)
            door_xs = [float(rng.uniform(door_lo, door_hi))]
            walls.append(("h", hy, -A, h_end, door_xs, dw_h))

        hider_zones = [(vx + 1.0, A - 0.5, -A + 0.5, hy - 0.5),
                        (-A + 0.5, vx - 0.5, -A + 0.5, hy - 1.0)]
        seeker_zones = [(-A + 0.5, vx - 0.5, hy + 1.0, A - 0.5)]

    else:  # rooms — two perpendicular walls creating 3-4 enclosed areas
        # ── Vertical wall from south to north edge ──
        vx = float(rng.uniform(-1.0, 2.0))
        dw_v = float(rng.uniform(1.0, 1.6))
        door_ys_v = rng.uniform(-A + 2.0, A - 2.0, size=int(rng.integers(1, 2))).tolist()
        walls.append(("v", vx, -A, A, door_ys_v, dw_v))

        # ── Horizontal wall from west to east edge (crosses vertical wall) ──
        hy = float(rng.uniform(-1.5, 1.5))
        dw_h = float(rng.uniform(1.0, 1.6))

        # Left stub: west edge → (vx - _WALL_GAP)
        left_end = vx - _WALL_GAP
        if left_end > -A + 1.5:
            door_lo = -A + 1.0
            door_hi = max(door_lo + 0.1, left_end - 0.5)
            door_xs_L = [float(rng.uniform(door_lo, door_hi))]
            walls.append(("h", hy, -A, left_end, door_xs_L, dw_h))

        # Right stub: (vx + _WALL_GAP) → east edge
        right_start = vx + _WALL_GAP
        if right_start < A - 1.5:
            door_lo_R = right_start + 0.5
            door_hi_R = max(door_lo_R + 0.1, A - 1.0)
            door_xs_R = [float(rng.uniform(door_lo_R, door_hi_R))]
            walls.append(("h", hy, right_start, A, door_xs_R, dw_h))

        # ── Optional extra stub for a 5th pocket ──
        if rng.random() < 0.5:
            extra_vx = float(rng.uniform(vx + 2.5, A - 1.0))
            if extra_vx < A - 1.0:
                extra_vy_start = hy + _WALL_GAP
                if extra_vy_start < A - 2.0:
                    dw_e = float(rng.uniform(1.0, 1.5))
                    door_ys_e = [float(rng.uniform(extra_vy_start + 1.0, A - 1.0))]
                    walls.append(("v", extra_vx, extra_vy_start, A, door_ys_e, dw_e))

        hider_zones = [(vx + 1.0, A - 0.5, -A + 0.5, hy - 0.5),
                        (-A + 0.5, vx - 0.5, -A + 0.5, hy - 0.5)]
        seeker_zones = [(-A + 0.5, vx - 0.5, hy + 0.5, A - 0.5),
                        (vx + 1.0, A - 0.5, hy + 0.5, A - 0.5)]

    # ── Apply perimeter escape gaps ──
    # For each wall that touches the outer arena boundary, randomly shorten it
    # to create a passable corridor along the perimeter.  This gives hiders
    # escape routes when spotted inside a room.
    trimmed_walls = []
    for orientation, pos, start, end, door_positions, door_width in walls:
        new_start, new_end = start, end
        # Check if start touches outer wall (-A)
        if abs(start - (-A)) < 0.3 and rng.random() < _PERIMETER_GAP_PROB:
            new_start = -A + _PERIMETER_GAP
        # Check if end touches outer wall (A)
        if abs(end - A) < 0.3 and rng.random() < _PERIMETER_GAP_PROB:
            new_end = A - _PERIMETER_GAP
        # Ensure wall is still long enough to matter (min ~2m)
        if new_end - new_start >= 2.0:
            # Filter door positions to only those within new wall span
            valid_doors = [d for d in door_positions
                           if new_start + 0.5 < d < new_end - 0.5]
            # Ensure at least one door if none remain
            if not valid_doors and (new_end - new_start) > door_width + 1.0:
                valid_doors = [float(rng.uniform(new_start + 0.8, new_end - 0.8))]
            trimmed_walls.append((orientation, pos, new_start, new_end,
                                  valid_doors, door_width))
        # else: wall too short after trimming — drop it entirely (more open layout)
    walls = trimmed_walls

    # ── Randomly swap team spawn sides (50%) ──
    # Prevents hiders from learning a directional bias (e.g. "south = safe").
    if rng.random() < 0.5:
        hider_zones, seeker_zones = seeker_zones, hider_zones

    return {
        "walls": walls,
        "hider_zones": hider_zones,
        "seeker_zones": seeker_zones,
        "layout_type": layout_type,
    }


# ─────────────────────────────────────────────────────────
# Floor tile grid (OpenAI-style seam lines)
# ─────────────────────────────────────────────────────────

_TILE_SIZE = 2.0        # metres per tile
_LINE_HALF_W = 0.02     # half-width of each grid line (thin dark seam)
_GRID_EXTENT = 50.0     # how far grid lines extend (covers visible floor)
_LINE_Z = 0.002         # tiny offset above floor to avoid z-fighting


def _tile_grid_xml(grid_rgba: str) -> str:
    """Generate thin box geoms forming a tile grid on the floor plane."""
    lines: list[str] = []
    idx = 0
    # Lines parallel to X axis (east-west)
    y = -_GRID_EXTENT
    while y <= _GRID_EXTENT + 0.01:
        lines.append(
            f'    <geom name="gridY_{idx}" type="box" '
            f'pos="0 {y:.2f} {_LINE_Z}" '
            f'size="{_GRID_EXTENT:.1f} {_LINE_HALF_W} {_LINE_Z}" '
            f'rgba="{grid_rgba}" contype="0" conaffinity="0"/>'
        )
        idx += 1
        y += _TILE_SIZE
    # Lines parallel to Y axis (north-south)
    x = -_GRID_EXTENT
    while x <= _GRID_EXTENT + 0.01:
        lines.append(
            f'    <geom name="gridX_{idx}" type="box" '
            f'pos="{x:.2f} 0 {_LINE_Z}" '
            f'size="{_LINE_HALF_W} {_GRID_EXTENT:.1f} {_LINE_Z}" '
            f'rgba="{grid_rgba}" contype="0" conaffinity="0"/>'
        )
        idx += 1
        x += _TILE_SIZE
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────

def generate_arena(
    rng: Generator,
    n_hiders: int = 2,
    n_seekers: int = 2,
    n_boxes: int | None = None,
    n_ramps: int | None = None,
    n_boxes_range: tuple[int, int] | None = None,
    n_ramps_range: tuple[int, int] | None = None,
    allowed_layouts: list[str] | None = None,
) -> tuple[str, dict]:
    """
    Generate a complete MuJoCo XML string for a randomised Hide & Seek arena.

    Args:
        rng: numpy random Generator for reproducibility
        n_hiders: number of hiders
        n_seekers: number of seekers
        n_boxes: exact number of boxes (overrides n_boxes_range)
        n_ramps: exact number of ramps (overrides n_ramps_range)
        n_boxes_range: (min, max) inclusive range for random box count (curriculum)
        n_ramps_range: (min, max) inclusive range for random ramp count (curriculum)
        allowed_layouts: list of allowed layout types for curriculum control

    Returns:
        (xml_string, metadata_dict) where metadata contains spawn positions
        and object counts for the environment to use.
    """
    global _wall_counter
    _wall_counter = 0

    A = ARENA_HALF

    # ── Random layout (optionally curriculum-filtered) ──
    layout = _gen_layout(rng, allowed_layouts=allowed_layouts)

    # ── Random object counts (curriculum-aware) ──
    is_open = layout["layout_type"] == "open"
    if n_boxes is None:
        if n_boxes_range is not None:
            n_boxes = int(rng.integers(n_boxes_range[0], n_boxes_range[1] + 1))
        elif is_open:
            n_boxes = int(rng.integers(4, 6))   # 4-5 boxes on open maps
        else:
            n_boxes = int(rng.integers(2, 6))   # 2-5 boxes otherwise
    if n_ramps is None:
        if n_ramps_range is not None:
            n_ramps = int(rng.integers(n_ramps_range[0], n_ramps_range[1] + 1))
        elif is_open:
            n_ramps = int(rng.integers(1, 3))   # 1-2 ramps on open maps
        else:
            n_ramps = int(rng.integers(0, 3))   # 0-2 ramps otherwise

    # ── Build XML ──
    # Pick a random lighting / colour mood
    mood_name = rng.choice(list(_MOODS.keys()))
    mood = _MOODS[mood_name]
    wall_rgba = mood["wall_rgba"]
    box_rgba = mood["box_rgba"]
    ramp_rgba = mood["ramp_rgba"]
    terrain_rgba = mood["terrain_rgba"]

    # First, generate ramp data so we know the mesh assets needed
    ramp_bodies = []
    ramp_mesh_assets = []
    all_zones_pre = layout["hider_zones"] + layout["seeker_zones"]

    # Ramps have a large footprint (2.4m x 0.8m).  Inset spawn positions so
    # the full ramp body (at any yaw) stays inside the arena walls.
    _RAMP_MARGIN = RAMP_LENGTH / 2 + 0.3  # half-diagonal + safety buffer

    for ri in range(n_ramps):
        zone = all_zones_pre[rng.integers(len(all_zones_pre))]
        # Clamp zone inward by ramp margin so ramp can't clip outer walls
        z_xmin = max(zone[0], -A + _RAMP_MARGIN)
        z_xmax = min(zone[1],  A - _RAMP_MARGIN)
        z_ymin = max(zone[2], -A + _RAMP_MARGIN)
        z_ymax = min(zone[3],  A - _RAMP_MARGIN)
        # If the clamped zone is degenerate, fall back to arena centre area
        if z_xmin >= z_xmax or z_ymin >= z_ymax:
            z_xmin, z_xmax = -A/2, A/2
            z_ymin, z_ymax = -A/2, A/2
        rx = float(rng.uniform(z_xmin, z_xmax))
        ry = float(rng.uniform(z_ymin, z_ymax))
        yaw = float(rng.uniform(0, 2 * np.pi))  # random facing direction
        body_xml, vert_str = _ramp_xml(ri, rx, ry, yaw, ramp_rgba=ramp_rgba)
        ramp_bodies.append(body_xml)
        ramp_mesh_assets.append(
            f'    <mesh name="ramp_wedge_{ri}" vertex="{vert_str}"/>'
        )

    # Build mesh_assets block
    mesh_block = "\n".join(ramp_mesh_assets) if ramp_mesh_assets else ""

    # Format header with mood-dependent lighting, floor, skybox
    header = _XML_HEADER.format(
        mesh_assets=mesh_block,
        skybox_rgb1=mood["skybox_rgb1"],
        skybox_rgb2=mood["skybox_rgb2"],
        floor_rgb=mood["floor_rgb"],
        headlight=mood["headlight"],
        key_light=mood["key_light"],
        fill_light=mood["fill_light"],
        rim_light=mood["rim_light"],
    )
    xml_parts = [header]

    # ── Floor grid lines (OpenAI-style tile seams) ──
    grid_rgba = mood["grid_rgba"]
    xml_parts.append(_tile_grid_xml(grid_rgba))

    # ── Decorative terrain outside the arena walls (OpenAI style) ──
    terrain_idx = 0
    for tx in np.arange(-_TERRAIN_EXTENT, _TERRAIN_EXTENT + 0.1, _TERRAIN_STEP):
        for ty in np.arange(-_TERRAIN_EXTENT, _TERRAIN_EXTENT + 0.1, _TERRAIN_STEP):
            # Skip blocks that would overlap the arena interior
            if abs(tx) < A + 0.5 and abs(ty) < A + 0.5:
                continue
            # Random height with some spatial variation
            h = float(rng.uniform(_TERRAIN_H_MIN, _TERRAIN_H_MAX))
            # Slight position jitter for organic feel
            jx = float(rng.uniform(-0.2, 0.2))
            jy = float(rng.uniform(-0.2, 0.2))
            xml_parts.append(
                f'    <body name="terrain_{terrain_idx}" pos="{tx + jx:.2f} {ty + jy:.2f} {h:.3f}">\n'
                f'      <geom type="box" size="{_TERRAIN_HX:.2f} {_TERRAIN_HY:.2f} {h:.3f}" '
                f'rgba="{terrain_rgba}" contype="0" conaffinity="0"/>\n'
                f'    </body>\n'
            )
            terrain_idx += 1

    # Outer walls (tall -- unscalable)
    # Wall half-thickness
    OH = WALL_H_OUTER
    WT = 0.2  # half-thickness of each wall segment
    # North/south span the inner gap between east/west walls (no corner overlap)
    ns_hx = A - WT
    xml_parts.append(f'    <body name="wall_north" pos="0 {A} {OH}">\n'
                     f'      <geom type="box" size="{ns_hx:.2f} {WT} {OH}" '
                     f'rgba="{wall_rgba}" conaffinity="1" condim="3"/>\n    </body>\n')
    xml_parts.append(f'    <body name="wall_south" pos="0 {-A} {OH}">\n'
                     f'      <geom type="box" size="{ns_hx:.2f} {WT} {OH}" '
                     f'rgba="{wall_rgba}" conaffinity="1" condim="3"/>\n    </body>\n')
    # East/west span the full height including corners
    ew_hy = A + WT
    xml_parts.append(f'    <body name="wall_east" pos="{A} 0 {OH}">\n'
                     f'      <geom type="box" size="{WT} {ew_hy:.2f} {OH}" '
                     f'rgba="{wall_rgba}" conaffinity="1" condim="3"/>\n    </body>\n')
    xml_parts.append(f'    <body name="wall_west" pos="{-A} 0 {OH}">\n'
                     f'      <geom type="box" size="{WT} {ew_hy:.2f} {OH}" '
                     f'rgba="{wall_rgba}" conaffinity="1" condim="3"/>\n    </body>\n')

    # Interior walls
    for wall_spec in layout["walls"]:
        orientation, pos, start, end, door_positions, door_width = wall_spec
        if orientation == "v":
            xml_parts.append(_vertical_wall_xml(pos, start, end, door_positions, door_width,
                                                wall_rgba=wall_rgba))
        else:
            xml_parts.append(_horizontal_wall_xml(pos, start, end, door_positions, door_width,
                                                  wall_rgba=wall_rgba))

    # ── Spawn agents ──
    hider_labels = ["Hider Jacob", "Hider Lucas"]
    seeker_labels = ["Seeker Mikael", "Seeker Kriss"]

    hider_positions = []
    for i in range(n_hiders):
        zone = layout["hider_zones"][rng.integers(len(layout["hider_zones"]))]
        x = float(rng.uniform(zone[0], zone[1]))
        y = float(rng.uniform(zone[2], zone[3]))
        hider_positions.append((x, y))
        xml_parts.append(_agent_xml(
            f"hider_{i}", hider_labels[i], x, y,
            _HIDER_RGBA, "0.31 0.76 0.97"
        ))

    seeker_positions = []
    for i in range(n_seekers):
        zone = layout["seeker_zones"][rng.integers(len(layout["seeker_zones"]))]
        x = float(rng.uniform(zone[0], zone[1]))
        y = float(rng.uniform(zone[2], zone[3]))
        seeker_positions.append((x, y))
        xml_parts.append(_agent_xml(
            f"seeker_{i}", seeker_labels[i], x, y,
            _SEEKER_RGBA, "0.94 0.33 0.31"
        ))

    # ── Spawn boxes (near walls/doors for strategic interest) ──
    all_zones = layout["hider_zones"] + layout["seeker_zones"]
    for bi in range(n_boxes):
        zone = all_zones[rng.integers(len(all_zones))]
        bx = float(rng.uniform(zone[0], zone[1]))
        by = float(rng.uniform(zone[2], zone[3]))
        # Random box shape: some square, some elongated
        if rng.random() < 0.3:
            hx, hy = float(rng.uniform(0.7, 1.2)), 0.3
        else:
            sz = float(rng.uniform(0.35, 0.6))
            hx, hy = sz, sz
        mass = float(rng.uniform(1.5, 3.0))
        xml_parts.append(_box_xml(bi, bx, by, hx, hy, 0.3, mass, box_rgba=box_rgba))

    # ── Spawn ramps (already generated above for mesh asset collection) ──
    for body_xml in ramp_bodies:
        xml_parts.append(body_xml)

    # ── Actuators ──
    actuator_lines = []
    for i in range(n_hiders):
        actuator_lines.append(
            f'    <motor name="hider_{i}_x" joint="hider_{i}_joint" '
            f'gear="50 0 0 0 0 0" ctrlrange="-1.0 1.0"/>')
        actuator_lines.append(
            f'    <motor name="hider_{i}_y" joint="hider_{i}_joint" '
            f'gear="0 50 0 0 0 0" ctrlrange="-1.0 1.0"/>')
    for i in range(n_seekers):
        actuator_lines.append(
            f'    <motor name="seeker_{i}_x" joint="seeker_{i}_joint" '
            f'gear="50 0 0 0 0 0" ctrlrange="-1.0 1.0"/>')
        actuator_lines.append(
            f'    <motor name="seeker_{i}_y" joint="seeker_{i}_joint" '
            f'gear="0 50 0 0 0 0" ctrlrange="-1.0 1.0"/>')

    xml_parts.append(_XML_FOOTER.format(actuators="\n".join(actuator_lines)))

    # ── Metadata ──
    metadata = {
        "layout_type": layout["layout_type"],
        "mood": mood_name,
        "n_boxes": n_boxes,
        "n_ramps": n_ramps,
        "hider_positions": hider_positions,
        "seeker_positions": seeker_positions,
        "hider_zones": layout["hider_zones"],
        "seeker_zones": layout["seeker_zones"],
    }

    return "".join(xml_parts), metadata


# ─────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import mujoco

    rng = np.random.default_rng(42)

    for i in range(5):
        xml, meta = generate_arena(rng)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        print(f"Layout {i+1}: {meta['layout_type']:8s} | {meta['mood']:7s} | "
              f"{meta['n_boxes']} boxes, {meta['n_ramps']} ramps | "
              f"bodies={model.nbody} geoms={model.ngeom}")

    print("\nAll 5 procedural arenas generated and simulated successfully!")
