"""
Integration Test: Agent Can Climb the Wedge Ramp
=================================================

Creates a minimal MuJoCo scene with:
  - A flat ground plane
  - One agent (sphere with freejoint, same physics as in env.py)
  - One wedge ramp (same mesh geometry as worldgen.py)

The agent is placed directly behind the low (approach) end of the ramp,
facing +X.  A constant +X force is applied for several hundred timesteps.
The test asserts that the agent's Z-position increases significantly —
proving the ramp is physically climbable.

Run:
    python test_ramp_climb.py              # headless (assertions only)
    python test_ramp_climb.py --render     # live MuJoCo viewer window
    python test_ramp_climb.py --record     # save MP4 video to disk
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import mujoco

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ── Import ramp constants from worldgen ──
from worldgen import (
    RAMP_LENGTH, RAMP_WIDTH, RAMP_HEIGHT,
    AGENT_RADIUS, AGENT_Z,
)

# ── Optional renderer imports (graceful fallback) ──
try:
    import mujoco.viewer as _mj_viewer
    _HAS_VIEWER = True
except ImportError:
    _HAS_VIEWER = False


def _parse_args():
    p = argparse.ArgumentParser(description="Ramp climb integration test")
    p.add_argument("--render", action="store_true",
                   help="Open a live MuJoCo viewer window")
    p.add_argument("--record", action="store_true",
                   help="Record the test as an MP4 video")
    p.add_argument("--slow", action="store_true",
                   help="Slow-motion playback (add delay between steps)")
    p.add_argument("--step_delay", type=float, default=0.03,
                   help="Seconds between rendered steps (default 0.03)")
    return p.parse_args()


def _build_test_xml() -> str:
    """Build a minimal MuJoCo XML with one agent and one ramp, perfectly aligned."""
    L = RAMP_LENGTH   # 2.4
    W = RAMP_WIDTH    # 0.8
    H = RAMP_HEIGHT   # 0.85
    hw = W / 2

    # Ramp at origin, yaw=0: low side at -X, tall edge at +X
    # Same vertex layout as worldgen._ramp_xml
    verts = [
        f"{-L/2:.4f} {-hw:.4f} {-H/3:.4f}",
        f"{-L/2:.4f} { hw:.4f} {-H/3:.4f}",
        f"{ L/2:.4f} {-hw:.4f} {-H/3:.4f}",
        f"{ L/2:.4f} { hw:.4f} {-H/3:.4f}",
        f"{ L/2:.4f} {-hw:.4f} { 2*H/3:.4f}",
        f"{ L/2:.4f} { hw:.4f} { 2*H/3:.4f}",
    ]
    vert_str = "  ".join(verts)
    body_z = H / 3 + 0.005  # same as worldgen

    # Agent starts just behind the low edge of the ramp, aligned on Y=0
    # The ramp's low edge is at x = -L/2 in body frame, ramp body is at x=0
    # So the low edge in world coords is at x = 0 - L/2 = -1.2
    # Place agent a bit behind that so it can accelerate into the slope
    agent_x = -L / 2 - AGENT_RADIUS - 0.3   # ~-1.85
    agent_y = 0.0

    xml = f"""\
<mujoco model="ramp_climb_test">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.02" gravity="0 0 -9.81" integrator="Euler">
    <flag warmstart="enable"/>
  </option>
  <default>
    <geom condim="6" friction="0.4 0.3 0.1" margin="0.001"/>
    <joint damping="0.5" armature="0.01"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0"/>
  </default>
  <asset>
    <mesh name="ramp_wedge_0" vertex="{vert_str}"/>
    <texture type="skybox" builtin="gradient"
             rgb1="0.165 0.165 0.243" rgb2="0.06 0.06 0.10"
             width="512" height="512"/>
    <texture name="groundplane" type="2d" builtin="checker"
             rgb1="0.24 0.30 0.34" rgb2="0.18 0.24 0.28"
             width="512" height="512"/>
    <material name="groundplane" texture="groundplane" texrepeat="10 10"
              texuniform="true" reflectance="0.15"/>
  </asset>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
  </visual>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" material="groundplane"
          conaffinity="1" condim="6"/>
    <light name="sun" pos="3 -3 8" dir="-0.3 0.3 -1" diffuse="1 0.95 0.9"
           specular="0.4 0.4 0.4" castshadow="true"/>

    <!-- Nice camera angle to watch the climb -->
    <camera name="track_side" pos="-0.5 -4.0 2.5" xyaxes="1 0 0 0 0.45 0.89" fovy="50"/>
    <camera name="track_top" pos="0 0 6" euler="0 0 0" fovy="60"/>

    <!-- Backstop wall so agent doesn't fly off forever -->
    <body name="backstop" pos="3.5 0 1.0">
      <geom type="box" size="0.2 2.0 1.0" rgba="0.4 0.4 0.4 0.3"
            conaffinity="1" condim="3"/>
    </body>

    <!-- Ramp at origin, approach from -X -->
    <body name="ramp_0" pos="0.000 0.000 {body_z:.4f}" euler="0 0 0">
      <freejoint name="ramp_0_joint"/>
      <geom name="ramp_0_geom" type="mesh" mesh="ramp_wedge_0"
            mass="5.0" rgba="0.506 0.780 0.518 1"
            conaffinity="1" condim="6" friction="0.8 0.01 0.001"/>
    </body>

    <!-- Agent: sphere with freejoint, same as env.py -->
    <body name="agent" pos="{agent_x:.3f} {agent_y:.3f} {AGENT_Z}">
      <freejoint name="agent_joint"/>
      <geom name="agent_geom" type="sphere" size="{AGENT_RADIUS}" density="11"
            rgba="0.31 0.76 0.97 1" conaffinity="1" condim="6"
            friction="0.3 0.1 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="agent_x" joint="agent_joint" gear="50 0 0 0 0 0"
           ctrlrange="-1.0 1.0"/>
    <motor name="agent_y" joint="agent_joint" gear="0 50 0 0 0 0"
           ctrlrange="-1.0 1.0"/>
  </actuator>
</mujoco>
"""
    return xml


def test_ramp_climb(render: bool = False, record: bool = False,
                    slow: bool = False, step_delay: float = 0.03):
    """
    Drive an agent straight into the ramp at full throttle (+X) and verify
    its Z-position rises while the agent is on the ramp surface.

    We track Z specifically while the agent's X-coordinate is within the
    ramp footprint (−L/2 to +L/2), so bouncing off into the sky doesn't
    count as "climbing".
    """
    xml = _build_test_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Find body and actuator IDs
    agent_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "agent")
    ramp_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ramp_0")
    act_x_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "agent_x")

    assert agent_body_id >= 0, "Agent body not found"
    assert ramp_body_id >= 0, "Ramp body not found"
    assert act_x_id >= 0, "Agent X actuator not found"

    # ── Set up renderer / viewer ──
    viewer = None
    renderer = None
    frames = []

    if render and _HAS_VIEWER:
        viewer = mujoco.viewer.launch_passive(model, data)
        # Set camera to our nice side view
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "track_side")
        if cam_id >= 0:
            viewer.cam.fixedcamid = cam_id
    elif render:
        print("⚠️  mujoco.viewer not available — falling back to --record mode")
        record = True

    if record:
        renderer = mujoco.Renderer(model, height=720, width=1280)

    # Initial forward simulation to let things settle on the ground
    mujoco.mj_forward(model, data)

    # Record starting positions
    agent_z_start = data.xpos[agent_body_id][2]
    ramp_z_start = data.xpos[ramp_body_id][2]

    print(f"Agent start pos:  x={data.xpos[agent_body_id][0]:.3f}, "
          f"y={data.xpos[agent_body_id][1]:.3f}, z={agent_z_start:.3f}")
    print(f"Ramp  start pos:  x={data.xpos[ramp_body_id][0]:.3f}, "
          f"y={data.xpos[ramp_body_id][1]:.3f}, z={ramp_z_start:.3f}")
    print(f"Ramp dimensions:  L={RAMP_LENGTH}m, W={RAMP_WIDTH}m, H={RAMP_HEIGHT}m")
    print(f"Ramp low edge at x ≈ {-RAMP_LENGTH/2:.2f}, high edge at x ≈ {RAMP_LENGTH/2:.2f}")
    if render:
        print(f"🎥 Rendering {'(viewer)' if viewer else '(recording)'} ...")
    print()

    def _sync_render():
        """Push current frame to viewer or capture buffer."""
        if viewer is not None:
            viewer.sync()
            if slow:
                time.sleep(step_delay)
        if renderer is not None:
            renderer.update_scene(data, camera="track_side")
            frames.append(renderer.render().copy())

    # ── Phase 1: Let the scene settle (50 steps, ~1 second) ──
    for _ in range(50):
        data.ctrl[act_x_id] = 0.0
        mujoco.mj_step(model, data)
        _sync_render()

    agent_z_settled = data.xpos[agent_body_id][2]
    print(f"After settling:   z={agent_z_settled:.4f}")

    # Ramp footprint boundaries in world X
    ramp_x_lo = -RAMP_LENGTH / 2 - AGENT_RADIUS
    ramp_x_hi = RAMP_LENGTH / 2 + AGENT_RADIUS + 0.5

    # ── Phase 2: Drive forward at full throttle ──
    max_z_on_ramp = agent_z_settled
    max_z_overall = agent_z_settled
    n_drive_steps = 300  # 6 seconds — enough to cross the 2.4m ramp

    for step in range(n_drive_steps):
        data.ctrl[act_x_id] = 1.0  # full +X force
        mujoco.mj_step(model, data)
        _sync_render()

        agent_z = data.xpos[agent_body_id][2]
        agent_x = data.xpos[agent_body_id][0]

        if agent_z > max_z_overall:
            max_z_overall = agent_z

        if ramp_x_lo <= agent_x <= ramp_x_hi:
            if agent_z > max_z_on_ramp:
                max_z_on_ramp = agent_z

        if (step + 1) % 50 == 0:
            on_ramp = "ON RAMP" if ramp_x_lo <= agent_x <= ramp_x_hi else "       "
            print(f"  Step {step+1:4d}: x={agent_x:+.3f}, z={agent_z:.4f}, "
                  f"max_z_ramp={max_z_on_ramp:.4f}  {on_ramp}")

    # ── Hold for a moment so user can see the final state ──
    if viewer is not None:
        for _ in range(100):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.02)

    # ── Clean up viewer ──
    if viewer is not None:
        viewer.close()
    if renderer is not None:
        renderer.close()

    # ── Save video ──
    if record and frames:
        try:
            import imageio
            video_path = "test_ramp_climb_test1.mp4"
            imageio.mimwrite(video_path, frames, fps=50)
            print(f"\n🎬 Video saved: {video_path} ({len(frames)} frames)")
        except ImportError:
            print("\n⚠️  imageio not installed — cannot save video. "
                  "Install with: pip install imageio[ffmpeg]")

    # ── Verify results ──
    z_gain = max_z_on_ramp - agent_z_settled
    print()
    print(f"Settled Z:         {agent_z_settled:.4f}")
    print(f"Max Z (on ramp):   {max_z_on_ramp:.4f}")
    print(f"Max Z (overall):   {max_z_overall:.4f}")
    print(f"Z gain (on ramp):  {z_gain:.4f} m")
    print(f"Ramp height:       {RAMP_HEIGHT:.2f} m")

    # The agent should gain at least 30% of the ramp height while on the ramp
    min_expected_gain = RAMP_HEIGHT * 0.30  # 0.255 m

    assert z_gain > min_expected_gain, (
        f"❌ FAIL: Agent Z gain on ramp ({z_gain:.4f}m) is less than "
        f"{min_expected_gain:.3f}m (30% of ramp height {RAMP_HEIGHT}m). "
        f"The ramp may not be climbable!"
    )

    print(f"\n✅ PASS: Agent climbed {z_gain:.3f}m on the ramp surface "
          f"(≥ {min_expected_gain:.3f}m = 30% of {RAMP_HEIGHT}m ramp height)")

    # Additional check: did agent reach at least halfway up?
    halfway = RAMP_HEIGHT * 0.50
    if z_gain > halfway:
        print(f"✅ BONUS: Agent reached >{halfway:.2f}m (50%+ of ramp height)")
    else:
        print(f"⚠️  NOTE: Agent only reached {z_gain:.3f}m on ramp "
              f"(< 50% of ramp height). Consider tuning friction or force.")


def test_ramp_climb_from_worldgen(render: bool = False, record: bool = False,
                                  slow: bool = False, step_delay: float = 0.03):
    """
    Same test but using the actual worldgen.generate_arena() output,
    with a fixed seed that guarantees at least 1 ramp.
    """
    from worldgen import generate_arena

    # Use a fixed seed and force 1 ramp
    rng = np.random.default_rng(12345)
    xml, meta = generate_arena(rng, n_hiders=2, n_seekers=2, n_boxes=0, n_ramps=1)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    ramp_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ramp_0")
    assert ramp_body_id >= 0, "ramp_0 not found in worldgen arena"

    # Pick hider_0 as our test agent
    agent_name = "hider_0"
    agent_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, agent_name)
    act_x_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{agent_name}_x")
    act_y_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{agent_name}_y")

    assert agent_body_id >= 0, f"{agent_name} body not found"
    assert act_x_id >= 0, f"{agent_name}_x actuator not found"

    mujoco.mj_forward(model, data)

    # Teleport the agent right next to the ramp's low edge
    ramp_pos = data.xpos[ramp_body_id].copy()
    ramp_quat = data.xquat[ramp_body_id].copy()
    local_neg_x = np.array([-1.0, 0.0, 0.0])
    approach_dir = np.zeros(3)
    mujoco.mju_rotVecQuat(approach_dir, local_neg_x, ramp_quat)
    approach_dir[2] = 0
    approach_dir /= np.linalg.norm(approach_dir[:2]) + 1e-8

    offset_dist = RAMP_LENGTH / 2 + AGENT_RADIUS + 0.2
    spawn_pos = ramp_pos + approach_dir * offset_dist
    spawn_pos[2] = AGENT_Z

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{agent_name}_joint")
    qpos_adr = model.jnt_qposadr[joint_id]
    data.qpos[qpos_adr:qpos_adr + 3] = spawn_pos

    qvel_adr = model.jnt_dofadr[joint_id]
    data.qvel[qvel_adr:qvel_adr + 6] = 0.0

    mujoco.mj_forward(model, data)

    # ── Set up renderer / viewer ──
    viewer = None
    renderer = None
    frames = []

    if render and _HAS_VIEWER:
        viewer = mujoco.viewer.launch_passive(model, data)
        # Use the overview camera from the arena
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
        if cam_id >= 0:
            viewer.cam.fixedcamid = cam_id
    elif render:
        print("⚠️  mujoco.viewer not available — falling back to --record mode")
        record = True

    if record:
        renderer = mujoco.Renderer(model, height=720, width=1280)

    def _sync_render():
        if viewer is not None:
            viewer.sync()
            if slow:
                time.sleep(step_delay)
        if renderer is not None:
            renderer.update_scene(data, camera="overview")
            frames.append(renderer.render().copy())

    print(f"\n── Worldgen Arena Test ──")
    print(f"Layout: {meta['layout_type']}, Ramps: {meta['n_ramps']}")
    print(f"Ramp pos:  {ramp_pos}")
    print(f"Agent spawned at: {spawn_pos}")
    print(f"Approach direction: {approach_dir[:2]}")
    if render:
        print(f"🎥 Rendering {'(viewer)' if viewer else '(recording)'} ...")

    drive_dir = -approach_dir[:2]
    drive_dir /= np.linalg.norm(drive_dir) + 1e-8

    # Settle
    for _ in range(50):
        mujoco.mj_step(model, data)
        _sync_render()

    agent_z_settled = data.xpos[agent_body_id][2]
    print(f"Settled Z: {agent_z_settled:.4f}")

    # Drive toward the ramp
    max_z = agent_z_settled
    for step in range(500):
        data.ctrl[act_x_id] = float(drive_dir[0])
        data.ctrl[act_y_id] = float(drive_dir[1])
        mujoco.mj_step(model, data)
        _sync_render()

        z = data.xpos[agent_body_id][2]
        if z > max_z:
            max_z = z

        if (step + 1) % 100 == 0:
            x = data.xpos[agent_body_id][0]
            y = data.xpos[agent_body_id][1]
            print(f"  Step {step+1:4d}: x={x:+.3f}, y={y:+.3f}, z={z:.4f}, max_z={max_z:.4f}")

    # Hold for a moment
    if viewer is not None:
        for _ in range(100):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.02)

    # Clean up
    if viewer is not None:
        viewer.close()
    if renderer is not None:
        renderer.close()

    if record and frames:
        try:
            import imageio
            video_path = "test_ramp_climb_test2.mp4"
            imageio.mimwrite(video_path, frames, fps=50)
            print(f"\n🎬 Video saved: {video_path} ({len(frames)} frames)")
        except ImportError:
            print("\n⚠️  imageio not installed — cannot save video.")

    z_gain = max_z - agent_z_settled
    min_expected = RAMP_HEIGHT * 0.25  # slightly more lenient for worldgen test

    print(f"\nZ gain: {z_gain:.4f}m (need ≥ {min_expected:.3f}m)")
    assert z_gain > min_expected, (
        f"❌ FAIL: Agent Z gain ({z_gain:.4f}m) < {min_expected:.3f}m in worldgen arena"
    )
    print(f"✅ PASS: Agent climbed {z_gain:.3f}m in worldgen arena")


if __name__ == "__main__":
    args = _parse_args()

    print("=" * 60)
    print("TEST 1: Controlled ramp climb (manual XML)")
    print("=" * 60)
    test_ramp_climb(render=args.render, record=args.record,
                    slow=args.slow, step_delay=args.step_delay)

    print("\n" + "=" * 60)
    print("TEST 2: Ramp climb in worldgen-generated arena")
    print("=" * 60)
    test_ramp_climb_from_worldgen(render=args.render, record=args.record,
                                  slow=args.slow, step_delay=args.step_delay)

    print("\n" + "=" * 60)
    print("ALL RAMP CLIMB TESTS PASSED ✅")
    print("=" * 60)
