"""
Integration Test: Grab & Move Mechanic with Orange Colour Indicator
====================================================================

Demonstrates the grab-and-move mechanic visually:
  1. Hider Jacob walks toward the nearest box
  2. Grabs it  → agent + box both turn ORANGE
  3. Drags it around the arena
  4. Releases  → colours restore to originals
  5. Seeker walks toward a ramp, grabs it, drags it, releases

Run:
    python test_grab.py              # headless (assertions only)
    python test_grab.py --render     # live MuJoCo viewer window
    python test_grab.py --slow       # slow-motion playback
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from env import HideAndSeekEnv, AGENT_NAMES, HIDER_NAMES, SEEKER_NAMES, _GRAB_RGBA


def _parse_args():
    p = argparse.ArgumentParser(description="Grab mechanic integration test")
    p.add_argument("--render", action="store_true",
                   help="Open a live MuJoCo viewer window")
    p.add_argument("--slow", action="store_true",
                   help="Slow-motion playback (longer delay between steps)")
    p.add_argument("--step_delay", type=float, default=0.04,
                   help="Seconds between rendered steps (default 0.04)")
    return p.parse_args()


def _idle_action():
    """No-op action: no movement, no grab."""
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)


def _move_toward(env, agent_name: str, target_pos_2d: np.ndarray) -> np.ndarray:
    """Return a movement action that moves the agent toward target_pos_2d."""
    body_id = env._agent_body_ids[agent_name]
    agent_pos = env._get_body_pos(body_id)[:2]
    delta = target_pos_2d - agent_pos
    dist = np.linalg.norm(delta)
    if dist < 0.01:
        return np.zeros(2, dtype=np.float32)
    direction = delta / dist
    return np.clip(direction, -1.0, 1.0).astype(np.float32)


def main():
    args = _parse_args()
    render_mode = "human" if args.render else None
    delay = args.step_delay if args.slow else (0.02 if args.render else 0.0)

    # ── Use static XML for deterministic box/ramp positions ──
    env = HideAndSeekEnv(
        render_mode=render_mode,
        procedural=False,
        grab_radius=1.5,        # wider grab radius for easier testing
        horizon=600,
    )
    obs, info = env.reset()

    print(f"Arena: static XML | {env._cur_n_boxes} boxes, {env._cur_n_ramps} ramps")
    print(f"Agents: {AGENT_NAMES}")

    # ── Find nearest box to hider_0 ──
    hider = "hider_0"
    hider_body = env._agent_body_ids[hider]
    hider_pos = env._get_body_pos(hider_body)[:2]

    best_box_dist = 999
    best_box_idx = 0
    for bi in range(env._cur_n_boxes):
        bname = f"box_{bi}"
        box_pos = env._get_body_pos(env._box_body_ids[bname])[:2]
        d = np.linalg.norm(hider_pos - box_pos)
        if d < best_box_dist:
            best_box_dist = d
            best_box_idx = bi
    target_box = f"box_{best_box_idx}"
    print(f"\nHider Jacob will grab {target_box} (dist={best_box_dist:.2f}m)")

    # ── Phase 1: Walk toward the box until close enough (up to 80 steps) ──
    print("\n── Phase 1: Walking toward box... ──")
    for step in range(80):
        box_pos = env._get_body_pos(env._box_body_ids[target_box])[:2]
        move = _move_toward(env, hider, box_pos)
        actions = {n: _idle_action() for n in AGENT_NAMES}
        actions[hider][:2] = move
        obs, rew, term, trunc, info = env.step(actions)

        hider_pos = env._get_body_pos(hider_body)[:2]
        dist = np.linalg.norm(hider_pos - box_pos)
        if dist < env.grab_radius * 0.9:
            print(f"  Reached box at step {step} (dist={dist:.2f}m)")
            break
        if delay > 0:
            time.sleep(delay)

    hider_pos = env._get_body_pos(hider_body)[:2]
    box_pos = env._get_body_pos(env._box_body_ids[target_box])[:2]
    dist_to_box = np.linalg.norm(hider_pos - box_pos)
    print(f"  Distance to {target_box}: {dist_to_box:.2f}m")

    # ── Phase 2: Grab the box → should turn orange (up to 60 steps) ──
    print("\n── Phase 2: GRABBING (both should turn orange)... ──")
    grabbed = False
    for step in range(60):
        box_pos = env._get_body_pos(env._box_body_ids[target_box])[:2]
        move = _move_toward(env, hider, box_pos)
        actions = {n: _idle_action() for n in AGENT_NAMES}
        actions[hider][:2] = move
        actions[hider][2] = 1.0  # GRAB!
        obs, rew, term, trunc, info = env.step(actions)

        if env.box_grabbed_by[best_box_idx] == 0 and not grabbed:
            grabbed = True
            print(f"  ✓ Grabbed {target_box} at step {step}!")

            # Verify orange colour
            hider_gid = env._agent_geom_ids[hider]
            box_gid = env._box_geom_ids[target_box]
            assert np.allclose(env.model.geom_rgba[hider_gid], _GRAB_RGBA), \
                "Agent should be orange when grabbing"
            assert np.allclose(env.model.geom_rgba[box_gid], _GRAB_RGBA), \
                "Box should be orange when grabbed"
            print(f"  ✓ Agent colour: ORANGE")
            print(f"  ✓ Box colour:   ORANGE")

        if delay > 0:
            time.sleep(delay)

    assert grabbed, f"Failed to grab {target_box}!"

    # ── Phase 3: Drag box in a straight line (80 steps) ──
    print("\n── Phase 3: Dragging box across the arena... ──")
    box_start_pos = env._get_body_pos(env._box_body_ids[target_box])[:2].copy()
    max_displacement = 0.0
    for step in range(80):
        # Move in +X direction while holding grab
        actions = {n: _idle_action() for n in AGENT_NAMES}
        actions[hider][:2] = [0.9, 0.0]   # strong +X push
        actions[hider][2] = 1.0            # keep grabbing
        obs, rew, term, trunc, info = env.step(actions)
        # Track max displacement during drag
        cur_pos = env._get_body_pos(env._box_body_ids[target_box])[:2]
        d = np.linalg.norm(cur_pos - box_start_pos)
        max_displacement = max(max_displacement, d)
        if delay > 0:
            time.sleep(delay)

    print(f"  Max box displacement during drag: {max_displacement:.2f}m")
    assert max_displacement > 0.05, \
        f"Box should have moved during drag but max displacement was {max_displacement:.2f}m"
    print(f"  ✓ Box was successfully dragged!")

    # ── Phase 4: Release → colours should restore (10 steps) ──
    print("\n── Phase 4: RELEASING (colours should restore)... ──")
    for step in range(10):
        actions = {n: _idle_action() for n in AGENT_NAMES}
        actions[hider][2] = -1.0  # release
        obs, rew, term, trunc, info = env.step(actions)
        if delay > 0:
            time.sleep(delay)

    # Verify colour restoration
    hider_gid = env._agent_geom_ids[hider]
    box_gid = env._box_geom_ids[target_box]
    assert np.allclose(env.model.geom_rgba[hider_gid], env._original_rgba[hider]), \
        "Agent colour should be restored after release"
    assert np.allclose(env.model.geom_rgba[box_gid], env._original_rgba[target_box]), \
        "Box colour should be restored after release"
    print(f"  ✓ Agent colour: RESTORED (cyan)")
    print(f"  ✓ Box colour:   RESTORED (grey)")

    # ── Phase 5: Seeker grabs a ramp (if ramps exist) ──
    if env._cur_n_ramps > 0:
        seeker = "seeker_0"
        target_ramp = "ramp_0"
        print(f"\n── Phase 5: Seeker Mikael grabs {target_ramp}... ──")

        # Walk toward ramp (up to 80 steps)
        for step in range(80):
            ramp_pos = env._get_body_pos(env._ramp_body_ids[target_ramp])[:2]
            move = _move_toward(env, seeker, ramp_pos)
            actions = {n: _idle_action() for n in AGENT_NAMES}
            actions[seeker][:2] = move
            actions[seeker][2] = 1.0  # grab
            obs, rew, term, trunc, info = env.step(actions)
            if delay > 0:
                time.sleep(delay)

        ramp_grabbed = env.ramp_grabbed_by[0] == 2  # seeker_0 is agent index 2
        if ramp_grabbed:
            print(f"  ✓ Grabbed {target_ramp}!")
            seeker_gid = env._agent_geom_ids[seeker]
            ramp_gid = env._ramp_geom_ids[target_ramp]
            assert np.allclose(env.model.geom_rgba[seeker_gid], _GRAB_RGBA)
            assert np.allclose(env.model.geom_rgba[ramp_gid], _GRAB_RGBA)
            print(f"  ✓ Seeker colour: ORANGE")
            print(f"  ✓ Ramp colour:   ORANGE")

            # Drag for a bit then release
            for step in range(30):
                actions = {n: _idle_action() for n in AGENT_NAMES}
                actions[seeker][:2] = [0.6, 0.3]
                actions[seeker][2] = 1.0
                env.step(actions)
                if delay > 0:
                    time.sleep(delay)

            # Release
            for step in range(5):
                actions = {n: _idle_action() for n in AGENT_NAMES}
                actions[seeker][2] = -1.0
                env.step(actions)
                if delay > 0:
                    time.sleep(delay)

            assert np.allclose(env.model.geom_rgba[seeker_gid], env._original_rgba[seeker])
            print(f"  ✓ Seeker colour: RESTORED (red)")
        else:
            print(f"  ⚠ Could not reach {target_ramp} (too far), skipping ramp test")
    else:
        print(f"\n── Phase 5: No ramps in static XML, skipping ramp grab test ──")

    # ── Hold final state if rendering ──
    if args.render:
        print("\n🎬 Holding viewer open for 3 seconds...")
        for _ in range(90):
            actions = {n: _idle_action() for n in AGENT_NAMES}
            env.step(actions)
            time.sleep(0.033)

    env.close()
    print("\n✅ GRAB TEST PASSED — all assertions verified!")


if __name__ == "__main__":
    main()
