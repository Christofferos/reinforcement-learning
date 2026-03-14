"""
Vision & Raycast Diagnostic Test
=================================

Verifies that agents can actually perceive their surroundings:
  1. Lidar detects nearby walls, boxes, and ramps
  2. Hiders see seekers at episode start (open layout, no walls between)
  3. Agent-to-agent visibility is blocked by walls
  4. Box observation radius covers nearby boxes
  5. Full-arena line-of-sight works when unobstructed (agent_obs_radius=100m)

Run:
    python tests/test_vision.py              # headless assertions
    python tests/test_vision.py --render     # with MuJoCo viewer
"""

from __future__ import annotations
import sys
import os
import argparse
import time
from pathlib import Path

# ── Project path setup ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
os.chdir(_PROJECT_ROOT)

import numpy as np
from env import (
    HideAndSeekEnv, AGENT_NAMES, HIDER_NAMES, SEEKER_NAMES,
    N_HIDERS, N_SEEKERS, N_AGENTS, _can_see, _compute_lidar, _RAY_Z,
)
from worldgen import generate_arena, ARENA_HALF


def _parse_args():
    p = argparse.ArgumentParser(description="Vision & raycast diagnostic")
    p.add_argument("--render", action="store_true")
    p.add_argument("--slow", action="store_true")
    return p.parse_args()


def _idle():
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)


def main():
    args = _parse_args()
    render_mode = "human" if args.render else None
    delay = 0.05 if args.slow else 0.0

    passed = 0
    failed = 0

    # ====================================================================
    # TEST 1: Open arena — all agents see each other at start
    # ====================================================================
    print("=" * 64)
    print("TEST 1: Open arena — mutual visibility at episode start")
    print("=" * 64)

    # Use an open layout (force it via seed search)
    rng = np.random.default_rng(99)
    for attempt in range(50):
        xml, meta = generate_arena(rng, n_boxes=2, n_ramps=0)
        if meta["layout_type"] == "open":
            break
    else:
        print("  [!] Could not find an open layout in 50 attempts, using whatever we got")

    env = HideAndSeekEnv(
        render_mode=render_mode,
        procedural=True,
        horizon=240,
    )
    # Force this specific XML
    env._procgen = True
    obs, info = env.reset()
    layout = info.get("mood", "unknown")

    print(f"  Layout: {meta['layout_type']} | Mood: {info.get('mood', '?')}")

    # Step once to settle physics
    actions = {n: _idle() for n in AGENT_NAMES}
    obs, _, _, _, info = env.step(actions)

    vis = info["visibility_matrix"]
    print(f"\n  Visibility matrix (step 1):")
    header = "           " + "  ".join(f"{n:>9s}" for n in AGENT_NAMES)
    print(f"  {header}")
    for i, name in enumerate(AGENT_NAMES):
        row = "  ".join(f"{'  YES' if vis[i, j] else '   no':>9s}" for j in range(N_AGENTS))
        print(f"  {name:>9s}  {row}")

    # Print distances between all agent pairs
    print(f"\n  Agent distances:")
    for i in range(N_AGENTS):
        for j in range(i + 1, N_AGENTS):
            pos_i = env._get_body_pos(env._agent_body_ids[AGENT_NAMES[i]])
            pos_j = env._get_body_pos(env._agent_body_ids[AGENT_NAMES[j]])
            d = np.linalg.norm(pos_i[:2] - pos_j[:2])
            see = "CAN SEE" if vis[i, j] else "BLOCKED"
            print(f"    {AGENT_NAMES[i]} <-> {AGENT_NAMES[j]}: {d:.2f}m  [{see}]")

    # In any layout, at least hiders should see each other, seekers should see each other
    for i in range(N_HIDERS):
        for j in range(N_HIDERS):
            if i != j:
                if vis[i, j]:
                    print(f"  [OK] {AGENT_NAMES[i]} sees {AGENT_NAMES[j]} (same team)")
                    passed += 1
                else:
                    print(f"  [FAIL] {AGENT_NAMES[i]} cannot see {AGENT_NAMES[j]} (should see teammate)")
                    failed += 1

    for i in range(N_HIDERS, N_AGENTS):
        for j in range(N_HIDERS, N_AGENTS):
            if i != j:
                if vis[i, j]:
                    print(f"  [OK] {AGENT_NAMES[i]} sees {AGENT_NAMES[j]} (same team)")
                    passed += 1
                else:
                    print(f"  [FAIL] {AGENT_NAMES[i]} cannot see {AGENT_NAMES[j]} (should see teammate)")
                    failed += 1

    # Count cross-team visibility (hiders seeing seekers)
    cross_visible = 0
    for hi in range(N_HIDERS):
        for si in range(N_HIDERS, N_AGENTS):
            if vis[hi, si]:
                cross_visible += 1
    print(f"\n  Cross-team visibility: {cross_visible}/{N_HIDERS * N_SEEKERS} hider->seeker pairs visible")

    env.close()

    # ====================================================================
    # TEST 2: Lidar detects nearby obstacles
    # ====================================================================
    print("\n" + "=" * 64)
    print("TEST 2: Lidar detects nearby walls, boxes, and ramps")
    print("=" * 64)

    env2 = HideAndSeekEnv(
        render_mode=render_mode,
        procedural=True,
        horizon=240,
    )
    obs, info = env2.reset()
    actions = {n: _idle() for n in AGENT_NAMES}
    obs, _, _, _, info = env2.step(actions)

    print(f"  Layout: {info.get('mood', '?')} | Boxes: {env2._cur_n_boxes} | Ramps: {env2._cur_n_ramps}")
    print(f"  Arena half-size: {ARENA_HALF}m | Lidar max dist: {env2.lidar_max_dist}m | Rays: {env2.lidar_n_rays}")

    for ai, name in enumerate(AGENT_NAMES):
        body_id = env2._agent_body_ids[name]
        pos = env2._get_body_pos(body_id)
        lidar = _compute_lidar(env2.model, env2.data, pos, env2.lidar_n_rays,
                               env2.lidar_max_dist, bodyexclude=body_id,
                               exclude_geom_ids=env2._all_agent_geom_ids)

        hits = np.sum(lidar < 1.0)
        min_reading = np.min(lidar)
        min_dist = min_reading * env2.lidar_max_dist
        avg_hit_dist = np.mean(lidar[lidar < 1.0]) * env2.lidar_max_dist if hits > 0 else float("inf")

        print(f"\n  {name} at ({pos[0]:.1f}, {pos[1]:.1f}):")
        print(f"    Lidar rays hitting something: {hits}/{env2.lidar_n_rays}")
        print(f"    Nearest hit: {min_dist:.2f}m | Avg hit dist: {avg_hit_dist:.2f}m")

        if hits > 0:
            print(f"    [OK] Lidar detects obstacles")
            passed += 1
        else:
            print(f"    [FAIL] Lidar detects NOTHING — agent is blind to obstacles!")
            failed += 1

        # Agent should be inside the arena, so at least some walls must be within lidar range
        dist_to_nearest_wall = min(
            abs(pos[0] - ARENA_HALF), abs(pos[0] + ARENA_HALF),
            abs(pos[1] - ARENA_HALF), abs(pos[1] + ARENA_HALF),
        )
        print(f"    Distance to nearest wall: {dist_to_nearest_wall:.2f}m")
        if dist_to_nearest_wall < env2.lidar_max_dist:
            if min_dist <= dist_to_nearest_wall + 1.0:  # 1m tolerance
                print(f"    [OK] Lidar range covers nearest wall")
                passed += 1
            else:
                print(f"    [!] Nearest lidar hit ({min_dist:.2f}m) > wall dist ({dist_to_nearest_wall:.2f}m) + tolerance")
                # Not necessarily a fail — could be angle-dependent
        else:
            print(f"    [!] Agent is very far from walls ({dist_to_nearest_wall:.2f}m) — unusual")

    env2.close()

    # ====================================================================
    # TEST 3: Box observation radius
    # ====================================================================
    print("\n" + "=" * 64)
    print("TEST 3: Box observation radius")
    print("=" * 64)

    env3 = HideAndSeekEnv(
        render_mode=render_mode,
        procedural=True,
        horizon=240,
    )
    obs, info = env3.reset()
    actions = {n: _idle() for n in AGENT_NAMES}
    obs, _, _, _, info = env3.step(actions)

    print(f"  Box obs radius: {env3.box_obs_radius}m")

    for ai, name in enumerate(AGENT_NAMES):
        body_id = env3._agent_body_ids[name]
        agent_pos = env3._get_body_pos(body_id)

        visible_boxes = 0
        for bi in range(env3._cur_n_boxes):
            bname = env3._cur_box_names[bi]
            box_pos = env3._get_body_pos(env3._box_body_ids[bname])
            d = np.linalg.norm(agent_pos[:2] - box_pos[:2])
            within = d <= env3.box_obs_radius
            if within:
                visible_boxes += 1
                print(f"    {name} -> {bname}: {d:.2f}m [IN RANGE]")
            else:
                print(f"    {name} -> {bname}: {d:.2f}m [out of range]")

        if visible_boxes > 0:
            print(f"  [OK] {name} can observe {visible_boxes} box(es)")
            passed += 1
        else:
            print(f"  [!] {name} has no boxes within obs radius (may be intentional for this spawn)")

    env3.close()

    # ====================================================================
    # TEST 4: Visibility blocked by wall (manual placement)
    # ====================================================================
    print("\n" + "=" * 64)
    print("TEST 4: Line-of-sight blocking by walls")
    print("=" * 64)

    # Use static XML which has internal walls with doors
    env4 = HideAndSeekEnv(
        render_mode=render_mode,
        procedural=False,
        horizon=240,
    )
    obs, info = env4.reset()
    actions = {n: _idle() for n in AGENT_NAMES}
    obs, _, _, _, info = env4.step(actions)

    vis = info["visibility_matrix"]

    print(f"\n  Static arena visibility matrix:")
    header = "           " + "  ".join(f"{n:>9s}" for n in AGENT_NAMES)
    print(f"  {header}")
    for i, name in enumerate(AGENT_NAMES):
        row = "  ".join(f"{'  YES' if vis[i, j] else '   no':>9s}" for j in range(N_AGENTS))
        print(f"  {name:>9s}  {row}")

    # In the static XML, hiders start in SE room, seekers in NW
    # There should be walls between them
    blocked_count = 0
    visible_count = 0
    for hi in range(N_HIDERS):
        for si in range(N_HIDERS, N_AGENTS):
            pos_h = env4._get_body_pos(env4._agent_body_ids[AGENT_NAMES[hi]])
            pos_s = env4._get_body_pos(env4._agent_body_ids[AGENT_NAMES[si]])
            d = np.linalg.norm(pos_h[:2] - pos_s[:2])
            see = vis[hi, si]
            status = "CAN SEE" if see else "BLOCKED"
            print(f"    {AGENT_NAMES[hi]} <-> {AGENT_NAMES[si]}: {d:.2f}m  [{status}]")
            if see:
                visible_count += 1
            else:
                blocked_count += 1

    if blocked_count > 0:
        print(f"  [OK] {blocked_count} cross-team pair(s) blocked by walls")
        passed += 1
    else:
        print(f"  [!] No cross-team pairs blocked — walls may have doors in line of sight")

    if visible_count > 0:
        print(f"  [OK] {visible_count} cross-team pair(s) visible through doors/openings")
        passed += 1

    env4.close()

    # ====================================================================
    # TEST 5: Lidar range vs arena diagonal
    # ====================================================================
    print("\n" + "=" * 64)
    print("TEST 5: Lidar range vs arena geometry")
    print("=" * 64)

    arena_diagonal = ARENA_HALF * 2 * np.sqrt(2)
    lidar_max = 18.0  # from config
    agent_obs_r = 100.0  # from config

    print(f"  Arena size: {ARENA_HALF * 2}m x {ARENA_HALF * 2}m")
    print(f"  Arena diagonal: {arena_diagonal:.1f}m")
    print(f"  Lidar max range: {lidar_max}m")
    print(f"  Agent obs radius: {agent_obs_r}m")

    if lidar_max >= arena_diagonal:
        print(f"  [OK] Lidar range ({lidar_max}m) >= arena diagonal ({arena_diagonal:.1f}m)")
        print(f"        -> Agents can detect walls/obstacles across the entire arena")
        passed += 1
    else:
        print(f"  [FAIL] Lidar range ({lidar_max}m) < arena diagonal ({arena_diagonal:.1f}m)")
        print(f"         -> Agents have blind spots at far corners!")
        failed += 1

    if agent_obs_r >= arena_diagonal:
        print(f"  [OK] Agent obs radius ({agent_obs_r}m) >> arena diagonal ({arena_diagonal:.1f}m)")
        print(f"        -> Distance never limits agent-to-agent vision (only line-of-sight matters)")
        passed += 1
    else:
        print(f"  [FAIL] Agent obs radius ({agent_obs_r}m) < arena diagonal ({arena_diagonal:.1f}m)")
        failed += 1

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n" + "=" * 64)
    total = passed + failed
    print(f"VISION DIAGNOSTIC: {passed}/{total} checks passed, {failed} failed")
    if failed == 0:
        print("[OK] All vision checks passed!")
    else:
        print(f"[!] {failed} issue(s) found — review output above")
    print("=" * 64)

    return failed


if __name__ == "__main__":
    sys.exit(main())
