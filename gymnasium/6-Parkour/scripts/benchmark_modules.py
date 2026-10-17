"""Benchmark environment throughput on each experimental module, and its rendering cost.

Physics cases step a single env with random actions and report control steps per second:

* ``procedural``: an ordinary Procedural course, the baseline;
* one case per experimental module (``water_tank``, ``spinning_barrel``, ...), a one-module
  course with the robot put back at the module's start after every reset, so the module's
  physics and actors run on every step;
* ``water_tank, physics off``: the tank with ``water_physics`` off, to price the water forces.

``--render`` adds offscreen ``rgb_array`` frames per second for the same cases, each in a
process of its own: on macOS, offscreen renderers made after the first can return black frames.

Example::

    python scripts/benchmark_modules.py --steps 3000 --render
    python scripts/benchmark_modules.py --skills water_tank sliding_elevator
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from g1_parkour import tasks  # noqa: E402
from g1_parkour.env import ParkourEnv  # noqa: E402
from g1_parkour.terrain.parkour import EXPERIMENTAL_SKILLS  # noqa: E402


def make_env(skill: str | None, water_physics: bool = True, render_mode: str | None = None,
             difficulty: float = 0.5) -> ParkourEnv:
    cfg = tasks.parkour_cfg()
    cfg.terrain.curriculum = False
    cfg.terrain.difficulty = difficulty
    if skill is not None:
        cfg.terrain.params = {"num_modules": 1, "skills": (skill,)}
    cfg.water_physics = water_physics
    return ParkourEnv(cfg, render_mode=render_mode, width=640, height=360)


def reset(env: ParkourEnv, seed: int, on_module: bool) -> None:
    """Reset, then stand the robot at the module's start (in the water, for a tank)."""
    env.reset(seed=seed)
    if not on_module:
        return
    if env.terrain.water:
        water = env.terrain.water[0]
        env.data.qpos[0] = water.center[0] - water.half_size[0] + 1.2
    else:
        env.data.qpos[0] = env.terrain.waypoints[0][0] + 0.3  # the start pad's end
    env.data.qpos[2] += 2.0
    mujoco.mj_forward(env.model, env.data)
    env._settle_feet_on_ground()


def steps_per_second(env: ParkourEnv, steps: int, on_module: bool, render: bool = False) -> float:
    rng = np.random.default_rng(0)
    episode = 0
    reset(env, episode, on_module)
    start = time.perf_counter()
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(rng.uniform(-0.3, 0.3, env.action_space.shape))
        if render:
            env.render()
        if terminated or truncated:
            episode += 1
            reset(env, episode, on_module)
    return steps / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--repeats", type=int, default=2, help="take the best of this many runs per case")
    parser.add_argument("--skills", nargs="+", default=list(EXPERIMENTAL_SKILLS))
    parser.add_argument("--render", action="store_true", help="also time offscreen rgb_array rendering")
    parser.add_argument("--render-one", default=None, metavar="CASE", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.render_one:
        render_one(None if args.render_one == "procedural" else args.render_one, max(200, args.steps // 10))
        return

    cases = [("procedural", None, {}, False)]
    cases += [(skill, skill, {}, True) for skill in args.skills]
    if "water_tank" in args.skills:
        cases.append(("water_tank, physics off", "water_tank", {"water_physics": False}, True))

    baseline = None
    print(f"{'case':26s} {'steps/s':>8s} {'vs procedural':>14s}")
    for name, skill, kwargs, on_module in cases:
        env = make_env(skill, **kwargs)
        best = max(steps_per_second(env, args.steps, on_module) for _ in range(args.repeats))
        env.close()
        baseline = baseline or best
        print(f"{name:26s} {best:8.0f} {100 * (best / baseline - 1):+13.0f}%")

    if args.render:
        print(f"\n{'rendering (640x360)':26s} {'frames/s':>8s}", flush=True)
        for case in ["procedural", *args.skills]:
            subprocess.run([sys.executable, __file__, "--render-one", case, "--steps", str(args.steps)], check=True)


def render_one(skill: str | None, steps: int) -> None:
    env = make_env(skill, render_mode="rgb_array")
    rate = steps_per_second(env, steps, skill is not None, render=True)
    env.close()
    print(f"{skill or 'procedural':26s} {rate:8.0f}")


if __name__ == "__main__":
    main()
