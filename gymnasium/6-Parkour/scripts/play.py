"""Replay a policy (or random/zero actions) on a parkour task in the MuJoCo viewer.

The viewer opens as a borderless full-screen window by default; pass --windowed to get a normal window.

Examples::

    python scripts/play.py Parkour-Procedural-Play-v0            # random actions
    python scripts/play.py Parkour-Procedural-v0 --checkpoint models/<run>/final.zip
    python scripts/play.py Parkour-Rough-v0 --episodes 3 --viewer none
    python scripts/play.py Parkour-Rough-v0 --windowed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

from g1_parkour import tasks  # noqa: E402,F401


def load_policy(checkpoint: str, algo: str):
    from stable_baselines3 import PPO, SAC

    algos = {"PPO": PPO, "SAC": SAC}
    try:
        from sb3_contrib import TQC

        algos["TQC"] = TQC
    except ImportError:
        pass
    return algos[algo].load(checkpoint)


def make_fullscreen(env) -> bool:
    """Make the MuJoCo viewer a borderless window covering the primary monitor.

    Works with Gymnasium's MujocoRenderer / WindowViewer, which creates the window on the
    first render() call. Returns False if no GLFW window could be found.
    """
    try:
        import glfw
    except ImportError:
        return False

    base = env.unwrapped
    # g1_parkour's env stores it as _mujoco_renderer; Gymnasium's MujocoEnv uses mujoco_renderer.
    renderer = getattr(base, "_mujoco_renderer", None) or getattr(base, "mujoco_renderer", None)
    viewer = getattr(renderer, "viewer", None)
    window = getattr(viewer, "window", None)
    if window is None:
        return False

    # Borderless "windowed fullscreen": remove the title bar and stretch the window over the
    # monitor's work area (the screen minus the menu bar/Dock or taskbar). The window stays a
    # normal window, so Cmd/Alt+Tab and other apps keep working.
    monitor = glfw.get_primary_monitor()
    x, y, width, height = glfw.get_monitor_workarea(monitor)
    glfw.set_window_attrib(window, glfw.DECORATED, glfw.FALSE)
    glfw.set_window_pos(window, x, y)
    glfw.set_window_size(window, width, height)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", nargs="?", default="Procedural")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--algo", default="PPO")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viewer", choices=["human", "none"], default="human")
    parser.add_argument("--camera", default="side_view", help="e.g. side_view, chase_view, egocentric")
    parser.add_argument("--windowed", action="store_true", help="open the viewer in a window instead of fullscreen")
    parser.add_argument("--overhead-observation", action=argparse.BooleanOptionalAction, default=True,
                        help="feed the three upward-ray distances to the policy (default); use --no-overhead-observation to only visualize them with older checkpoints")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="playback speed as a fraction of real time (default 0.5); 1.0 runs as fast as the sim allows")
    parser.add_argument("--until-fall", action=argparse.BooleanOptionalAction, default=True,
                        help="remove the time limit and stall termination so episodes only end on a fall, going out of bounds, or reaching the goal (default; use --no-until-fall to restore the task's limits)")
    parser.add_argument("--soft-termination", action=argparse.BooleanOptionalAction, default=True,
                        help="relax the fall thresholds (deeper crouch, more tilt, deeper pits, wider bounds) so stumbles get a chance to recover (default; use --no-soft-termination for the task's strict limits)")
    args = parser.parse_args()

    cfg = tasks.TASKS[args.task]()
    cfg.observation.overhead_scan_observation = args.overhead_observation
    env = gym.make(
        args.task,
        cfg=cfg,
        render_mode=None if args.viewer == "none" else "human",
        camera_name=args.camera,
    )
    policy = load_policy(args.checkpoint, args.algo) if args.checkpoint else None

    if args.until_fall:
        base = env.unwrapped
        base.cfg.episode_length_s = 1e9  # max_episode_steps becomes ~6.7e10, never reached
        base.cfg.termination.stall = False

    if args.soft_termination:
        term = env.unwrapped.cfg.termination
        term.bad_height_scale = 0.5   # torso may drop to ~0.4 m above the terrain (stumble/crouch)
        term.orientation_limit = 0.1  # allow tilting to ~84 deg before calling it a fall
        term.pit_margin = 2.0         # must fall 2 m below the lowest waypoint, not 1 m
        term.lateral_limit = 12.0
        term.backward_limit = 8.0

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)

        # The viewer window only exists after a render, and the env can close and recreate
        # its renderer on reset, so (re)apply fullscreen at the start of every episode.
        if args.viewer == "human" and not args.windowed:
            env.render()
            # Enable fullscreen without windowed mode.
            """ if not make_fullscreen(env) and episode == 0:
                print("warning: couldn't find the viewer window, staying windowed") """

        total, steps = 0.0, 0
        step_dt = env.unwrapped.dt
        next_step = time.perf_counter()
        print(f"episode {episode}: modules = {' -> '.join(info['terrain_modules'])}")
        while True:
            if policy is None:
                action = np.zeros(env.action_space.shape)
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if args.viewer == "human":
                # Pace playback to args.speed * real time (1 step = dt of sim time).
                next_step += step_dt / max(args.speed, 1e-6)
                delay = next_step - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_step = time.perf_counter()  # sim is slower than target; don't accumulate lag
            total += reward
            steps += 1
            if terminated or truncated:
                print(
                    f"  steps={steps} reward={total:8.1f} "
                    f"completion={info['course_completion']:.0%} "
                    f"end={info['termination'] or 'timeout'} x={info['x_position']:.1f}"
                )
                break
    env.close()


if __name__ == "__main__":
    main()