"""Replay a policy (or random/zero actions) on a parkour task in the MuJoCo viewer.

The viewer opens as a borderless full-screen window by default; pass --windowed to get a normal window.

Examples::

    python scripts/play.py Flat                 # random actions
    python scripts/play.py Rough                # random actions
    python scripts/play.py Procedural           # random actions

    python scripts/play.py Procedural --checkpoint models/<run>/final.zip
    python scripts/play.py Procedural --checkpoint models/<run>/final.zip --difficulty 0.3
    python scripts/play.py Procedural --skills water_tank --difficulty 1   # one module type only

Playback fixes the terrain difficulty with the curriculum off: ``--difficulty`` for every
module if given, otherwise the levels saved with the checkpoint (``.state.json``; each skill's
mean over the training workers for a per-skill run, so courses look as they did in training),
otherwise the task's starting difficulty. A skill the checkpoint never trained, such as a newly
added module, starts at the task's starting difficulty.

Procedural courses draw from every module, experimental ones included, each equally likely, so
evaluation sees them all; ``--training-mix`` draws only the training skills, also each equally
likely, and ``--skills`` names the modules.
"""

from __future__ import annotations

import argparse
import json
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


def load_obs_normalizer(checkpoint: str):
    """Return ``obs -> normalised obs`` using the ``.vecnormalize.pkl`` saved with a checkpoint."""
    path = Path(checkpoint)
    stats = path.with_name(path.stem + ".vecnormalize.pkl")
    if not stats.exists():
        return lambda obs: obs
    import pickle

    with stats.open("rb") as handle:
        normalizer = pickle.load(handle)
    normalizer.training = False
    print(f"using observation normalisation from {stats}")
    return normalizer.normalize_obs


def saved_difficulty(checkpoint, step: float) -> float | dict[str, float] | None:
    """Mean worker difficulty saved with ``checkpoint``, snapped to the curriculum grid: one
    level, or each skill's for a per-skill run."""
    path = Path(checkpoint).with_suffix(".state.json")
    if not path.exists():
        return None
    from g1_parkour.env import mean_curriculum_state

    return mean_curriculum_state(json.loads(path.read_text())["difficulties"], step)


def configure_difficulty(cfg, requested: float | None, checkpoint: str | None) -> str:
    """Fix ``cfg``'s terrain difficulty for playback, turn the curriculum off, and name the source."""
    terrain = cfg.terrain
    saved = saved_difficulty(checkpoint, terrain.difficulty_step) if checkpoint and requested is None else None
    if requested is not None:
        level, source = requested, "--difficulty"
    elif saved is not None:
        level, source = saved, f"saved with {Path(checkpoint).name}"
    else:
        level, source = terrain.difficulty, "task default"
    low, high = terrain.difficulty_range
    if isinstance(level, dict):
        terrain.skill_difficulty = level  # skills it leaves out start at the task's level
    else:
        terrain.difficulty = float(np.clip(level, low, high))
        terrain.skill_difficulty = None
    terrain.curriculum = False
    return source


def configure_skills(cfg, skills: list[str] | None, training_mix: bool) -> str:
    """Pick the modules playback courses draw from, each equally likely, and say which: ``skills``
    if named, the training skills with ``training_mix``, otherwise every module (experimental
    ones included). Only parkour courses have modules."""
    from g1_parkour.terrain.parkour import MODULES

    if cfg.terrain.kind != "parkour":
        return ""
    if skills:
        cfg.terrain.params = {**cfg.terrain.params, "skills": tuple(skills)}
        return f"modules: {', '.join(skills)}"
    if training_mix:
        chosen = tasks.equal_module_mix(cfg)
        return f"modules: the {len(chosen)} training skills, equally likely"
    tasks.equal_module_mix(cfg, tuple(MODULES))
    return f"modules: all {len(MODULES)}, equally likely (--training-mix for the training skills only)"


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
    parser.add_argument("--camera", default="side_view",
                        help="side_view, chase_view, front_view or egocentric (first person, steadied like a "
                             "head; TAB in the viewer also cycles through them)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--speed", type=float, default=0.5,
                        help="playback speed as a fraction of real time (default 0.5); 1.0 runs as fast as the sim allows")
    parser.add_argument("--until-fall", action=argparse.BooleanOptionalAction, default=False,
                        help="remove the time limit and stall termination so episodes only end on a fall, going out of bounds, or reaching the goal (default; use --no-until-fall to restore the task's limits)")
    parser.add_argument("--soft-termination", action=argparse.BooleanOptionalAction, default=False,
                        help="relax the fall thresholds (deeper crouch, more tilt, deeper pits, wider bounds) so stumbles get a chance to recover (default; use --no-soft-termination for the task's strict limits)")
    parser.add_argument("--windowed", action="store_true", help="open the viewer in a window instead of fullscreen")
    parser.add_argument("task", nargs="?", default="Procedural")
    parser.add_argument("--algo", default="PPO")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--viewer", choices=["human", "none"], default="human")
    parser.add_argument("--difficulty", type=float, default=None,
                        help="fixed terrain difficulty in [0, 1]; by default the level saved with --checkpoint, "
                             "else the task's starting difficulty")
    parser.add_argument("--skills", nargs="+", default=None,
                        help="Procedural courses built from only these modules, e.g. water_tank, "
                             "which training does not use yet, each at its saved level or --difficulty")
    parser.add_argument("--training-mix", action="store_true",
                        help="Procedural courses from the training skills only, each equally often; by default "
                             "playback draws every module, experimental ones included, equally often")
    args = parser.parse_args()
    if args.difficulty is not None and not 0.0 <= args.difficulty <= 1.0:
        parser.error("difficulty must be in [0, 1]")

    cfg = tasks.TASKS[args.task]()
    source = configure_difficulty(cfg, args.difficulty, args.checkpoint)
    mix = configure_skills(cfg, args.skills, args.training_mix)
    if mix:
        print(mix)
    if cfg.terrain.skill_difficulty:
        levels = ", ".join(f"{skill} {level:.2f}" for skill, level in cfg.terrain.skill_difficulty.items())
        print(f"per-skill difficulty from {source}: {levels}; curriculum off for playback")
    else:
        print(f"difficulty {cfg.terrain.difficulty:.2f} from {source}; curriculum off for playback")
    env = gym.make(
        args.task,
        cfg=cfg,
        render_mode=None if args.viewer == "none" else "human",
        camera_name=args.camera,
    )
    policy = load_policy(args.checkpoint, args.algo) if args.checkpoint else None
    normalize = load_obs_normalizer(args.checkpoint) if args.checkpoint else (lambda obs: obs)

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

        # The viewer window only exists after the first render; the env keeps it across resets.
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
                action, _ = policy.predict(normalize(obs), deterministic=True)
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