"""MDP ablation harness.

Two modes:

``train`` sweeps a set of ablations, training each with the same budget/seed and
writing a summary CSV::

    python scripts/ablate.py train --ablations blind noisy-obs sparse-reward \\
        --timesteps 2_000_000 --num-envs 8

``eval`` takes one trained checkpoint and scores it under every ablation env, which
isolates which MDP ingredient the policy actually depends on::

    python scripts/ablate.py eval --checkpoint models/<run>/final.zip --episodes 20
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

from g1_parkour import tasks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def ablation_task_id(name: str) -> str:
    slug = "".join(part.capitalize() for part in name.split("-"))
    return f"Parkour-Ablation-{slug}-v0"


def rollout(task_id: str, policy, episodes: int, seed: int) -> dict[str, float] | None:
    env = gym.make(task_id)
    if policy is not None and policy.observation_space.shape != env.observation_space.shape:
        # Observation-space ablations change the obs size, so a checkpoint only
        # transfers to envs that keep the same terms it was trained with.
        print(
            f"  skipped: policy expects {policy.observation_space.shape}, "
            f"{task_id} provides {env.observation_space.shape}"
        )
        env.close()
        return None
    returns, completions, lengths, distances = [], [], [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total, steps = 0.0, 0
        while True:
            if policy is None:
                action = env.action_space.sample()
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            if terminated or truncated:
                returns.append(total)
                completions.append(info["course_completion"])
                distances.append(info["x_position"])
                lengths.append(steps)
                break
    env.close()
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "completion_mean": float(np.mean(completions)),
        "distance_mean": float(np.mean(distances)),
        "length_mean": float(np.mean(lengths)),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")


def cmd_eval(args) -> None:
    policy = None
    if args.checkpoint:
        from stable_baselines3 import PPO

        policy = PPO.load(args.checkpoint)

    names = args.ablations or list(tasks.ABLATIONS)
    rows = []
    for name in ["baseline", *names]:
        task_id = "Parkour-Procedural-v0" if name == "baseline" else ablation_task_id(name)
        print(f"{name:24s}")
        metrics = rollout(task_id, policy, args.episodes, args.seed)
        if metrics is None:
            continue
        rows.append({"ablation": name, "task": task_id, **metrics})
        print(
            f"  return={metrics['return_mean']:9.2f} "
            f"completion={metrics['completion_mean']:.0%} x={metrics['distance_mean']:.1f}"
        )
    if not rows:
        raise SystemExit("no ablation env matched the checkpoint's observation space")
    write_csv(rows, RESULTS_DIR / f"ablation_eval_{datetime.now():%Y%m%d_%H%M%S}.csv")


def cmd_train(args) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecMonitor

    names = args.ablations or list(tasks.ABLATIONS)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    rows = []
    for name in ["baseline", *names]:
        task_id = "Parkour-Procedural-v0" if name == "baseline" else ablation_task_id(name)
        print(f"\n=== training {name} ({task_id}) ===")
        env = VecMonitor(make_vec_env(task_id, n_envs=args.num_envs, seed=args.seed))
        model = PPO("MlpPolicy", env, verbose=0, seed=args.seed, tensorboard_log=str(ROOT / "logs"))
        model.learn(total_timesteps=args.timesteps, tb_log_name=f"ablate_{stamp}_{name}")
        out_dir = ROOT / "models" / f"ablate_{stamp}" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(out_dir / "final"))
        env.close()

        metrics = rollout(task_id, model, args.episodes, args.seed + 10_000)
        if metrics is None:
            continue
        rows.append({"ablation": name, "task": task_id, **metrics})
        print(f"{name:24s} return={metrics['return_mean']:9.2f} completion={metrics['completion_mean']:.0%}")
    write_csv(rows, RESULTS_DIR / f"ablation_train_{stamp}.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ablations", nargs="*", default=None, choices=list(tasks.ABLATIONS))
    common.add_argument("--episodes", type=int, default=10)
    common.add_argument("--seed", type=int, default=0)

    eval_parser = sub.add_parser("eval", parents=[common])
    eval_parser.add_argument("--checkpoint", default=None)
    eval_parser.set_defaults(func=cmd_eval)

    train_parser = sub.add_parser("train", parents=[common])
    train_parser.add_argument("--timesteps", type=int, default=2_000_000)
    train_parser.add_argument("--num-envs", type=int, default=8)
    train_parser.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
