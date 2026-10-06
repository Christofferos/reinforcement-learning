"""Train a policy on any registered parkour task with Stable-Baselines3.

Examples::

    python scripts/train.py Parkour-Flat-v0 --num-envs 8 --timesteps 5_000_000
    python scripts/train.py Parkour-Procedural-v0 --algo PPO --num-envs 16
    python scripts/train.py Parkour-Procedural-v0 --resume models/.../final.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_parkour import tasks  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
TERMINATION_REASONS = (
    "goal", "bad_height", "bad_orientation", "fell_off_course",
    "out_of_bounds", "stall", "timeout",
)


def evaluate_progress(model, env, episodes: int = 20) -> dict:
    completions, durations, rewards = [], [], []
    velocity_errors = []
    endings = Counter()
    for seed in range(episodes):
        obs, _ = env.reset(seed=seed)
        reward_sum = 0.0
        for step in range(env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            if terminated or truncated:
                break
        endings[info["termination"] or "timeout"] += 1
        completions.append(info["course_completion"])
        durations.append((step + 1) * env.dt)
        rewards.append(reward_sum)
        if "velocity_tracking_error" in info:
            velocity_errors.append(info["velocity_tracking_error"])
    stats = {
        "success_rate": endings["goal"] / episodes,
        "completion_mean": float(np.mean(completions)),
        "duration_s_mean": float(np.mean(durations)),
        "reward_mean": float(np.mean(rewards)),
        "difficulty": env.cfg.terrain.difficulty,
        "endings": dict(endings),
    }
    if velocity_errors:
        stats["velocity_tracking_error_mean"] = float(np.mean(velocity_errors))
    return stats


class ProgressCallback(BaseCallback):
    def __init__(self, model_dir: Path, eval_env, eval_every: int, hours: float | None = None):
        super().__init__()
        self.model_dir = model_dir
        self.eval_env = eval_env
        self.eval_every = eval_every
        self.episodes = deque(maxlen=100)
        self.last_eval_step = 0
        self.hours = hours
        self.deadline = None

    def save_state(self, name: str) -> None:
        state = {
            "num_timesteps": self.model.num_timesteps,
            "difficulties": self.training_env.get_attr("_difficulty"),
        }
        (self.model_dir / f"{name}.state.json").write_text(json.dumps(state, indent=2))

    def _on_training_start(self) -> None:
        self.last_eval_step = self.model.num_timesteps
        if self.hours is not None:
            self.deadline = time.monotonic() + self.hours * 3600.0
        self._evaluate()

    def _on_training_end(self) -> None:
        self._evaluate()

    def _evaluate(self) -> None:
        stats = evaluate_progress(self.model, self.eval_env)
        stats["num_timesteps"] = self.model.num_timesteps
        with (self.model_dir / "evaluations.jsonl").open("a") as output:
            output.write(json.dumps(stats) + "\n")
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                self.logger.record(f"eval/{key}", value)
        for reason in TERMINATION_REASONS:
            self.logger.record(f"eval/end_{reason}", stats["endings"].get(reason, 0) / 20)
        print(f"fixed-course evaluation: {stats}", flush=True)

    def _on_step(self) -> bool:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            print(f"wall-clock limit of {self.hours:g} hours reached; saving final checkpoint", flush=True)
            return False
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.episodes.append(info)
        if self.model.num_timesteps - self.last_eval_step >= self.eval_every:
            self.last_eval_step = self.model.num_timesteps
            self._evaluate()
        return True

    def _on_rollout_end(self) -> None:
        if not self.episodes:
            return
        for key in ("course_completion", "difficulty", "is_success"):
            self.logger.record(f"progress/{key}", float(np.mean([info[key] for info in self.episodes])))
        velocity_errors = [info["velocity_tracking_error"] for info in self.episodes
                           if "velocity_tracking_error" in info]
        if velocity_errors:
            self.logger.record("progress/velocity_tracking_error", float(np.mean(velocity_errors)))
        for reason in TERMINATION_REASONS:
            self.logger.record(f"termination/{reason}", sum(info["termination"] == reason for info in self.episodes) / len(self.episodes))
        term_names = {name for info in self.episodes for name in info.get("episode_reward_terms", {})}
        for name in sorted(term_names):
            self.logger.record(f"reward_terms/{name}", float(np.mean([
                info.get("episode_reward_terms", {}).get(name, 0.0) for info in self.episodes
            ])))


def build_algo(name: str):
    from stable_baselines3 import PPO, SAC

    algos = {"PPO": PPO, "SAC": SAC}
    try:
        from sb3_contrib import TQC

        algos["TQC"] = TQC
    except ImportError:
        pass
    if name not in algos:
        raise SystemExit(f"Algorithm '{name}' unavailable. Installed: {sorted(algos)}")
    return algos[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", help="registered task id, see scripts/list_envs.py")
    parser.add_argument("--algo", default="PPO")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=20_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=None, help="path to a .zip checkpoint")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--subproc", action="store_true", help="use SubprocVecEnv workers")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="PPO learning rate, also applied on resume")
    parser.add_argument("--gamma", type=float, default=0.999, help="PPO discount factor")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO minibatch size")
    parser.add_argument("--n-epochs", type=int, default=5, help="PPO update epochs")
    parser.add_argument("--target-kl", type=float, default=0.02, help="PPO early-stop KL threshold")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="PPO entropy regularization weight")
    parser.add_argument("--policy-hidden-sizes", type=int, nargs="+", default=None, metavar="WIDTH",
                        help="PPO hidden layer widths for a fresh policy; cannot be changed on resume")
    parser.add_argument("--eval-every", type=int, default=1_000_000)
    parser.add_argument("--eval-difficulty", type=float, default=0.45)
    parser.add_argument("--hours", type=float, default=None,
                        help="stop after this many wall-clock hours and save final.zip; timesteps is still an upper limit")
    args = parser.parse_args()
    if args.num_envs < 1 or args.timesteps < 1 or args.eval_every < 1:
        parser.error("environment count and step intervals must be positive")
    if not 0 < args.gamma < 1 or args.learning_rate <= 0 or args.target_kl <= 0:
        parser.error("gamma must be between 0 and 1; learning rate and target KL must be positive")
    if args.batch_size < 2 or args.n_epochs < 1 or not 0 <= args.eval_difficulty <= 1:
        parser.error("invalid PPO batch size, epoch count, or evaluation difficulty")
    if not np.isfinite(args.ent_coef) or args.ent_coef < 0:
        parser.error("entropy coefficient must be finite and non-negative")
    if args.policy_hidden_sizes is not None:
        if args.algo != "PPO" or args.resume or any(width < 1 for width in args.policy_hidden_sizes):
            parser.error("policy hidden sizes require a fresh PPO policy and positive widths")
    if args.hours is not None and (not np.isfinite(args.hours) or args.hours <= 0):
        parser.error("hours must be a finite positive number")

    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from g1_parkour.env import ParkourEnv

    run_name = args.run_name or f"{args.task}_{datetime.now():%Y%m%d_%H%M%S}"
    model_dir = ROOT / "models" / run_name
    log_dir = ROOT / "logs"
    model_dir.mkdir(parents=True, exist_ok=False)

    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    env = make_vec_env(
        args.task,
        n_envs=args.num_envs,
        seed=args.seed,
        vec_env_cls=vec_cls,
        env_kwargs={},
    )
    eval_cfg = tasks.TASKS[args.task]()
    eval_cfg.terrain.curriculum = False
    eval_cfg.terrain.resample_every_n_resets = 1
    eval_cfg.terrain.difficulty = args.eval_difficulty
    eval_env = ParkourEnv(eval_cfg)
    config = {"arguments": vars(args), "environment": asdict(env.get_attr("cfg")[0])}
    (model_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=lambda value: value.tolist()))

    algo = build_algo(args.algo)
    options = {}
    if args.algo == "PPO":
        options = dict(learning_rate=args.learning_rate, gamma=args.gamma,
                       batch_size=args.batch_size, n_epochs=args.n_epochs, target_kl=args.target_kl,
                       ent_coef=args.ent_coef)
        if args.policy_hidden_sizes is not None:
            options["policy_kwargs"] = {"net_arch": args.policy_hidden_sizes}
    if args.resume:
        model = algo.load(args.resume, env=env, tensorboard_log=str(log_dir), **options)
        model.ep_info_buffer = deque(maxlen=model._stats_window_size)
        model.ep_success_buffer = deque(maxlen=model._stats_window_size)
        print(f"resumed from {args.resume}")
        state_path = Path(args.resume).with_suffix(".state.json")
        if state_path.exists():
            state = json.loads(state_path.read_text())
            difficulties = state["difficulties"]
            if len(difficulties) != args.num_envs:
                raise ValueError("resume difficulty state requires the same number of environments")
            for index, difficulty in enumerate(difficulties):
                env.set_attr("_difficulty", difficulty, indices=index)
            print(f"restored terrain difficulties: {difficulties}")
        else:
            print("no difficulty state in this older checkpoint; using the task's initial difficulty")
    else:
        model = algo("MlpPolicy", env, verbose=1, seed=args.seed, tensorboard_log=str(log_dir), **options)
    print(f"training options: {options}")

    metrics = ProgressCallback(model_dir, eval_env, args.eval_every, args.hours)
    try:
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=args.resume is None,
            callback=metrics,
            tb_log_name=run_name,
        )
        model.save(str(model_dir / "final"))
        metrics.save_state("final")
        print(f"saved to {model_dir / 'final.zip'}")
    except KeyboardInterrupt:
        model.save(str(model_dir / "interrupted"))
        metrics.save_state("interrupted")
        print(f"\ninterrupted at {model.num_timesteps} steps, saved to {model_dir / 'interrupted.zip'}")
        raise SystemExit(130)  # non-zero so run_curriculum.sh stops instead of starting the next stage
    finally:
        eval_env.close()
        env.close()


if __name__ == "__main__":
    main()