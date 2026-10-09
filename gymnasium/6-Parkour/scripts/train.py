"""


New Training::

caffeinate -i python scripts/train.py Flat --run-name "flat_0.0.1" --hours 1




Resume Training::

caffeinate -i python scripts/train.py Procedural --resume models/.../final.zip --hours 1


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
from stable_baselines3.common.logger import configure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_parkour import tasks  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
TERMINATION_REASONS = (
    "bad_orientation", "bad_height", "stall", "goal", "timeout", "fell_off_course",
    "out_of_bounds",
)


def evaluate_progress(model, env, episodes: int = 20) -> dict:
    completions, durations, rewards = [], [], []
    velocity_errors = []
    endings = Counter()
    normalizer = getattr(model, "get_vec_normalize_env", lambda: None)()
    for seed in range(episodes):
        obs, _ = env.reset(seed=seed)
        reward_sum = 0.0
        for step in range(env.max_episode_steps):
            policy_obs = normalizer.normalize_obs(obs) if normalizer is not None else obs
            action, _ = model.predict(policy_obs, deterministic=True)
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
        "difficulty": env.terrain.difficulty,
        "endings": dict(endings),
    }
    if velocity_errors:
        stats["velocity_tracking_error_mean"] = float(np.mean(velocity_errors))
    return stats


def normalizer_path(checkpoint) -> Path:
    """Observation/reward running statistics stored next to a ``.zip`` checkpoint."""
    checkpoint = Path(checkpoint)
    return checkpoint.with_name(checkpoint.stem + ".vecnormalize.pkl")


def state_path(checkpoint) -> Path:
    """Curriculum state (each worker's terrain difficulty) stored next to a ``.zip`` checkpoint."""
    return Path(checkpoint).with_suffix(".state.json")


def restore_difficulties(env, checkpoint, num_envs: int) -> list[float] | None:
    """Put each worker back at the difficulty saved with ``checkpoint``; None if no state file."""
    path = state_path(checkpoint)
    if not path.exists():
        return None
    difficulties = json.loads(path.read_text())["difficulties"]
    if len(difficulties) != num_envs:
        raise ValueError("resume difficulty state requires the same number of environments")
    for index, difficulty in enumerate(difficulties):
        # Go through the environment's own method: set_attr would only set the attribute on the
        # outer Monitor wrapper, leaving the real environment at its initial difficulty and
        # hiding its later curriculum changes from get_attr.
        env.env_method("set_difficulty", difficulty, indices=index)
    return difficulties


def save_checkpoint(model, model_dir: Path, name: str) -> None:
    model.save(str(model_dir / name))
    normalizer = getattr(model, "get_vec_normalize_env", lambda: None)()
    if normalizer is not None:
        normalizer.save(str(normalizer_path(model_dir / f"{name}.zip")))


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
        self.last_checkpoint_time = None

    def save_state(self, name: str) -> None:
        state = {
            "num_timesteps": self.model.num_timesteps,
            "difficulties": self.training_env.get_attr("_difficulty"),
        }
        (self.model_dir / f"{name}.state.json").write_text(json.dumps(state, indent=2))

    def _on_training_start(self) -> None:
        self.last_eval_step = self.model.num_timesteps
        self.last_checkpoint_time = time.monotonic()
        if self.hours is not None:
            self.deadline = self.last_checkpoint_time + self.hours * 3600.0
        self._evaluate()
        self.logger.dump(step=self.model.num_timesteps)

    def _on_training_end(self) -> None:
        self._evaluate()
        self.logger.dump(step=self.model.num_timesteps)

    def _evaluate(self) -> None:
        # Follow the workers so evaluation measures the current curriculum level, snapped
        # to the difficulty grid they train on.
        step = self.eval_env.cfg.terrain.difficulty_step
        mean_difficulty = float(np.mean(self.training_env.get_attr("_difficulty")))
        self.eval_env.set_difficulty(round(mean_difficulty / step) * step)
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
        if self.last_checkpoint_time is not None and time.monotonic() - self.last_checkpoint_time >= 600.0:
            name = f"checkpoint_{self.model.num_timesteps}_steps"
            save_checkpoint(self.model, self.model_dir, name)
            self.save_state(name)
            self.last_checkpoint_time = time.monotonic()
            print(f"saved periodic checkpoint to {self.model_dir / f'{name}.zip'}", flush=True)
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
        for key in ("avg_speed", "avg_height"):
            values = [info[key] for info in self.episodes if key in info]
            if values:
                self.logger.record(f"progress/{key}", float(np.mean(values)))
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
            per_step = [
                info.get("episode_reward_terms", {}).get(name, 0.0) / info["episode"]["l"]
                for info in self.episodes if info.get("episode", {}).get("l", 0) > 0
            ]
            if per_step:
                self.logger.record(f"reward_terms_per_step/{name}", float(np.mean(per_step)))


def build_algo(name: str):
    from stable_baselines3 import SAC
    from g1_parkour.symmetric_ppo import SymmetricPPO

    # SymmetricPPO is plain PPO when --symmetry none.
    algos = {"PPO": SymmetricPPO, "SAC": SAC}
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
    parser.add_argument("--timesteps", type=int, default=100_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=None, help="path to a .zip checkpoint")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default=str(ROOT / "logs"),
                        help="parent directory for per-run TensorBoard, CSV and JSON logs")
    parser.add_argument("--subproc", action="store_true", help="use SubprocVecEnv workers")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="PPO learning rate, also applied on resume")
    parser.add_argument("--gamma", type=float, default=0.999, help="PPO discount factor")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO minibatch size")
    parser.add_argument("--n-epochs", type=int, default=5, help="PPO update epochs")
    parser.add_argument("--target-kl", type=float, default=0.02, help="PPO early-stop KL threshold")
    parser.add_argument("--ent-coef", type=float, default=0.002, help="PPO entropy regularization weight")
    parser.add_argument("--policy-hidden-sizes", type=int, nargs="+", default=None, metavar="WIDTH",
                        help="PPO hidden layer widths for a fresh policy (default 256 256); cannot be changed on resume")
    parser.add_argument("--log-std-init", type=float, default=-1.0,
                        help="initial log standard deviation of a fresh PPO policy (exp(-1) ~ 0.37 of the action range)")
    parser.add_argument("--symmetry", choices=("loss", "augment", "both", "none"), default="both",
                        help="left/right mirror symmetry for PPO: a mirror loss on the policy mean (loss), "
                             "mirrored minibatch copies with a mirror-symmetric action std (augment), "
                             "both (default), or none")
    parser.add_argument("--symmetry-coef", type=float, default=1.0,
                        help="weight of the mirror loss for --symmetry loss/both")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="wrap training in VecNormalize (running observation/reward normalisation, saved with each checkpoint)")
    parser.add_argument("--eval-every", type=int, default=1_000_000)
    parser.add_argument("--hours", type=float, default=None,
                        help="stop after this many wall-clock hours and save final.zip; timesteps is still an upper limit")
    args = parser.parse_args()
    if args.num_envs < 1 or args.timesteps < 1 or args.eval_every < 1:
        parser.error("environment count and step intervals must be positive")
    if not 0 < args.gamma < 1 or args.learning_rate <= 0 or args.target_kl <= 0:
        parser.error("gamma must be between 0 and 1; learning rate and target KL must be positive")
    if args.batch_size < 2 or args.n_epochs < 1:
        parser.error("invalid PPO batch size or epoch count")
    if not np.isfinite(args.ent_coef) or args.ent_coef < 0:
        parser.error("entropy coefficient must be finite and non-negative")
    if args.policy_hidden_sizes is not None:
        if args.algo != "PPO" or args.resume or any(width < 1 for width in args.policy_hidden_sizes):
            parser.error("policy hidden sizes require a fresh PPO policy and positive widths")
    if args.hours is not None and (not np.isfinite(args.hours) or args.hours <= 0):
        parser.error("hours must be a finite positive number")
    if args.symmetry != "none" and args.algo != "PPO":
        parser.error("--symmetry needs PPO; pass --symmetry none for other algorithms")
    if not np.isfinite(args.symmetry_coef) or args.symmetry_coef < 0:
        parser.error("symmetry coefficient must be finite and non-negative")

    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from g1_parkour.env import ParkourEnv

    run_name = args.run_name or f"{args.task}_{datetime.now():%Y%m%d_%H%M%S}"
    model_dir = ROOT / "models" / run_name
    log_dir = Path(args.log_dir).expanduser().resolve()
    run_log_dir = log_dir / run_name
    model_dir.mkdir(parents=True, exist_ok=False)

    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    env = make_vec_env(
        args.task,
        n_envs=args.num_envs,
        seed=args.seed,
        vec_env_cls=vec_cls,
        env_kwargs={},
    )
    if args.normalize:
        stats_path = normalizer_path(args.resume) if args.resume else None
        if stats_path is not None and stats_path.exists():
            env = VecNormalize.load(str(stats_path), env)
            env.training = True
            print(f"restored normalisation statistics from {stats_path}")
        else:
            # Wide reward clip so the waypoint/goal bonuses survive return scaling.
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0,
                               clip_reward=100.0, gamma=args.gamma)
    eval_cfg = tasks.TASKS[args.task]()
    eval_cfg.terrain.curriculum = False
    eval_cfg.terrain.resample_every_n_resets = 1
    eval_env = ParkourEnv(eval_cfg)
    config = {"arguments": vars(args), "environment": asdict(env.get_attr("cfg")[0]),
              "log_directory": str(run_log_dir)}
    (model_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=lambda value: value.tolist()))

    algo = build_algo(args.algo)
    options = {}
    if args.algo == "PPO":
        options = dict(learning_rate=args.learning_rate, gamma=args.gamma,
                       batch_size=args.batch_size, n_epochs=args.n_epochs, target_kl=args.target_kl,
                       ent_coef=args.ent_coef, symmetry=args.symmetry, symmetry_coef=args.symmetry_coef,
                       mirror=(eval_env.observation_mirror(), eval_env.action_mirror())
                       if args.symmetry != "none" else None)
        if not args.resume:
            options["policy_kwargs"] = {
                "net_arch": args.policy_hidden_sizes or [256, 256],
                "log_std_init": args.log_std_init,
            }
    if args.resume:
        model = algo.load(args.resume, env=env, tensorboard_log=str(log_dir), **options)
        model.ep_info_buffer = deque(maxlen=model._stats_window_size)
        model.ep_success_buffer = deque(maxlen=model._stats_window_size)
        print(f"resumed from {args.resume}")
        difficulties = restore_difficulties(env, args.resume, args.num_envs)
        if difficulties is not None:
            print(f"restored terrain difficulties: {difficulties}")
        else:
            print("no difficulty state in this older checkpoint; using the task's initial difficulty")
    else:
        model = algo("MlpPolicy", env, verbose=1, seed=args.seed, tensorboard_log=str(log_dir), **options)
    model.set_logger(configure(str(run_log_dir), ["stdout", "csv", "json", "tensorboard"]))
    print(f"training options: { {key: value for key, value in options.items() if key != 'mirror'} }")
    print(f"monitor with TensorBoard: {sys.executable} -m tensorboard.main --logdir {log_dir}")

    metrics = ProgressCallback(model_dir, eval_env, args.eval_every, args.hours)
    try:
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=args.resume is None,
            callback=metrics,
            tb_log_name=run_name,
        )
        save_checkpoint(model, model_dir, "final")
        metrics.save_state("final")
        print(f"saved to {model_dir / 'final.zip'}")
    except KeyboardInterrupt:
        save_checkpoint(model, model_dir, "interrupted")
        metrics.save_state("interrupted")
        print(f"\ninterrupted at {model.num_timesteps} steps, saved to {model_dir / 'interrupted.zip'}")
        raise SystemExit(130)  # non-zero so all_curriculum.sh stops instead of starting the next stage
    finally:
        if model.logger.name_to_value:
            model.logger.dump(step=model.num_timesteps)
        model.logger.close()
        eval_env.close()
        env.close()


if __name__ == "__main__":
    main()