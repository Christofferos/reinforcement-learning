"""
Training script for Hide and Seek with MAPPO.
=============================================

Uses SyncVectorMultiAgentEnv for N parallel environments.

Usage:
    python scripts/train.py
    python scripts/train.py --render
    python scripts/train.py --n_episodes 10000 --device cuda
    python scripts/train.py --n_envs 16
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import platform
import ctypes
import numpy as np
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow imports from src/
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Ensure relative paths (runs/, models/, assets/) resolve from project root
os.chdir(_PROJECT_ROOT)


# ── Prevent Windows from sleeping during training ──
def _prevent_sleep():
    """Tell Windows not to sleep / turn off display while training."""
    if platform.system() == "Windows":
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002  # optional: keeps screen on too
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        print("[POWER] Windows sleep prevention ENABLED (SetThreadExecutionState)")


def _allow_sleep():
    """Re-allow Windows to sleep normally."""
    if platform.system() == "Windows":
        ES_CONTINUOUS = 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("[POWER] Windows sleep prevention DISABLED")

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from env import HIDER_NAMES, SEEKER_NAMES, AGENT_NAMES
from vec_env import SyncVectorMultiAgentEnv
from mappo import TeamPolicy
from config import ENV_CONFIG, MAPPO_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hide & Seek with MAPPO")

    # Environment
    parser.add_argument("--horizon", type=int, default=ENV_CONFIG["horizon"])
    parser.add_argument("--prep_fraction", type=float, default=ENV_CONFIG["prep_fraction"])
    parser.add_argument("--reward_type", type=str, default=ENV_CONFIG["reward_type"],
                        choices=["joint_zero_sum", "joint_mean", "selfish"])
    parser.add_argument("--n_envs", type=int, default=MAPPO_CONFIG["n_envs"],
                        help="Number of parallel environments")

    # Training
    parser.add_argument("--n_episodes", type=int, default=20_000,
                        help="Total episodes across all envs (rounded to multiples of n_envs)")
    parser.add_argument("--ppo_epochs", type=int, default=MAPPO_CONFIG["ppo_epochs"])
    parser.add_argument("--lr_actor", type=float, default=MAPPO_CONFIG["lr_actor"])
    parser.add_argument("--lr_critic", type=float, default=MAPPO_CONFIG["lr_critic"])
    parser.add_argument("--hidden_dim", type=int, default=MAPPO_CONFIG["hidden_dim"])
    parser.add_argument("--gamma", type=float, default=MAPPO_CONFIG["gamma"])
    parser.add_argument("--clip_epsilon", type=float, default=MAPPO_CONFIG["clip_epsilon"])
    parser.add_argument("--entropy_coef", type=float, default=MAPPO_CONFIG["entropy_coef"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Logging
    parser.add_argument("--log_interval", type=int, default=MAPPO_CONFIG["log_interval"])
    parser.add_argument("--save_interval", type=int, default=MAPPO_CONFIG["save_interval"])
    parser.add_argument("--log_dir", type=str, default=MAPPO_CONFIG["log_dir"])
    parser.add_argument("--save_dir", type=str, default=MAPPO_CONFIG["save_dir"])

    # Render
    parser.add_argument("--render", action="store_true", help="Render env 0 during training")

    # Procedural world generation
    parser.add_argument("--no_procedural", action="store_true",
                        help="Disable procedural arena generation (use static XML)")

    # Resume from a previous run
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a model dir to resume from (loads latest checkpoint)")

    return parser.parse_args()


def train(args):
    # ── Prevent OS sleep for the duration of training ──
    _prevent_sleep()

    # ── Setup ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"hideseek_mappo_{timestamp}"

    # If resuming, save into the same directory we're loading from
    if args.resume:
        save_path = args.resume
        log_path = os.path.join(args.log_dir, os.path.basename(args.resume) + "_resumed")
    else:
        log_path = os.path.join(args.log_dir, run_name)
        save_path = os.path.join(args.save_dir, run_name)

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    writer = SummaryWriter(log_path)

    n_envs = args.n_envs
    vec_env = SyncVectorMultiAgentEnv(
        n_envs=n_envs,
        horizon=args.horizon,
        prep_fraction=args.prep_fraction,
        reward_type=args.reward_type,
        render_mode="human" if args.render else None,
        procedural=not args.no_procedural,
    )

    # Get dimensions from a test reset
    all_obs, all_info = vec_env.reset()
    obs_dim = all_obs[0][HIDER_NAMES[0]].shape[0]
    act_dim = 3  # [move_x, move_y, grab]
    global_state_dim = all_info[0]["global_state"].shape[0]

    # How many "rollout rounds" (each round = 1 episode from each of n_envs)
    n_rounds = max(1, args.n_episodes // n_envs)
    total_episodes_planned = n_rounds * n_envs

    print(f"{'='*60}")
    print(f"Hide & Seek — MAPPO Training (Vectorized)")
    print(f"{'='*60}")
    print(f"  Obs dim:          {obs_dim}")
    print(f"  Action dim:       {act_dim}")
    print(f"  Global state dim: {global_state_dim}")
    print(f"  Horizon:          {args.horizon}")
    print(f"  Prep fraction:    {args.prep_fraction}")
    print(f"  Reward type:      {args.reward_type}")
    print(f"  Procedural:       {not args.no_procedural}")
    print(f"  Device:           {args.device}")
    print(f"  Parallel envs:    {n_envs}")
    print(f"  Rollout rounds:   {n_rounds}")
    print(f"  Total episodes:   ~{total_episodes_planned}")
    print(f"  Steps/round:      {args.horizon} x {n_envs} = {args.horizon * n_envs}")
    print(f"{'='*60}\n")

    # ── Build policy config ──
    policy_config = {
        "gamma": args.gamma,
        "gae_lambda": MAPPO_CONFIG["gae_lambda"],
        "clip_epsilon": args.clip_epsilon,
        "entropy_coef": args.entropy_coef,
        "value_coef": MAPPO_CONFIG["value_coef"],
        "max_grad_norm": MAPPO_CONFIG["max_grad_norm"],
        "ppo_epochs": args.ppo_epochs,
        "mini_batch_size": MAPPO_CONFIG["mini_batch_size"],
        "hidden_dim": args.hidden_dim,
        "n_layers": MAPPO_CONFIG["n_layers"],
        "use_feature_norm": MAPPO_CONFIG["use_feature_norm"],
        "lr_actor": args.lr_actor,
        "lr_critic": args.lr_critic,
    }

    # ── One shared policy per team ──
    # Buffer keys: (env_idx, agent_name) — so GAE is computed per-episode
    hider_buffer_keys = [(ei, name) for ei in range(n_envs) for name in HIDER_NAMES]
    seeker_buffer_keys = [(ei, name) for ei in range(n_envs) for name in SEEKER_NAMES]

    hider_policy = TeamPolicy("hiders", obs_dim, act_dim, global_state_dim,
                              policy_config, device=args.device)
    hider_policy.init_buffers(hider_buffer_keys)

    seeker_policy = TeamPolicy("seekers", obs_dim, act_dim, global_state_dim,
                               policy_config, device=args.device)
    seeker_policy.init_buffers(seeker_buffer_keys)

    # ── Optionally resume from a previous run ──
    start_round = 1
    total_steps = 0
    total_episodes = 0
    if args.resume:
        import glob
        # Find the latest checkpoint by episode number (sort numerically)
        hider_ckpts = glob.glob(os.path.join(args.resume, "hider_policy_ep*.pt"))
        seeker_ckpts = glob.glob(os.path.join(args.resume, "seeker_policy_ep*.pt"))

        def _ep_num(path):
            return int(os.path.basename(path).split("_ep")[1].split(".pt")[0])

        hider_ckpts = sorted(hider_ckpts, key=_ep_num)
        seeker_ckpts = sorted(seeker_ckpts, key=_ep_num)
        if hider_ckpts and seeker_ckpts:
            latest_hider = hider_ckpts[-1]
            latest_seeker = seeker_ckpts[-1]
            hider_policy.load(latest_hider)
            seeker_policy.load(latest_seeker)
            # Parse episode number from filename
            ep_num = int(os.path.basename(latest_hider).split("_ep")[1].split(".pt")[0])
            start_round = ep_num // n_envs + 1
            total_episodes = ep_num
            total_steps = ep_num * args.horizon
            print(f"  [>>] Resumed from {latest_hider}")
            print(f"     Starting at round {start_round}, episode {total_episodes}")
        else:
            print(f"  [!]  No checkpoints found in {args.resume}, starting fresh")

    def get_policy(agent_name):
        return hider_policy if agent_name in HIDER_NAMES else seeker_policy

    def buf_key(env_idx, agent_name):
        return (env_idx, agent_name)

    # ── Learning-rate schedulers (linear annealing to 0) ──
    hider_actor_scheduler = torch.optim.lr_scheduler.LinearLR(
        hider_policy.actor_optimizer, start_factor=1.0, end_factor=0.0,
        total_iters=n_rounds)
    hider_critic_scheduler = torch.optim.lr_scheduler.LinearLR(
        hider_policy.critic_optimizer, start_factor=1.0, end_factor=0.0,
        total_iters=n_rounds)
    seeker_actor_scheduler = torch.optim.lr_scheduler.LinearLR(
        seeker_policy.actor_optimizer, start_factor=1.0, end_factor=0.0,
        total_iters=n_rounds)
    seeker_critic_scheduler = torch.optim.lr_scheduler.LinearLR(
        seeker_policy.critic_optimizer, start_factor=1.0, end_factor=0.0,
        total_iters=n_rounds)

    # If resuming, fast-forward schedulers to the correct LR
    if start_round > 1:
        for _ in range(start_round - 1):
            hider_actor_scheduler.step()
            hider_critic_scheduler.step()
            seeker_actor_scheduler.step()
            seeker_critic_scheduler.step()
        print(f"     LR schedulers stepped to round {start_round - 1} "
              f"(lr={hider_actor_scheduler.get_last_lr()[0]:.2e})")

    # ── Training loop ──
    best_hider_reward = -float("inf")
    episode_rewards_hiders = []
    episode_rewards_seekers = []

    # ── CUDA error recovery helper ──
    last_checkpoint_round = 0
    cuda_retries = 0
    MAX_CUDA_RETRIES = 3  # give up after 3 consecutive CUDA errors

    def _recover_from_cuda_error(round_at_crash):
        """Reset CUDA state and reload weights from last checkpoint."""
        nonlocal cuda_retries
        cuda_retries += 1
        if cuda_retries > MAX_CUDA_RETRIES:
            print(f"\n[X] Too many consecutive CUDA errors ({MAX_CUDA_RETRIES}). Aborting.")
            raise RuntimeError("CUDA recovery failed after max retries")

        print(f"\n[!]  CUDA error at round {round_at_crash}. Attempting recovery "
              f"({cuda_retries}/{MAX_CUDA_RETRIES})...")

        # Reset CUDA state
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()

        # Find the latest checkpoint to reload
        ckpt_ep = last_checkpoint_round * n_envs  # episode number of last save
        hider_ckpt = os.path.join(save_path, f"hider_policy_ep{ckpt_ep}.pt")
        seeker_ckpt = os.path.join(save_path, f"seeker_policy_ep{ckpt_ep}.pt")

        if os.path.exists(hider_ckpt) and os.path.exists(seeker_ckpt):
            print(f"  [>>] Reloading checkpoint from episode {ckpt_ep} ...")
            hider_policy.load(hider_ckpt)
            seeker_policy.load(seeker_ckpt)
        else:
            print(f"  [!]  No checkpoint file found -- continuing with current weights")

        # Clear rollout buffers (the partially filled data is likely corrupted)
        hider_policy.init_buffers(hider_buffer_keys)
        seeker_policy.init_buffers(seeker_buffer_keys)
        print(f"  [OK] Recovery complete -- resuming from round {last_checkpoint_round + 1}\n")

    pbar = tqdm(range(start_round, n_rounds + 1), desc="Training", unit="round",
                initial=start_round - 1, total=n_rounds)

    for round_num in pbar:
      try:
        # Reset all envs at the start of each round
        all_obs, all_info = vec_env.reset()
        all_global_states = [info["global_state"] for info in all_info]

        # Per-env episode reward tracking
        ep_rewards = [
            {name: 0.0 for name in AGENT_NAMES}
            for _ in range(n_envs)
        ]

        # Run for exactly `horizon` steps (all envs in lockstep)
        for step in range(args.horizon):
            # Collect actions from all envs x all agents
            all_actions = [{} for _ in range(n_envs)]
            all_action_data = [{} for _ in range(n_envs)]

            for ei in range(n_envs):
                for name in AGENT_NAMES:
                    policy = get_policy(name)
                    action, log_prob, value = policy.select_action(
                        all_obs[ei][name], all_global_states[ei],
                        deterministic=False,
                    )
                    all_actions[ei][name] = action
                    all_action_data[ei][name] = (action, log_prob, value)

            # Step all envs
            next_all_obs, all_rewards, all_terminated, all_truncated, all_info = \
                vec_env.step(all_actions)
            next_global_states = [info["global_state"] for info in all_info]

            # Store transitions for each env x each agent
            for ei in range(n_envs):
                for name in AGENT_NAMES:
                    policy = get_policy(name)
                    act, lp, val = all_action_data[ei][name]
                    agent_done = (all_terminated[ei].get(name, False) or
                                  all_truncated[ei].get(name, False))
                    policy.store_transition(
                        buf_key(ei, name),
                        all_obs[ei][name],
                        all_global_states[ei],
                        act, lp,
                        all_rewards[ei][name],
                        agent_done, val,
                    )
                    ep_rewards[ei][name] += all_rewards[ei][name]

            all_obs = next_all_obs
            all_global_states = next_global_states
            total_steps += n_envs  # N envs stepped simultaneously

        # ── Round complete: all envs finished 1 episode ──
        total_episodes += n_envs

        # ── Update policies (with N x more data than before) ──
        hider_stats = hider_policy.update()
        seeker_stats = seeker_policy.update()

        # If we got here, this round succeeded — reset retry counter
        cuda_retries = 0

        # ── Step learning-rate schedulers ──
        hider_actor_scheduler.step()
        hider_critic_scheduler.step()
        seeker_actor_scheduler.step()
        seeker_critic_scheduler.step()

        # ── Logging (average across N envs) ──
        round_hider_rews = []
        round_seeker_rews = []
        for ei in range(n_envs):
            h_rew = np.mean([ep_rewards[ei][n] for n in HIDER_NAMES])
            s_rew = np.mean([ep_rewards[ei][n] for n in SEEKER_NAMES])
            round_hider_rews.append(h_rew)
            round_seeker_rews.append(s_rew)
            episode_rewards_hiders.append(h_rew)
            episode_rewards_seekers.append(s_rew)

        avg_h = np.mean(round_hider_rews)
        avg_s = np.mean(round_seeker_rews)

        # TensorBoard
        writer.add_scalar("reward/hiders_mean", avg_h, total_episodes)
        writer.add_scalar("reward/seekers_mean", avg_s, total_episodes)
        writer.add_scalar("reward/hiders_running_avg",
                          np.mean(episode_rewards_hiders[-100:]), total_episodes)
        writer.add_scalar("reward/seekers_running_avg",
                          np.mean(episode_rewards_seekers[-100:]), total_episodes)
        writer.add_scalar("episode/total_episodes", total_episodes, round_num)
        writer.add_scalar("episode/total_steps", total_steps, round_num)
        writer.add_scalar("lr/actor", hider_actor_scheduler.get_last_lr()[0], round_num)

        if hider_stats:
            writer.add_scalar("loss/hider_policy", hider_stats["policy_loss"], total_episodes)
            writer.add_scalar("loss/hider_value", hider_stats["value_loss"], total_episodes)
            writer.add_scalar("loss/hider_entropy", hider_stats["entropy"], total_episodes)
        if seeker_stats:
            writer.add_scalar("loss/seeker_policy", seeker_stats["policy_loss"], total_episodes)
            writer.add_scalar("loss/seeker_value", seeker_stats["value_loss"], total_episodes)
            writer.add_scalar("loss/seeker_entropy", seeker_stats["entropy"], total_episodes)

        # Progress bar
        if round_num % args.log_interval == 0:
            run_h = np.mean(episode_rewards_hiders[-n_envs * args.log_interval:])
            run_s = np.mean(episode_rewards_seekers[-n_envs * args.log_interval:])
            pbar.set_postfix({
                "H": f"{run_h:.1f}",
                "S": f"{run_s:.1f}",
                "ep": total_episodes,
                "steps": f"{total_steps/1e6:.2f}M",
            })

        # Save checkpoints
        if round_num % args.save_interval == 0:
            hider_policy.save(os.path.join(save_path, f"hider_policy_ep{total_episodes}.pt"))
            seeker_policy.save(os.path.join(save_path, f"seeker_policy_ep{total_episodes}.pt"))
            last_checkpoint_round = round_num
            print(f"\n  [SAVE] Checkpoint saved at episode {total_episodes}")

        # Track best
        if avg_h > best_hider_reward:
            best_hider_reward = avg_h
            hider_policy.save(os.path.join(save_path, "best_hider_policy.pt"))
            seeker_policy.save(os.path.join(save_path, "best_seeker_policy.pt"))

      except (RuntimeError, torch.cuda.CudaError) as e:
        err_msg = str(e).lower()
        if "cuda" in err_msg or "illegal memory" in err_msg or "device-side" in err_msg:
            _recover_from_cuda_error(round_num)
            continue  # retry from next round
        else:
            raise  # non-CUDA RuntimeError — reraise
      except Exception as e:
        # PyTorch 2.10+ raises torch.AcceleratorError instead of RuntimeError
        err_msg = str(e).lower()
        if "cuda" in err_msg or "illegal memory" in err_msg:
            _recover_from_cuda_error(round_num)
            continue
        else:
            raise

    # ── Final save ──
    hider_policy.save(os.path.join(save_path, "final_hider_policy.pt"))
    seeker_policy.save(os.path.join(save_path, "final_seeker_policy.pt"))
    writer.close()
    vec_env.close()

    # ── Re-allow OS sleep ──
    _allow_sleep()

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Rounds:     {n_rounds}")
    print(f"  Episodes:   {total_episodes}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Models:     {save_path}")
    print(f"  Logs:       {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
