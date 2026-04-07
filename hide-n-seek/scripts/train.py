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
import glob
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
from config import ENV_CONFIG, MAPPO_CONFIG, CURRICULUM_PHASES


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hide & Seek with MAPPO")

    # Curriculum
    parser.add_argument("--curriculum_phase", type=int, default=None,
                        choices=[1, 2, 3],
                        help="Curriculum phase (1=simple, 2=complex, 3=full). "
                             "Overrides horizon, n_episodes, hidden_dim, entropy schedule.")

    # Environment
    parser.add_argument("--horizon", type=int, default=None,
                        help=f"Episode horizon (default: {ENV_CONFIG['horizon']}, overridden by curriculum)")
    parser.add_argument("--prep_fraction", type=float, default=ENV_CONFIG["prep_fraction"])
    parser.add_argument("--reward_type", type=str, default=ENV_CONFIG["reward_type"],
                        choices=["joint_zero_sum", "joint_mean", "selfish"])
    parser.add_argument("--n_envs", type=int, default=MAPPO_CONFIG["n_envs"],
                        help="Number of parallel environments")

    # Training
    parser.add_argument("--n_episodes", type=int, default=None,
                        help="Total episodes across all envs (default: 20000, overridden by curriculum)")
    parser.add_argument("--ppo_epochs", type=int, default=MAPPO_CONFIG["ppo_epochs"])
    parser.add_argument("--n_accum_rounds", type=int, default=MAPPO_CONFIG["n_accum_rounds"],
                        help="Accumulate N rounds before PPO update (effective batch = n_envs × horizon × N)")
    parser.add_argument("--lr_actor", type=float, default=MAPPO_CONFIG["lr_actor"])
    parser.add_argument("--lr_critic", type=float, default=MAPPO_CONFIG["lr_critic"])
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help=f"Critic hidden dim (default: {MAPPO_CONFIG['hidden_dim']}, overridden by curriculum)")
    parser.add_argument("--gamma", type=float, default=MAPPO_CONFIG["gamma"])
    parser.add_argument("--clip_epsilon", type=float, default=MAPPO_CONFIG["clip_epsilon"])
    parser.add_argument("--entropy_coef", type=float, default=None,
                        help=f"Entropy coefficient (default: {MAPPO_CONFIG['entropy_coef']}, overridden by curriculum)")
    _default_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--device", type=str, default=_default_device)

    # Logging
    parser.add_argument("--log_interval", type=int, default=MAPPO_CONFIG["log_interval"],
                        help="Rounds between progress-bar updates")
    parser.add_argument("--save_interval", type=int, default=MAPPO_CONFIG["save_interval"],
                        help="Episodes between checkpoint saves (e.g. 5000 → ep5000, ep10000, ...)")
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

    # ── Apply curriculum phase overrides ──
    cur_phase = None
    allowed_layouts = None
    n_boxes_range = None
    n_ramps_range = None
    entropy_start = MAPPO_CONFIG["entropy_coef"]
    entropy_end = MAPPO_CONFIG.get("entropy_coef_end", 0.005)

    if args.curriculum_phase is not None:
        cur_phase = CURRICULUM_PHASES[args.curriculum_phase]
        print(f"\n  [CURRICULUM] Phase {args.curriculum_phase}: {cur_phase['description']}")
        # Override env params from curriculum if not explicitly set on CLI
        if args.horizon is None:
            args.horizon = cur_phase["horizon"]
        if args.n_episodes is None:
            args.n_episodes = cur_phase["suggested_episodes"]
        if args.hidden_dim is None:
            args.hidden_dim = cur_phase["hidden_dim"]
        if args.entropy_coef is None:
            entropy_start = cur_phase["entropy_coef_start"]
            entropy_end = cur_phase["entropy_coef_end"]
        else:
            entropy_start = args.entropy_coef
        allowed_layouts = cur_phase["allowed_layouts"]
        n_boxes_range = cur_phase["n_boxes_range"]
        n_ramps_range = cur_phase["n_ramps_range"]
    else:
        # Defaults when no curriculum phase specified
        if args.horizon is None:
            args.horizon = ENV_CONFIG["horizon"]
        if args.n_episodes is None:
            args.n_episodes = 2_000_000
        if args.hidden_dim is None:
            args.hidden_dim = MAPPO_CONFIG["hidden_dim"]
        if args.entropy_coef is not None:
            entropy_start = args.entropy_coef

    n_envs = args.n_envs
    vec_env = SyncVectorMultiAgentEnv(
        n_envs=n_envs,
        horizon=args.horizon,
        prep_fraction=args.prep_fraction,
        reward_type=args.reward_type,
        render_mode="human" if args.render else None,
        procedural=not args.no_procedural,
        allowed_layouts=allowed_layouts,
        n_boxes_range=n_boxes_range,
        n_ramps_range=n_ramps_range,
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
    if cur_phase:
        print(f"  Curriculum:       Phase {args.curriculum_phase} \u2014 {cur_phase['description']}")
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
    print(f"  Accum rounds:     {args.n_accum_rounds} (update every {args.n_accum_rounds} rounds)")
    print(f"  Effective batch:  {args.horizon * n_envs * args.n_accum_rounds:,} steps/update")
    print(f"  Entropy:          {entropy_start:.4f} \u2192 {entropy_end:.4f} (linear decay)")
    print(f"  LR end factor:    {MAPPO_CONFIG.get('lr_end_factor', 0.1)}")
    print(f"{'='*60}\n")

    # ── Build policy config ──
    # Use curriculum-appropriate network sizes
    attn_embed = (cur_phase["attn_embed_dim"] if cur_phase
                  else MAPPO_CONFIG.get("attn_embed_dim", 128))
    attn_heads = (cur_phase["attn_n_heads"] if cur_phase
                  else MAPPO_CONFIG.get("attn_n_heads", 4))
    attn_layers = (cur_phase["attn_n_layers"] if cur_phase
                   else MAPPO_CONFIG.get("attn_n_layers", 2))

    policy_config = {
        "gamma": args.gamma,
        "gae_lambda": MAPPO_CONFIG["gae_lambda"],
        "clip_epsilon": args.clip_epsilon,
        "entropy_coef": entropy_start,
        "value_coef": MAPPO_CONFIG["value_coef"],
        "max_grad_norm": MAPPO_CONFIG["max_grad_norm"],
        "ppo_epochs": args.ppo_epochs,
        "mini_batch_size": MAPPO_CONFIG["mini_batch_size"],
        "hidden_dim": args.hidden_dim,
        "n_layers": MAPPO_CONFIG["n_layers"],
        "use_feature_norm": MAPPO_CONFIG["use_feature_norm"],
        "lr_actor": args.lr_actor,
        "lr_critic": args.lr_critic,
        # Entity-attention actor config (curriculum-aware sizes)
        "attn_embed_dim": attn_embed,
        "attn_n_heads": attn_heads,
        "attn_n_layers": attn_layers,
        # Value normalizer warmup
        "value_norm_warmup_rounds": MAPPO_CONFIG.get("value_norm_warmup_rounds", 50),
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
            try:
                hider_policy.load(latest_hider)
                seeker_policy.load(latest_seeker)
                # Parse episode number from filename
                ep_num = int(os.path.basename(latest_hider).split("_ep")[1].split(".pt")[0])
                start_round = ep_num // n_envs + 1
                total_episodes = ep_num
                total_steps = ep_num * args.horizon
                print(f"  [>>] Resumed from {latest_hider}")
                print(f"       Starting at round {start_round}, episode {total_episodes}")
            except RuntimeError as e:
                if "size mismatch" in str(e) or "Missing key" in str(e) or "Unexpected key" in str(e):
                    print(f"  [!]  Architecture changed (curriculum phase transition).")
                    print(f"       Cannot load weights — starting with fresh network.")
                    print(f"       (This is expected when moving between curriculum phases")
                    print(f"        with different network sizes.)")
                else:
                    raise
        else:
            # Try crash checkpoint first (most recent state before CUDA death)
            crash_h = os.path.join(args.resume, "crash_hider_policy.pt")
            crash_s = os.path.join(args.resume, "crash_seeker_policy.pt")
            crash_meta = os.path.join(args.resume, "crash_meta.txt")
            best_h = os.path.join(args.resume, "best_hider_policy.pt")
            best_s = os.path.join(args.resume, "best_seeker_policy.pt")

            if os.path.exists(crash_h) and os.path.exists(crash_s):
                try:
                    hider_policy.load(crash_h)
                    seeker_policy.load(crash_s)
                    # Try to read episode counter from metadata
                    if os.path.exists(crash_meta):
                        meta = {}
                        for line in open(crash_meta):
                            k, v = line.strip().split("=")
                            meta[k] = int(v)
                        total_episodes = meta.get("total_episodes", 0)
                        total_steps = meta.get("total_steps", 0)
                        start_round = total_episodes // n_envs + 1
                        print(f"  [>>] Resumed from crash checkpoint (ep {total_episodes})")
                        print(f"       Starting at round {start_round}")
                    else:
                        print(f"  [>>] Resumed from crash checkpoint (no metadata)")
                    # Clean up crash files after loading
                    try:
                        os.remove(crash_h)
                        os.remove(crash_s)
                        if os.path.exists(crash_meta):
                            os.remove(crash_meta)
                    except Exception:
                        pass
                except RuntimeError:
                    print(f"  [!]  Architecture mismatch — starting fresh")
            elif os.path.exists(best_h) and os.path.exists(best_s):
                try:
                    hider_policy.load(best_h)
                    seeker_policy.load(best_s)
                    print(f"  [>>] Resumed from best_*_policy.pt (no episode counter)")
                    print(f"       Weights + optimizer state loaded, starting round 1")
                except RuntimeError:
                    print(f"  [!]  Architecture mismatch — starting fresh")
            else:
                print(f"  [!]  No checkpoints found in {args.resume}, starting fresh")

    def get_policy(agent_name):
        return hider_policy if agent_name in HIDER_NAMES else seeker_policy

    def buf_key(env_idx, agent_name):
        return (env_idx, agent_name)

    # \u2500\u2500 Learning-rate schedulers (linear annealing to 10% \u2014 NOT zero) \u2500\u2500
    lr_end_factor = MAPPO_CONFIG.get("lr_end_factor", 0.1)
    hider_actor_scheduler = torch.optim.lr_scheduler.LinearLR(
        hider_policy.actor_optimizer, start_factor=1.0, end_factor=lr_end_factor,
        total_iters=n_rounds)
    hider_critic_scheduler = torch.optim.lr_scheduler.LinearLR(
        hider_policy.critic_optimizer, start_factor=1.0, end_factor=lr_end_factor,
        total_iters=n_rounds)
    seeker_actor_scheduler = torch.optim.lr_scheduler.LinearLR(
        seeker_policy.actor_optimizer, start_factor=1.0, end_factor=lr_end_factor,
        total_iters=n_rounds)
    seeker_critic_scheduler = torch.optim.lr_scheduler.LinearLR(
        seeker_policy.critic_optimizer, start_factor=1.0, end_factor=lr_end_factor,
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

    # ── CUDA error handling ──
    # Once CUDA context is corrupted, in-process recovery is impossible.
    # Strategy: save emergency checkpoint on CPU, exit with code 42 so the
    # wrapper script (run_training.sh) can auto-restart with --resume.
    last_checkpoint_round = 0

    def _emergency_save_and_exit(round_at_crash, error):
        """Save weights on CPU and exit with restart code 42."""
        print(f"\n{'='*60}")
        print(f"[!] CUDA error at round {round_at_crash} (ep ~{total_episodes}):")
        print(f"    {error}")
        print(f"{'='*60}")

        # Try to save models on CPU
        try:
            for name, pol in [("hider", hider_policy), ("seeker", seeker_policy)]:
                pol.actor.cpu()
                pol.critic.cpu()
                ckpt = {
                    "actor": pol.actor.state_dict(),
                    "critic": pol.critic.state_dict(),
                    "actor_optimizer": pol.actor_optimizer.state_dict(),
                    "critic_optimizer": pol.critic_optimizer.state_dict(),
                }
                path = os.path.join(save_path, f"crash_{name}_policy.pt")
                torch.save(ckpt, path)
            # Also save a metadata file so resume knows where we were
            meta_path = os.path.join(save_path, "crash_meta.txt")
            with open(meta_path, "w") as f:
                f.write(f"round={round_at_crash}\n")
                f.write(f"total_episodes={total_episodes}\n")
                f.write(f"total_steps={total_steps}\n")
            print(f"[OK] Emergency checkpoint saved to {save_path}/crash_*_policy.pt")
        except Exception as save_err:
            print(f"[X]  Emergency save failed: {save_err}")

        # Clean up
        try:
            writer.close()
        except Exception:
            pass
        try:
            vec_env.close()
        except Exception:
            pass
        _allow_sleep()

        print(f"[>>] Exiting with code 42 (auto-restart).\n")
        sys.exit(42)

    pbar = tqdm(range(start_round, n_rounds + 1), desc="Training", unit="round",
                initial=start_round - 1, total=n_rounds)

    for round_num in pbar:
      try:
        round_t0 = time.time()

        # Reset all envs at the start of each round
        all_obs, all_info = vec_env.reset()
        all_global_states = [info["global_state"] for info in all_info]

        # Per-env episode reward tracking
        ep_rewards = [
            {name: 0.0 for name in AGENT_NAMES}
            for _ in range(n_envs)
        ]

        # Per-env per-agent reward breakdown accumulation
        ep_reward_breakdown = [
            {name: {} for name in AGENT_NAMES}
            for _ in range(n_envs)
        ]

        # Run for exactly `horizon` steps (all envs in lockstep)
        # Pre-compute team membership indices for fast slicing
        n_hiders = len(HIDER_NAMES)
        n_seekers = len(SEEKER_NAMES)

        for step in range(args.horizon):
            # ── Batched action selection ──
            # Stack obs for each team across ALL envs into (n_envs*n_team, obs_dim)
            hider_obs_list = []
            hider_state_list = []
            seeker_obs_list = []
            seeker_state_list = []

            for ei in range(n_envs):
                gs = all_global_states[ei]
                for name in HIDER_NAMES:
                    hider_obs_list.append(all_obs[ei][name])
                    hider_state_list.append(gs)
                for name in SEEKER_NAMES:
                    seeker_obs_list.append(all_obs[ei][name])
                    seeker_state_list.append(gs)

            hider_obs_batch = np.stack(hider_obs_list, axis=0)    # (n_envs*n_hiders, obs_dim)
            hider_state_batch = np.stack(hider_state_list, axis=0)
            seeker_obs_batch = np.stack(seeker_obs_list, axis=0)   # (n_envs*n_seekers, obs_dim)
            seeker_state_batch = np.stack(seeker_state_list, axis=0)

            # Two batched GPU calls instead of n_envs * n_agents individual calls
            h_actions, h_log_probs, h_values = hider_policy.select_actions_batch(
                hider_obs_batch, hider_state_batch, deterministic=False)
            s_actions, s_log_probs, s_values = seeker_policy.select_actions_batch(
                seeker_obs_batch, seeker_state_batch, deterministic=False)

            # Unpack into per-env action dicts for vec_env.step()
            all_actions = [{} for _ in range(n_envs)]
            # Also store per-agent data for buffer insertion
            all_action_data = [{} for _ in range(n_envs)]

            for ei in range(n_envs):
                for hi, name in enumerate(HIDER_NAMES):
                    idx = ei * n_hiders + hi
                    all_actions[ei][name] = h_actions[idx]
                    all_action_data[ei][name] = (h_actions[idx], h_log_probs[idx], h_values[idx])
                for si, name in enumerate(SEEKER_NAMES):
                    idx = ei * n_seekers + si
                    all_actions[ei][name] = s_actions[idx]
                    all_action_data[ei][name] = (s_actions[idx], s_log_probs[idx], s_values[idx])

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

            # Accumulate reward breakdown from this step
            for ei in range(n_envs):
                step_breakdown = all_info[ei].get("reward_breakdown", {})
                for name in AGENT_NAMES:
                    if name in step_breakdown:
                        for comp, val in step_breakdown[name].items():
                            ep_reward_breakdown[ei][name][comp] = (
                                ep_reward_breakdown[ei][name].get(comp, 0.0) + val
                            )

            all_obs = next_all_obs
            all_global_states = next_global_states
            total_steps += n_envs  # N envs stepped simultaneously

        # ── Round complete: all envs finished 1 episode ──
        total_episodes += n_envs
        round_dt = time.time() - round_t0
        round_steps = n_envs * args.horizon  # total agent-steps this round

        if round_num <= 3:
            sps = round_steps / max(round_dt, 1e-6)
            print(f"\n  [PERF] Round {round_num}: {round_dt:.1f}s "
                  f"({sps:.0f} env-steps/s, "
                  f"{n_envs * args.horizon * 4 / max(round_dt, 1e-6):.0f} agent-steps/s)")

        # ── Accumulate N rounds before PPO update ──
        # Effective batch = n_envs × horizon × n_accum_rounds
        # 128 × 200 × 4 = 102,400  (≈ OpenAI's 115,200)
        if round_num % args.n_accum_rounds == 0:
            hider_stats = hider_policy.update()
            seeker_stats = seeker_policy.update()
        else:
            hider_stats = {}
            seeker_stats = {}

        # ── Periodic weight-NaN check (catches silent corruption early) ──
        if hider_policy.has_nan_weights() or seeker_policy.has_nan_weights():
            print(f"\n  [!] NaN detected in model weights at round {round_num}!")
            _emergency_save_and_exit(round_num, ValueError("NaN in model weights"))

        # ── Step learning-rate schedulers ──
        hider_actor_scheduler.step()
        hider_critic_scheduler.step()
        seeker_actor_scheduler.step()
        seeker_critic_scheduler.step()

        # ── Entropy coefficient linear decay ──
        progress = (round_num - start_round) / max(n_rounds - start_round, 1)
        current_entropy = entropy_start + (entropy_end - entropy_start) * progress
        current_entropy = max(current_entropy, entropy_end)
        hider_policy.entropy_coef = current_entropy
        seeker_policy.entropy_coef = current_entropy

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
        writer.add_scalar("lr/entropy_coef", current_entropy, round_num)
        writer.add_scalar("perf/round_time_s", round_dt, round_num)
        writer.add_scalar("perf/steps_per_sec", round_steps / max(round_dt, 1e-6), round_num)

        # ── Reward breakdown per component (averaged across envs & agents) ──
        hider_comp_sums = {}
        seeker_comp_sums = {}
        for ei in range(n_envs):
            for name in HIDER_NAMES:
                for comp, val in ep_reward_breakdown[ei][name].items():
                    hider_comp_sums[comp] = hider_comp_sums.get(comp, 0.0) + val
            for name in SEEKER_NAMES:
                for comp, val in ep_reward_breakdown[ei][name].items():
                    seeker_comp_sums[comp] = seeker_comp_sums.get(comp, 0.0) + val
        n_hider_agents = n_envs * len(HIDER_NAMES)
        n_seeker_agents = n_envs * len(SEEKER_NAMES)
        for comp, total in hider_comp_sums.items():
            writer.add_scalar(f"reward_hider/{comp}", total / n_hider_agents, total_episodes)
        for comp, total in seeker_comp_sums.items():
            writer.add_scalar(f"reward_seeker/{comp}", total / n_seeker_agents, total_episodes)

        # ── Game outcome stats ──
        # Count how many envs had hiders fully hidden at last step
        hider_win_count = 0
        for ei in range(n_envs):
            base_vals = [ep_reward_breakdown[ei][n].get("base", 0.0) for n in HIDER_NAMES]
            if base_vals and np.mean(base_vals) > 0:
                hider_win_count += 1
        hider_win_rate = hider_win_count / n_envs
        writer.add_scalar("game/hider_win_rate", hider_win_rate, total_episodes)
        writer.add_scalar("game/seeker_win_rate", 1.0 - hider_win_rate, total_episodes)

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
                "s/rnd": f"{round_dt:.1f}",
                "sps": f"{round_steps / max(round_dt, 1e-6):.0f}",
            })

        # Save checkpoints (save_interval is in EPISODES, not rounds)
        # Check if we crossed a save_interval boundary this round
        prev_episodes = total_episodes - n_envs  # episodes before this round
        if (total_episodes // args.save_interval) > (prev_episodes // args.save_interval):
            save_ep_label = (total_episodes // args.save_interval) * args.save_interval
            hider_policy.save(os.path.join(save_path, f"hider_policy_ep{save_ep_label}.pt"))
            seeker_policy.save(os.path.join(save_path, f"seeker_policy_ep{save_ep_label}.pt"))
            last_checkpoint_round = round_num
            print(f"\n  [SAVE] Checkpoint saved at episode {save_ep_label}")

        # Track best
        if avg_h > best_hider_reward:
            best_hider_reward = avg_h
            hider_policy.save(os.path.join(save_path, "best_hider_policy.pt"))
            seeker_policy.save(os.path.join(save_path, "best_seeker_policy.pt"))

      except (RuntimeError, ValueError, torch.cuda.CudaError) as e:
        err_msg = str(e).lower()
        if ("cuda" in err_msg or "illegal memory" in err_msg
                or "device-side" in err_msg or "nan" in err_msg
                or "invalid values" in err_msg or "constraint" in err_msg):
            _emergency_save_and_exit(round_num, e)
        else:
            raise  # non-recoverable error — reraise
      except Exception as e:
        # PyTorch 2.10+ raises torch.AcceleratorError instead of RuntimeError
        err_msg = str(e).lower()
        if ("cuda" in err_msg or "illegal memory" in err_msg
                or "nan" in err_msg or "invalid values" in err_msg):
            _emergency_save_and_exit(round_num, e)
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
