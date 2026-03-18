"""
Evaluate and visualise trained Hide & Seek policies.
=====================================================

Usage:
    python scripts/evaluate.py --model_dir models/hideseek_mappo_XXXXXXXX
    python scripts/evaluate.py --model_dir models/hideseek_mappo_XXXXXXXX --record
    python scripts/evaluate.py --random                         # random actions (sanity check)
    python scripts/evaluate.py --random --slow --horizon 600    # slow-motion random
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow imports from src/
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Ensure relative paths (models/, assets/) resolve from project root
os.chdir(_PROJECT_ROOT)

from env import HideAndSeekEnv, HIDER_NAMES, SEEKER_NAMES, AGENT_NAMES
from mappo import TeamPolicy
from config import ENV_CONFIG, MAPPO_CONFIG, CURRICULUM_PHASES


# ─── Minimal color helpers for terminal summary lines ──────────────────────

CYAN  = "\033[96m"
RED   = "\033[91m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Hide & Seek")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to saved model directory")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=ENV_CONFIG["horizon"],
                        help="Steps per episode (default from config)")
    parser.add_argument("--random", action="store_true", help="Use random actions")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Load a specific checkpoint, e.g. 'ep3500' or 'crash'")
    parser.add_argument("--record", action="store_true", help="Record video")
    parser.add_argument("--slow", action="store_true",
                        help="Slow-motion playback (adds delay between steps)")
    parser.add_argument("--step_delay", type=float, default=0.08,
                        help="Seconds between steps in slow mode (default 0.08)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no_procedural", action="store_true",
                        help="Use static XML arena instead of procedural generation")
    parser.add_argument("--curriculum_phase", type=int, default=None,
                        help="Curriculum phase (1/2/3) — sets correct network architecture")
    return parser.parse_args()


def evaluate(args):
    render_mode = "rgb_array" if args.record else "human"
    if args.random and not args.record:
        render_mode = "human"

    # Apply curriculum phase environment constraints if specified
    env_kwargs = {}
    if args.curriculum_phase and args.curriculum_phase in CURRICULUM_PHASES:
        phase = CURRICULUM_PHASES[args.curriculum_phase]
        env_kwargs["allowed_layouts"] = phase.get("allowed_layouts")
        env_kwargs["n_boxes_range"] = phase.get("n_boxes_range")
        env_kwargs["n_ramps_range"] = phase.get("n_ramps_range")
        if "horizon" in phase:
            args.horizon = phase["horizon"]

    env = HideAndSeekEnv(
        horizon=args.horizon,
        render_mode=render_mode,
        procedural=not args.no_procedural,
        **env_kwargs,
    )

    hider_policy = None
    seeker_policy = None

    if not args.random and args.model_dir:
        obs, info = env.reset()
        obs_dim = obs[HIDER_NAMES[0]].shape[0]
        act_dim = 3
        global_state_dim = info["global_state"].shape[0]

        # Use curriculum phase architecture if specified
        cfg = MAPPO_CONFIG
        if args.curriculum_phase and args.curriculum_phase in CURRICULUM_PHASES:
            phase = CURRICULUM_PHASES[args.curriculum_phase]
            cfg = {**cfg, **{k: phase[k] for k in ("hidden_dim", "attn_embed_dim", "attn_n_heads", "attn_n_layers") if k in phase}}

        policy_config = {
            "hidden_dim": cfg["hidden_dim"],
            "n_layers": cfg["n_layers"],
            "use_feature_norm": cfg["use_feature_norm"],
            "attn_embed_dim": cfg.get("attn_embed_dim", 128),
            "attn_n_heads": cfg.get("attn_n_heads", 4),
            "attn_n_layers": cfg.get("attn_n_layers", 2),
        }

        hider_policy = TeamPolicy("hiders", obs_dim, act_dim, global_state_dim,
                                  policy_config, device=args.device)
        seeker_policy = TeamPolicy("seekers", obs_dim, act_dim, global_state_dim,
                                   policy_config, device=args.device)

        if args.checkpoint:
            hider_path = os.path.join(args.model_dir, f"hider_policy_{args.checkpoint}.pt")
            seeker_path = os.path.join(args.model_dir, f"seeker_policy_{args.checkpoint}.pt")
        else:
            hider_path = os.path.join(args.model_dir, "best_hider_policy.pt")
            seeker_path = os.path.join(args.model_dir, "best_seeker_policy.pt")

        if os.path.exists(hider_path):
            hider_policy.load(hider_path)
            print(f"  {CYAN}[OK]{RESET} Loaded hider policy from {hider_path}")
        else:
            print(f"  {DIM}[!]  Hider policy not found, using random{RESET}")
            hider_policy = None

        if os.path.exists(seeker_path):
            seeker_policy.load(seeker_path)
            print(f"  {RED}[OK]{RESET} Loaded seeker policy from {seeker_path}")
        else:
            print(f"  {DIM}[!]  Seeker policy not found, using random{RESET}")
            seeker_policy = None
        print()

    frames = []

    for ep in range(1, args.n_episodes + 1):
        obs, info = env.reset()
        global_state = info["global_state"]
        layout_type = info.get("layout_type", "static")
        mood = info.get("mood", "")
        n_boxes = info.get("n_boxes", "?")
        n_ramps = info.get("n_ramps", "?")
        ep_reward = {name: 0.0 for name in AGENT_NAMES}

        done = False
        step = 0

        while not done:
            actions = {}

            for name in AGENT_NAMES:
                if args.random or (name in HIDER_NAMES and hider_policy is None) or \
                   (name in SEEKER_NAMES and seeker_policy is None):
                    actions[name] = np.random.uniform(-1, 1, size=3).astype(np.float32)
                else:
                    policy = hider_policy if name in HIDER_NAMES else seeker_policy
                    action, _, _ = policy.select_action(
                        obs[name], global_state, deterministic=True
                    )
                    actions[name] = action

            obs, rewards, terminated, truncated, info = env.step(actions)
            global_state = info["global_state"]

            for name in AGENT_NAMES:
                ep_reward[name] += rewards[name]

            if args.record and render_mode == "rgb_array":
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            step += 1

            if args.slow:
                time.sleep(args.step_delay)

            done = all(truncated.get(n, False) or terminated.get(n, False) for n in AGENT_NAMES)

        # ── Episode summary (one clean line) ──
        hider_rew = np.mean([ep_reward[n] for n in HIDER_NAMES])
        seeker_rew = np.mean([ep_reward[n] for n in SEEKER_NAMES])
        winner = f"{CYAN}HIDERS WIN{RESET}" if hider_rew > seeker_rew else f"{RED}SEEKERS WIN{RESET}"
        layout_tag = f"{DIM}[{layout_type:8s} {mood:7s} B={n_boxes} R={n_ramps}]{RESET}"
        print(f"  Episode {ep:3d} | {step:3d} steps | "
              f"{CYAN}H: {hider_rew:+7.1f}{RESET} | "
              f"{RED}S: {seeker_rew:+7.1f}{RESET} | {winner} | {layout_tag}")

    if args.record and frames:
        import imageio
        video_path = "hide_and_seek_eval.mp4"
        imageio.mimwrite(video_path, frames, fps=30)
        print(f"\n[Video] Saved to {video_path}")

    env.close()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
