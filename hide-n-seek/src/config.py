"""
Configuration for the Hide and Seek environment and training.

Tuned for **single-machine training** (one GPU or CPU).
See CURRICULUM_PHASES for the staged difficulty schedule.
"""

# ─────────────────────── Curriculum Phases ───────────────────────
# Each phase gradually increases environment complexity.
# Start with Phase 1; advance when reward plateaus (or use episode thresholds).
#
# Usage:
#   python scripts/train.py --curriculum_phase 1 --n_episodes 5000
#   python scripts/train.py --curriculum_phase 2 --n_episodes 10000 --resume models/<prev_run>
#   python scripts/train.py --curriculum_phase 3 --n_episodes 20000 --resume models/<prev_run>

CURRICULUM_PHASES = {
    1: {
        "description": "2v2, simple arenas, 2 boxes, no ramps — learn basic movement & chasing",
        "n_hiders": 2,
        "n_seekers": 2,
        "n_boxes_range": (2, 2),       # fixed 2 boxes
        "n_ramps_range": (0, 0),       # no ramps
        "allowed_layouts": ["divider", "cross"],  # simple layouts only
        "horizon": 200,                # shorter episodes
        "prep_fraction": 0.4,
        # Smaller networks learn faster with limited data
        "attn_embed_dim": 64,
        "attn_n_heads": 2,
        "attn_n_layers": 1,
        "hidden_dim": 128,
        # Higher entropy for exploration
        "entropy_coef_start": 0.03,
        "entropy_coef_end": 0.01,
        "suggested_episodes": 5_000,
    },
    2: {
        "description": "2v2, complex arenas, 2-3 boxes, 0-1 ramps — learn tool use",
        "n_hiders": 2,
        "n_seekers": 2,
        "n_boxes_range": (2, 3),
        "n_ramps_range": (0, 1),
        "allowed_layouts": ["divider", "cross", "L_shape"],
        "horizon": 240,
        "prep_fraction": 0.4,
        "attn_embed_dim": 64,
        "attn_n_heads": 2,
        "attn_n_layers": 1,
        "hidden_dim": 128,
        "entropy_coef_start": 0.02,
        "entropy_coef_end": 0.008,
        "suggested_episodes": 10_000,
    },
    3: {
        "description": "2v2, full complexity — coordinated team strategies",
        "n_hiders": 2,
        "n_seekers": 2,
        "n_boxes_range": (3, 5),
        "n_ramps_range": (0, 2),
        "allowed_layouts": ["divider", "cross", "L_shape", "rooms"],
        "horizon": 240,
        "prep_fraction": 0.4,
        # Full-size networks for 2v2 coordination
        "attn_embed_dim": 128,
        "attn_n_heads": 4,
        "attn_n_layers": 2,
        "hidden_dim": 256,
        "entropy_coef_start": 0.02,
        "entropy_coef_end": 0.005,
        "suggested_episodes": 20_000,
    },
}

# ─────────────────────── Environment ───────────────────────

ENV_CONFIG = {
    # Arena
    "floor_size": 6.0,
    "xml_path": "assets/hide_and_seek.xml",

    # Procedural world generation
    "procedural": True,           # generate random layouts each reset
    "max_boxes": 5,               # max boxes (obs/state dimension is fixed to this)
    "max_ramps": 2,               # max ramps

    # Teams (defaults — overridden by curriculum phase)
    "n_hiders": 2,
    "n_seekers": 2,

    # Objects (only used in non-procedural mode)
    "n_boxes": 3,
    "n_ramps": 1,

    # Episode
    "horizon": 240,               # total episode steps
    "prep_fraction": 0.4,         # 40% preparation phase (seekers frozen)
    "n_substeps": 15,             # MuJoCo substeps per action

    # Observations
    "lidar_n_rays": 30,
    "lidar_max_dist": 18.0,  # ≥ arena diagonal (~17 m)
    "agent_obs_radius": 100.0,    # effectively unlimited — only line-of-sight matters (OpenAI style)
    "box_obs_radius": 6.0,

    # Rewards
    "reward_type": "joint_zero_sum",   # 'selfish', 'joint_mean', 'joint_zero_sum'
    "reward_scale": 1.0,

    # Reward shaping — DOUBLED prep-phase bonuses for single-machine training
    "shape_prep_movement": 0.01,       # hider bonus per unit speed during prep  (was 0.005)
    "shape_grab_and_move": 0.02,       # bonus for grabbing + moving an object   (was 0.01)
    "shape_hider_near_cover": 0.04,    # hider bonus during prep if wall ≤ 2m    (was 0.02)
    "shape_prep_near_object": 0.02,    # hider bonus during prep if near box/ramp(was 0.01)
    "shape_box_toward_wall": 0.03,     # NEW: hider bonus for pushing box closer to a wall
    "shape_hider_dist_from_seeker": 0.05,  # hider play bonus: flee from seekers
    "shape_hider_occluded": 0.03,      # hider play bonus per blocked seeker LOS
    "shape_hider_seen_proximity": 0.05,# hider play penalty when seen (scaled by closeness)
    "shape_hider_move_when_seen": 0.02,# hider play bonus for moving while visible
    "shape_seeker_dist_to_hider": 0.05,# seeker play bonus: chase hiders
    "shape_seeker_coverage": 0.03,     # seeker bonus per new 2m×2m cell visited
    "shape_seeker_team_coverage": 0.02,# extra bonus when cell is new for whole team
    "shape_seeker_center_post_prep": 0.03, # seeker bonus for approaching center after prep
    "shape_individual_blend": 0.4,     # fraction of per-agent reward blended in (was 0.25)

    # Actions
    "movement_scale": 1.0,        # max force applied per step
    "grab_radius": 0.8,           # radius to grab objects
}

# ─────────────────────── MAPPO Training ───────────────────────

MAPPO_CONFIG = {
    # PPO core — tuned for sample efficiency on single machine
    "gamma": 0.995,               # longer effective horizon (was 0.99)
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.02,         # start value — decayed during training (was 0.01)
    "entropy_coef_end": 0.005,    # final entropy coef after linear decay
    "value_coef": 0.5,
    "max_grad_norm": 0.5,

    # Learning rates — slightly lower for stability with fewer envs
    "lr_actor": 3e-4,             # (was 5e-4)
    "lr_critic": 3e-4,            # (was 5e-4)
    "lr_end_factor": 0.1,         # LR anneals to 10% of initial (was 0.0)

    # Network (MLP critic) — defaults for Phase 3
    "hidden_dim": 256,
    "n_layers": 2,
    "use_orthogonal_init": True,
    "use_feature_norm": True,

    # Network (Entity-attention actor — OpenAI-style) — defaults for Phase 3
    "attn_embed_dim": 128,
    "attn_n_heads": 4,
    "attn_n_layers": 2,

    # Training — tuned for higher sample reuse
    "n_rollout_steps": 240,       # steps per rollout (= 1 episode)
    "ppo_epochs": 15,             # more gradient steps per rollout (was 10)
    "mini_batch_size": 128,       # smaller batches → more updates (was 256)
    "n_total_steps": 50_000_000,  # total environment steps
    "n_envs": 16,                  # fewer envs → higher reuse rate (was 32)

    # Value normalizer warmup
    "value_norm_warmup_rounds": 50,  # skip normalisation for first 50 rounds

    # Shared policy per team
    "share_policy_within_team": True,

    # Logging
    "log_interval": 10,           # rounds between log prints
    "save_interval": 500,         # episodes between checkpoints
    "eval_interval": 100,
    "eval_episodes": 10,
    "log_dir": "runs",
    "save_dir": "models",
}

# ─────────────────────── Rendering ───────────────────────

RENDER_CONFIG = {
    "render_mode": "human",       # 'human', 'rgb_array', None
    "camera_id": 0,
    "width": 1280,
    "height": 720,
    "fps": 30,
}
