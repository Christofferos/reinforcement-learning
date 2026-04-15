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
        "horizon": 240,                # OpenAI paper: 240
        "prep_fraction": 0.4,          # OpenAI paper: ~40% prep
        # Same network arch across all phases so weights transfer
        "attn_embed_dim": 128,
        "attn_n_heads": 4,
        "attn_n_layers": 2,
        "hidden_dim": 256,
        # Higher entropy for exploration in early phase
        "entropy_coef_start": 0.01,
        "entropy_coef_end": 0.005,
        "suggested_episodes": 500_000,
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
        "attn_embed_dim": 128,
        "attn_n_heads": 4,
        "attn_n_layers": 2,
        "hidden_dim": 256,
        "entropy_coef_start": 0.01,
        "entropy_coef_end": 0.003,
        "suggested_episodes": 1_000_000,
    },
    3: {
        "description": "2v2, full complexity — coordinated team strategies",
        "n_hiders": 2,
        "n_seekers": 2,
        "n_boxes_range": (3, 5),
        "n_ramps_range": (0, 2),
        "allowed_layouts": ["divider", "cross", "L_shape", "rooms"],
        "horizon": 240,                # OpenAI paper: 240
        "prep_fraction": 0.4,          # OpenAI paper: ~40% prep
        "attn_embed_dim": 128,
        "attn_n_heads": 4,
        "attn_n_layers": 2,
        "hidden_dim": 256,
        "entropy_coef_start": 0.01,
        "entropy_coef_end": 0.0,       # OpenAI: anneal to 0
        "suggested_episodes": 2_000_000,
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
    "horizon": 240,               # OpenAI paper: 240
    "prep_fraction": 0.4,         # OpenAI paper: ~40% preparation phase (seekers frozen)
    "n_substeps": 15,             # MuJoCo substeps per action

    # Observations
    "lidar_n_rays": 30,
    "lidar_max_dist": 18.0,  # ≥ arena diagonal (~17 m)
    "agent_obs_radius": 100.0,    # effectively unlimited — only line-of-sight matters (OpenAI style)
    "box_obs_radius": 6.0,

    # Rewards
    "reward_type": "joint_zero_sum",   # 'selfish', 'joint_mean', 'joint_zero_sum'
    "reward_scale": 1.0,

    # Reward shaping — light bonuses, base ±1 game reward dominates
    # (Actual values are hardcoded in env.py — these are for reference only)
    # Play-phase shaping is OFF to allow emergent strategy discovery.
    # Only prep-phase object interaction and seeker exploration are lightly shaped.
    "shape_individual_blend": 0.3,     # 30% individual / 70% team (OpenAI-style)

    # Level design
    "perimeter_gap": 1.5,                # gap between interior walls and outer walls for escape routes
    "perimeter_gap_prob": 0.7,           # probability of adding escape gap to each wall end

    # Actions
    "movement_scale": 1.0,        # max force applied per step
    "grab_radius": 0.8,           # radius to grab objects
}

# ─────────────────────── MAPPO Training ───────────────────────

MAPPO_CONFIG = {
    # PPO core — aligned with OpenAI "Emergent Tool Use" paper
    "gamma": 0.998,               # OpenAI paper discount factor
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.01,         # OpenAI paper: 0.01 decaying
    "entropy_coef_end": 0.0,      # OpenAI: anneal to zero
    "value_coef": 0.5,
    "max_grad_norm": 0.5,

    # Learning rates — OpenAI paper: ~1e-4
    "lr_actor": 1e-4,
    "lr_critic": 1e-4,
    "lr_end_factor": 0.0,         # OpenAI: anneal LR to zero

    # Network (MLP critic) — defaults for Phase 3
    "hidden_dim": 256,
    "n_layers": 2,
    "use_orthogonal_init": True,
    "use_feature_norm": True,

    # Network (Entity-attention actor — OpenAI-style) — defaults for Phase 3
    "attn_embed_dim": 128,
    "attn_n_heads": 4,
    "attn_n_layers": 2,

    # Training — aligned with OpenAI paper
    "n_rollout_steps": 240,       # steps per rollout (= 1 episode = horizon)
    "ppo_epochs": 1,              # OpenAI paper: 1 epoch per rollout
    "mini_batch_size": 960,       # 4 episodes per mini-batch (240 × 4)
    "n_accum_rounds": 4,          # accumulate 4 rounds before PPO update
                                   # effective batch = 128 × 240 × 4 = 122,880
                                   # (OpenAI: 480 × 240 = 115,200)
    "n_total_steps": 700_000_000, # ~700M steps across all curriculum phases
    "n_envs": 128,                # 128 parallel envs (batch ≈ 30,720 steps/round)

    # Value normalizer warmup
    "value_norm_warmup_rounds": 50,  # skip normalisation for first 50 rounds

    # Shared policy per team
    "share_policy_within_team": True,

    # Logging
    "log_interval": 50,           # rounds between log prints
    "save_interval": 50_000,      # episodes between checkpoints (~390 rounds)
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
