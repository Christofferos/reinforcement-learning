"""
Configuration for the Hide and Seek environment and training.
"""

# ─────────────────────── Environment ───────────────────────

ENV_CONFIG = {
    # Arena
    "floor_size": 6.0,
    "xml_path": "assets/hide_and_seek.xml",

    # Procedural world generation
    "procedural": True,           # generate random layouts each reset
    "max_boxes": 5,               # max boxes (obs/state dimension is fixed to this)
    "max_ramps": 2,               # max ramps

    # Teams
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

    # Actions
    "movement_scale": 1.0,        # max force applied per step
    "grab_radius": 0.8,           # radius to grab objects
}

# ─────────────────────── MAPPO Training ───────────────────────

MAPPO_CONFIG = {
    # PPO core
    "gamma": 0.998,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.05,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,

    # Learning rates
    "lr_actor": 3e-4,
    "lr_critic": 1e-3,

    # Network (MLP critic)
    "hidden_dim": 256,
    "n_layers": 2,
    "use_orthogonal_init": True,
    "use_feature_norm": True,

    # Network (Entity-attention actor — OpenAI-style)
    "attn_embed_dim": 128,
    "attn_n_heads": 4,
    "attn_n_layers": 2,

    # Training
    "n_rollout_steps": 240,       # steps per rollout (= 1 episode)
    "ppo_epochs": 10,
    "mini_batch_size": 256,
    "n_total_steps": 50_000_000,  # total environment steps
    "n_envs": 32,                  # parallel environments

    # Shared policy per team
    "share_policy_within_team": True,

    # Logging
    "log_interval": 10,           # episodes between logs
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
