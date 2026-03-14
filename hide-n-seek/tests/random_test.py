"""
Quick sanity check: run the environment with random actions.
Verifies that the MuJoCo model loads, observations are correct,
and the game loop works end-to-end.

Usage:
    python random_test.py
    python random_test.py --render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from env import HideAndSeekEnv, AGENT_NAMES, HIDER_NAMES, SEEKER_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--n_episodes", type=int, default=3)
    args = parser.parse_args()

    env = HideAndSeekEnv(
        render_mode="human" if args.render else None,
        horizon=120,
    )

    print("=" * 60)
    print("Hide & Seek — Random Actions Sanity Check")
    print("=" * 60)

    obs, info = env.reset()

    # Print observation shapes
    for name in AGENT_NAMES:
        print(f"  {name}: obs shape = {obs[name].shape}")
    print(f"  Global state shape = {info['global_state'].shape}")
    print()

    for ep in range(1, args.n_episodes + 1):
        obs, info = env.reset()
        total_reward = {name: 0.0 for name in AGENT_NAMES}
        step = 0
        done = False

        while not done:
            actions = {
                name: np.random.uniform(-1, 1, size=3).astype(np.float32)
                for name in AGENT_NAMES
            }

            obs, rewards, terminated, truncated, info = env.step(actions)

            for name in AGENT_NAMES:
                total_reward[name] += rewards[name]

            step += 1
            done = all(truncated.get(n, False) for n in AGENT_NAMES)

        hider_rew = np.mean([total_reward[n] for n in HIDER_NAMES])
        seeker_rew = np.mean([total_reward[n] for n in SEEKER_NAMES])

        print(f"Episode {ep} | Steps: {step} | "
              f"Hiders avg reward: {hider_rew:+.2f} | "
              f"Seekers avg reward: {seeker_rew:+.2f} | "
              f"Prep phase: {info.get('prep_phase', '?')}")

    env.close()
    print("\n[OK] Sanity check passed!")


if __name__ == "__main__":
    main()
