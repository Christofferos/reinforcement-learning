# 🔍 Hide and Seek — Multi-Agent Reinforcement Learning

A modern recreation of OpenAI's **"Emergent Tool Use From Multi-Agent Autocurricula"** (2019), built with current open-source tooling.

<p align="center">
  <img src="https://openai.com/content/images/2019/09/image3.gif" width="500" alt="Hide and Seek demo"/>
</p>

## 🏗️ Architecture

| Component | Technology | Why |
|-----------|-----------|-----|
| **Physics** | [MuJoCo](https://mujoco.org/) 3.x (DeepMind) | The standard for RL robotics physics. Free, fast, GPU-ready. |
| **Single-agent API** | [Gymnasium](https://gymnasium.farama.org/) 1.x (Farama) | Modern successor to OpenAI Gym. |
| **Multi-agent API** | [PettingZoo](https://pettingzoo.farama.org/) (Farama) | De-facto standard MARL API, works with RLlib/CleanRL/etc. |
| **Algorithm** | **MAPPO** (custom PyTorch) | Multi-Agent PPO with CTDE (Centralized Training, Decentralized Execution). Still the gold standard for cooperative-competitive MARL — see [Yu et al. 2021](https://arxiv.org/abs/2103.01955). |
| **ML Framework** | PyTorch 2.x | Latest stable, with `torch.compile` support. |

### Why MAPPO over alternatives?

| Algorithm | Pros | Cons | Verdict |
|-----------|------|------|---------|
| **MAPPO** | Stable, sample-efficient, handles mixed coop/competitive, shared policies | On-policy (less sample efficient than off-policy) | ⭐ **Best fit** — proven on hide-and-seek scale tasks |
| MADDPG | Off-policy, good for competitive | Less stable, harder to tune | Good alternative for pure competition |
| QMIX/QPLEX | Value decomposition | Only cooperative, discrete actions | Not suitable (need competitive) |
| MAT (Multi-Agent Transformer) | Attention-based coordination | Newer, less battle-tested | Promising future direction |

## 🎮 Environment Design

Inspired directly by the [OpenAI hide-and-seek paper](https://arxiv.org/abs/1909.07528):

- **Arena**: 12×12m walled quadrant with divider walls and a door gap
- **Teams**: 2 Hiders (cyan) vs 2 Seekers (red)
- **Objects**: 3 moveable boxes + 1 ramp
- **Phases**:
  - **Preparation** (40% of episode): Seekers are frozen, hiders can arrange objects
  - **Play** (60% of episode): Seekers unfreeze and hunt
- **Actions** (continuous): Move X/Y, Grab object
- **Observations**: Self state, relative positions of visible agents/objects, 30-ray lidar
- **Rewards** (joint zero-sum):
  - Hiders: +1 if ALL hidden, -1 otherwise
  - Seekers: +1 if ANY sees a hider, -1 otherwise
- **Line-of-sight**: MuJoCo raycasting — walls and boxes block vision

## 📁 Project Structure

```
hide-n-seek/
├── src/                         # Source modules
│   ├── __init__.py
│   ├── config.py                # All hyperparameters (env + MAPPO)
│   ├── env.py                   # Core MuJoCo environment (Gymnasium API)
│   ├── worldgen.py              # Procedural arena generator (4 layout families)
│   ├── mappo.py                 # MAPPO algorithm (Actor, Critic, TeamPolicy)
│   ├── vec_env.py               # Synchronous vectorized multi-agent env
│   └── pettingzoo_wrapper.py    # PettingZoo Parallel API wrapper
├── scripts/                     # Entry-point scripts
│   ├── train.py                 # Training entrypoint
│   └── evaluate.py              # Evaluation & video recording
├── tests/                       # Integration & regression tests
│   ├── random_test.py           # Sanity check with random actions
│   ├── test_grab.py             # Grab mechanic + colour indicator test
│   ├── test_procgen.py          # Procedural worldgen stress test
│   └── test_ramp_climb.py       # Ramp physics integration test
├── assets/                      # Static MuJoCo XML (non-procedural mode)
│   └── hide_and_seek.xml
├── models/                      # Saved model checkpoints (gitignored)
├── runs/                        # TensorBoard logs (gitignored)
├── docs/                        # Design docs & mockups
│   └── hidenseek-html-mockup.html
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install
pip install -r requirements.txt
```

### 2. Sanity check

```bash
# Headless (no window)
python tests/random_test.py

# With MuJoCo viewer
python tests/random_test.py --render
```

### 3. Train

```bash
# Basic training
python scripts/train.py

# With visualization
python scripts/train.py --render

# Custom hyperparameters
python scripts/train.py --n_episodes 50000 --device cuda --hidden_dim 512

# Monitor with TensorBoard
tensorboard --logdir runs/
```

### 4. Evaluate

```bash
# Evaluate trained model
python scripts/evaluate.py --model_dir models/hideseek_mappo_XXXXXXXX

# Record video
python scripts/evaluate.py --model_dir models/hideseek_mappo_XXXXXXXX --record
```

## 🧠 Key Design Decisions

### Shared Policy per Team
Like OpenAI's original, each team shares a single neural network. All hiders use the same actor/critic; all seekers share another. This enables:
- Emergent coordination (agents learn team strategies)
- Better sample efficiency (more data per policy update)
- Scalability (add more agents without more parameters)

### Centralized Training, Decentralized Execution (CTDE)
- **Actor** (decentralized): Only sees local observation → portable to real deployment
- **Critic** (centralized): Sees global state during training → better value estimates

### Continuous Actions
Unlike the original (which discretized movement), we use continuous actions with Gaussian policies. This allows smoother movement and more nuanced object manipulation.

## 📊 Expected Emergent Behaviors

With sufficient training (~50M+ environment steps), the original paper observed these phases:

1. **Random** — Agents move aimlessly
2. **Running** — Seekers chase hiders, hiders run away
3. **Fort building** — Hiders learn to push boxes to block the door
4. **Ramp exploitation** — Seekers learn to use ramps to jump over walls
5. **Ramp denial** — Hiders learn to lock ramps during prep phase
6. **Box surfing** — Seekers discover physics exploits (emergent!)

## 📚 References

- Baker et al. **"Emergent Tool Use From Multi-Agent Autocurricula"** (2019) — [Paper](https://arxiv.org/abs/1909.07528) | [Blog](https://openai.com/blog/emergent-tool-use/)
- Yu et al. **"The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"** (2021) — [Paper](https://arxiv.org/abs/2103.01955)
- [OpenAI's original environment code](https://github.com/openai/multi-agent-emergence-environments) (archived)
- [PettingZoo documentation](https://pettingzoo.farama.org/)
- [MuJoCo documentation](https://mujoco.readthedocs.io/)
