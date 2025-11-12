# Simple Tag - Multi-Agent Reinforcement Learning

A multi-agent reinforcement learning implementation for the **Simple Tag** environment from [MPE2 (Multi-Agent Particle Environment)](https://mpe2.farama.org/).

## Environment Description

Simple Tag is a predator-prey environment where:

- **Good agents** (green, 1 by default) are faster and try to avoid adversaries
- **Adversaries** (red, 3 by default) are slower and try to catch good agents
- **Obstacles** (black circles, 2 by default) block movement

### Rewards

- Adversaries: **+10** for each collision with a good agent
- Good agents: **-10** for being hit by adversaries
- Good agents: Penalty for leaving the bounded area

### Environment Details

- **Agents**: 4 (3 adversaries + 1 good agent)
- **Action Space**: Discrete(5) - [no action, left, right, down, up] or Continuous
- **Observation**: Agent velocity, position, relative positions of landmarks and other agents
- **Max Cycles**: 25 steps per episode

## Project Structure

```
MPE2/
├── config/
│   ├── __init__.py
│   └── config.py              # Environment and training configurations
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # Abstract base class for agents
│   ├── random_agent.py        # Random baseline agent
│   ├── dqn_agent.py           # Deep Q-Network agent
│   ├── maddpg_agent.py        # 🎯 MADDPG (CTDE algorithm)
│   └── mappo_agent.py         # 🎯 MAPPO (CTDE algorithm)
├── utils/
│   ├── __init__.py
│   ├── helpers.py             # Utility functions
│   ├── logger.py              # Training logger
│   └── visualizer.py          # Plotting and visualization
├── train.py                   # Training script (DQN/Random)
├── train_maddpg.py            # 🎯 MADDPG training script
├── evaluate.py                # Evaluation script (DQN/Random)
├── evaluate_maddpg.py         # 🎯 MADDPG evaluation script
├── MADDPG_GUIDE.md            # 📖 Complete MADDPG guide
├── MADDPG_QUICKSTART.md       # ⚡ Quick reference for MADDPG
├── ALGORITHMS.md              # 📖 Algorithm comparison
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore file
└── README.md                 # This file
```

## Installation

1. **Clone the repository**

```bash
cd /Users/kriwer/Documents/Github/reinforcement-learning/MARL/MPE2
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Usage

### Training

#### Train with DQN (Independent Learning)

Train agents using the default DQN algorithm:

```bash
python train.py
```

**Training options:**

```bash
python train.py --episodes 5000 --agent-type dqn --experiment-name my_experiment
```

Arguments:

- `--episodes`: Number of training episodes (default: 10000)
- `--agent-type`: Type of agent (`random`, `dqn`)
- `--experiment-name`: Custom name for the experiment

#### Train with MADDPG (CTDE) 🎯 Recommended

Train agents using MADDPG with Centralized Training and Decentralized Execution:

```bash
# Quick test
python train_maddpg.py --episodes 1000 --experiment-name maddpg_test

# Full training
python train_maddpg.py --episodes 10000 --experiment-name maddpg_full
```

**👉 See [MADDPG_QUICKSTART.md](MADDPG_QUICKSTART.md) for complete guide!**

### Evaluation

#### Evaluate DQN Agents

Evaluate trained agents:

```bash
python evaluate.py --checkpoint-dir checkpoints/my_experiment_1000
```

**Evaluation options:**

```bash
python evaluate.py --checkpoint-dir checkpoints/my_experiment_1000 --episodes 100 --render --save-video
```

#### Evaluate MADDPG Agents 🎯

```bash
# Basic evaluation
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_test_1000 --episodes 100

# With visualization
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_test_final --render --episodes 10
```

Arguments:

- `--checkpoint-dir`: Directory containing trained agent checkpoints (required)
- `--episodes`: Number of evaluation episodes (default: 100)
- `--agent-type`: Type of agent (`random`, `dqn`) - for evaluate.py
- `--render`: Display episodes during evaluation
- `--save-video`: Save evaluation as video

## Configuration

Modify training parameters in `config/config.py`:

### Environment Configuration

- `num_good`: Number of good agents (default: 1)
- `num_adversaries`: Number of adversaries (default: 3)
- `num_obstacles`: Number of obstacles (default: 2)
- `max_cycles`: Maximum steps per episode (default: 25)
- `continuous_actions`: Use continuous action space (default: False)

### Training Configuration

- `total_episodes`: Total training episodes (default: 10000)
- `learning_rate`: Learning rate for optimizer (default: 1e-4)
- `gamma`: Discount factor (default: 0.99)
- `epsilon_start/end/decay`: Epsilon-greedy exploration parameters

## Agents

### Random Agent

Baseline agent that selects random actions. Useful for comparison.

### DQN Agent

Deep Q-Network implementation with:

- Experience replay buffer
- Target network for stable learning
- Epsilon-greedy exploration
- Soft updates for target network

### Adding New Agents

Create a new agent class inheriting from `BaseAgent`:

```python
from agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def select_action(self, observation, training=True):
        # Your action selection logic
        pass

    def update(self, experience):
        # Your learning logic
        pass
```

## Monitoring Training

Training logs and checkpoints are saved to:

- **Logs**: `logs/<experiment_name>/`
- **Checkpoints**: `checkpoints/<experiment_name>_<episode>/`
- **Videos**: `videos/` (if enabled during evaluation)

View training progress in the logs:

```bash
tail -f logs/<experiment_name>/training.log
```

## Visualization

Plot training curves from saved metrics:

```python
import json
import matplotlib.pyplot as plt
from utils import plot_training_curves

# Load metrics
with open('logs/<experiment_name>/metrics.json', 'r') as f:
    data = json.load(f)

# Plot
plot_training_curves(data['metrics'], save_path='training_curves.png')
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- MPE2 (Multi-Agent Particle Environment)
- Matplotlib
- imageio (optional, for video recording)

## References

- [MPE2 Documentation](https://mpe2.farama.org/)
- [Simple Tag Environment](https://mpe2.farama.org/environments/simple_tag/)
- [PettingZoo](https://pettingzoo.farama.org/) - Multi-agent RL library

## License

This project is for educational purposes.

## Contributing

Feel free to submit issues or pull requests for improvements!

---

**Author**: Your Name  
**Date**: November 2025
