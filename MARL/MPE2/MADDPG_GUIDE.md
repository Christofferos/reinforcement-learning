# 🚀 MADDPG Training Guide

## What is MADDPG?

**MADDPG** (Multi-Agent Deep Deterministic Policy Gradient) is a CTDE (Centralized Training with Decentralized Execution) algorithm perfect for the Simple Tag environment.

### Key Features:

- ✅ **Centralized Critic**: Sees all agents' observations and actions during training
- ✅ **Decentralized Actors**: Each agent acts independently using only local observations
- ✅ **Perfect for Simple Tag**: Handles adversarial predator-prey scenarios
- ✅ **Experience Replay**: Shared replay buffer for all agents
- ✅ **Soft Updates**: Stable learning with target networks

---

## 📁 Files You Need

I've created these files for you:

1. **`agents/maddpg_agent.py`** ✅ - Algorithm implementation (already exists)
2. **`train_maddpg.py`** ✅ - Training script (just created!)
3. **`evaluate_maddpg.py`** ✅ - Evaluation script (just created!)

---

## 🎯 Quick Start

### Step 1: Activate Environment

```bash
source venv/bin/activate
```

### Step 2: Train MADDPG (Quick Test)

```bash
# Quick test with 1000 episodes
python train_maddpg.py --episodes 1000 --experiment-name maddpg_test
```

### Step 3: Full Training

```bash
# Full training (10,000 episodes, ~2-3 hours)
python train_maddpg.py --episodes 10000 --experiment-name maddpg_full
```

### Step 4: Evaluate

```bash
# Evaluate the trained model
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_test_1000 --episodes 100
```

---

## 📚 Understanding MADDPG Training

### Training Process

```
1. WARMUP PHASE (1000 steps)
   ├── Random actions to fill replay buffer
   └── No learning, just exploration

2. TRAINING PHASE (per episode)
   ├── For each step:
   │   ├── Actors select actions (with noise for exploration)
   │   ├── Environment steps
   │   ├── Store experience in shared replay buffer
   │   └── Update all agents using centralized critics
   └── Decay exploration noise over time

3. CHECKPOINTING
   └── Save models every 100 episodes
```

### What Makes MADDPG Special for Simple Tag?

```
During Training (Centralized):
┌────────────────────────────────────────┐
│  Adversary 0 Critic                    │
│  ┌──────────────────────────────────┐  │
│  │ ALL agents' observations         │  │
│  │ ALL agents' actions              │  │
│  │ → Can learn to coordinate!       │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘

During Execution (Decentralized):
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Adversary 0 │  │ Adversary 1 │  │ Good Agent  │
│ Only sees   │  │ Only sees   │  │ Only sees   │
│ local obs   │  │ local obs   │  │ local obs   │
└─────────────┘  └─────────────┘  └─────────────┘
```

---

## ⚙️ Configuration

### Key Hyperparameters in `train_maddpg.py`:

```python
MADDPG_CONFIG = {
    'gamma': 0.95,              # Discount factor
    'tau': 0.01,                # Soft update rate
    'lr_actor': 1e-4,           # Actor learning rate
    'lr_critic': 1e-3,          # Critic learning rate (higher)
    'hidden_dim': 128,          # Neural network size
    'noise_scale': 0.2,         # Exploration noise
    'noise_decay': 0.9999,      # Noise decay per step
    'min_noise': 0.01,          # Minimum noise
    'buffer_size': 100000,      # Replay buffer size
    'batch_size': 64,           # Mini-batch size
    'warmup_steps': 1000,       # Random exploration steps
}
```

### Tuning Tips:

| Parameter     | Lower Value           | Higher Value          |
| ------------- | --------------------- | --------------------- |
| `lr_actor`    | More stable, slower   | Faster, less stable   |
| `lr_critic`   | More stable           | Faster value learning |
| `noise_scale` | Less exploration      | More exploration      |
| `gamma`       | Short-term rewards    | Long-term rewards     |
| `tau`         | Slower target updates | Faster target updates |

---

## 📊 Expected Results

### Simple Tag Performance (10,000 episodes):

```
Episode 1000:
  Adversary Reward: +30 to +50
  Good Agent Reward: -30 to -50

Episode 5000:
  Adversary Reward: +50 to +70
  Good Agent Reward: -50 to -70

Episode 10000:
  Adversary Reward: +60 to +80
  Good Agent Reward: -60 to -80
```

**Note**: Adversaries learn to coordinate and catch the good agent more effectively!

---

## 🔧 Command Line Options

### Training:

```bash
python train_maddpg.py [OPTIONS]

Options:
  --episodes N          Number of episodes (default: 10000)
  --experiment-name S   Experiment name (default: maddpg_TIMESTAMP)
  --warmup N           Warmup steps (default: 1000)
```

### Evaluation:

```bash
python evaluate_maddpg.py [OPTIONS]

Options:
  --checkpoint-dir S   Checkpoint directory (required)
  --episodes N         Number of episodes (default: 100)
  --render            Show visualization
  --save-video        Save as video
```

---

## 📝 Example Training Session

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Train MADDPG
python train_maddpg.py --episodes 5000 --experiment-name my_maddpg

# Expected output:
# ============================================================
# MADDPG Setup
# ============================================================
# Number of agents: 4
# Agent names: ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0']
# ...
# Warmup phase: 1000 steps with random actions
# ...
# Episode 100/5000 | Total: -10.50 | Adversary: 20.30 | ...

# 3. Checkpoints are saved every 100 episodes:
# checkpoints/my_maddpg_100/
# checkpoints/my_maddpg_200/
# ...
# checkpoints/my_maddpg_5000/
# checkpoints/my_maddpg_final/

# 4. Evaluate at different checkpoints
python evaluate_maddpg.py --checkpoint-dir checkpoints/my_maddpg_1000 --episodes 50
python evaluate_maddpg.py --checkpoint-dir checkpoints/my_maddpg_5000 --episodes 100

# 5. Render and visualize
python evaluate_maddpg.py --checkpoint-dir checkpoints/my_maddpg_final --render --episodes 10
```

---

## 🐛 Troubleshooting

### Issue: "Import errors"

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Check if packages are installed
pip list | grep -E "torch|numpy|mpe2"
```

### Issue: "CUDA out of memory" (if using GPU)

```python
# MADDPG runs fine on CPU for Simple Tag
# No GPU needed for this environment
```

### Issue: "Agents not learning"

**Possible causes:**

1. Warmup steps too low → Increase to 2000-5000
2. Learning rates too high → Decrease both by 10x
3. Batch size too large → Try 32 instead of 64
4. Noise decay too fast → Use 0.99995 instead of 0.9999

### Issue: "Training is very slow"

**Solutions:**

1. Reduce `buffer_size` to 50000
2. Reduce `batch_size` to 32
3. Update less frequently (every 2-4 steps)
4. Use fewer episodes for testing first

---

## 📈 Monitoring Training

### During Training:

- Watch the terminal for episode statistics
- Check `logs/maddpg_TIMESTAMP/training.log`
- Adversary rewards should increase
- Good agent rewards should decrease (more negative)

### After Training:

```bash
# View training log
cat logs/maddpg_TIMESTAMP/training.log | tail -100

# Check saved metrics
cat logs/maddpg_TIMESTAMP/metrics.json
```

---

## 🎓 What to Read/Learn

### 1. **ALGORITHMS.md** (In your project)

- Complete MADDPG overview
- Comparison with other algorithms
- Code examples

### 2. **Original MADDPG Paper**

- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
- Authors: Lowe et al., 2017

### 3. **Key Concepts**

- Actor-Critic methods
- Policy gradients
- Experience replay
- Centralized training vs decentralized execution

---

## 🚀 Next Steps After MADDPG

1. **Compare with other algorithms:**

   ```bash
   # Train MAPPO
   python train_mappo.py --episodes 5000

   # Compare results
   ```

2. **Experiment with hyperparameters:**

   - Try different learning rates
   - Adjust exploration noise
   - Change network architecture

3. **Try other MPE2 environments:**

   - Simple Adversary
   - Simple Spread
   - Simple Push

4. **Visualize results:**
   - Plot training curves
   - Create videos of learned policies
   - Analyze agent behaviors

---

## ✅ Quick Checklist

Before training:

- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip list | grep mpe2`)
- [ ] Reviewed `MADDPG_CONFIG` in `train_maddpg.py`
- [ ] Chosen experiment name

During training:

- [ ] Monitor terminal output
- [ ] Check logs periodically
- [ ] Watch for increasing adversary rewards

After training:

- [ ] Evaluate multiple checkpoints
- [ ] Compare with baseline (random, DQN)
- [ ] Visualize with `--render`

---

## 💡 Pro Tips

1. **Start small**: Test with 1000 episodes first
2. **Save checkpoints**: Every 100 episodes for comparison
3. **Monitor closely**: First 1000 episodes are critical
4. **Expect slow start**: Warmup phase is necessary
5. **Be patient**: Full training takes 2-3 hours

---

**Ready to train? Run:**

```bash
python train_maddpg.py --episodes 1000 --experiment-name my_first_maddpg
```

Good luck! 🎯
