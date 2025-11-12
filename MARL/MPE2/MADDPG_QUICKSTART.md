# 🎯 MADDPG Quick Reference

## One-Liner to Start Training

```bash
source venv/bin/activate && python train_maddpg.py --episodes 1000 --experiment-name maddpg_test
```

## Files Created for You

✅ `train_maddpg.py` - Training script  
✅ `evaluate_maddpg.py` - Evaluation script  
✅ `MADDPG_GUIDE.md` - Complete guide  
✅ `agents/maddpg_agent.py` - Algorithm (already existed)

## What to Read (In Order)

### 1. **MADDPG_GUIDE.md** ⭐ START HERE

- Complete training guide
- Hyperparameter explanations
- Example commands
- Troubleshooting

### 2. **ALGORITHMS.md** (Section on MADDPG)

- Algorithm comparison
- When to use MADDPG vs MAPPO
- Code examples

### 3. **agents/maddpg_agent.py** (Optional)

- Implementation details
- Network architectures
- If you want to understand the internals

## Training Commands

### Quick Test (10 minutes)

```bash
python train_maddpg.py --episodes 100 --experiment-name quick_test
```

### Short Training (30 minutes)

```bash
python train_maddpg.py --episodes 1000 --experiment-name maddpg_1k
```

### Full Training (2-3 hours)

```bash
python train_maddpg.py --episodes 10000 --experiment-name maddpg_full
```

## Evaluation Commands

### Basic Evaluation

```bash
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_1k_1000 --episodes 100
```

### With Visualization

```bash
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_1k_final --render --episodes 10
```

### Save as Video

```bash
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_1k_final --save-video --episodes 50
```

## Key Differences from Regular Training

| Feature       | Regular `train.py` | MADDPG `train_maddpg.py` |
| ------------- | ------------------ | ------------------------ |
| Algorithm     | DQN (Independent)  | MADDPG (CTDE)            |
| Critic        | Local              | **Centralized** ✨       |
| Actor         | Local              | Local (Decentralized)    |
| Replay Buffer | Per agent          | **Shared** ✨            |
| Coordination  | None               | **Learned** ✨           |
| Best for      | Simple tasks       | **Adversarial** ✨       |

## Expected Training Output

```
============================================================
MADDPG Setup
============================================================
Number of agents: 4
Agent names: ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0']
Total observation dim: 62
Total action dim: 20
============================================================

Warmup phase: 1000 steps with random actions
Warmup: 10 episodes, 250/1000 steps
Warmup: 20 episodes, 500/1000 steps
...
Warmup completed! Replay buffer size: 1000

Starting main training loop...

Episode 10/1000 | Total: -5.20 | Adversary: 15.30 | ...
Episode 20/1000 | Total: -8.40 | Adversary: 18.60 | ...
...
✓ Saved checkpoint at episode 100
```

## Checkpoints Structure

```
checkpoints/
├── maddpg_test_100/
│   ├── adversary_0.pth
│   ├── adversary_1.pth
│   ├── adversary_2.pth
│   └── agent_0.pth
├── maddpg_test_200/
│   └── ...
└── maddpg_test_final/
    └── ...
```

## Common Issues & Quick Fixes

| Issue         | Fix                                           |
| ------------- | --------------------------------------------- |
| Import errors | `source venv/bin/activate`                    |
| Slow training | Reduce episodes for testing                   |
| Not learning  | Check warmup completed, increase warmup steps |
| Out of memory | Reduce buffer_size or batch_size              |

## Key Hyperparameters

```python
# In train_maddpg.py - edit if needed
MADDPG_CONFIG = {
    'lr_actor': 1e-4,      # Decrease if unstable
    'lr_critic': 1e-3,     # Usually higher than actor
    'noise_scale': 0.2,    # Increase for more exploration
    'warmup_steps': 1000,  # Increase if not learning
    'batch_size': 64,      # Decrease if slow
}
```

## Performance Benchmarks

| Episodes | Time   | Adversary Reward | Good Agent Reward |
| -------- | ------ | ---------------- | ----------------- |
| 100      | 5 min  | +10 to +20       | -10 to -20        |
| 1000     | 30 min | +30 to +50       | -30 to -50        |
| 5000     | 1.5 hr | +50 to +70       | -50 to -70        |
| 10000    | 3 hr   | +60 to +80       | -60 to -80        |

## Workflow

1. **Activate environment**: `source venv/bin/activate`
2. **Train**: `python train_maddpg.py --episodes 1000`
3. **Monitor**: Watch terminal or check `logs/`
4. **Evaluate**: `python evaluate_maddpg.py --checkpoint-dir checkpoints/...`
5. **Compare**: Try different checkpoints, compare with DQN

## Next Steps

After successful MADDPG training:

- [ ] Compare with DQN (baseline)
- [ ] Try MAPPO (`train_mappo.py` - to be created)
- [ ] Tune hyperparameters
- [ ] Visualize learned behaviors
- [ ] Try different MPE2 environments

---

**Start now:**

```bash
source venv/bin/activate
python train_maddpg.py --episodes 1000 --experiment-name my_first_maddpg
```
