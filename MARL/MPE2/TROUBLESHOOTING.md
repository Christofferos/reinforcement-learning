# Common Errors and Solutions

## Error 1: AttributeError: 'NoneType' object has no attribute 'get'

**Error Message:**

```
AttributeError: 'NoneType' object has no attribute 'get'
```

**Cause:**
The `load_agents` function in `evaluate.py` was not providing a default config when creating DQN agents.

**Solution:**
✅ **FIXED!** The `evaluate.py` file has been updated to use `TRAINING_CONFIG` as the default configuration when loading agents.

---

## Error 2: ModuleNotFoundError: No module named 'numpy'

**Error Message:**

```
ModuleNotFoundError: No module named 'numpy'
```

**Cause:**
Virtual environment is not activated, or dependencies are not installed.

**Solution:**

```bash
# Step 1: Activate virtual environment
source venv/bin/activate

# You should see (venv) in your prompt:
# (venv) ➜  MPE2

# Step 2: Verify activation
which python
# Should show: /Users/kriwer/Documents/Github/reinforcement-learning/MARL/MPE2/venv/bin/python

# Step 3: If packages aren't installed, install them
pip install -r requirements.txt
```

---

## Error 3: No checkpoint found

**Error Message:**

```
Warning: No checkpoint found for agent_0 at checkpoints/...
```

**Cause:**
You're trying to evaluate with a checkpoint type that doesn't match the agent type.

**Solution:**
For **random agents**, they don't have checkpoints (they don't learn). Use:

```bash
python evaluate.py --checkpoint-dir checkpoints/random_XXXXX --agent-type random
```

For **trained agents** (DQN, MADDPG, MAPPO), the checkpoint directory should contain `.pth` files:

```bash
python evaluate.py --checkpoint-dir checkpoints/dqn_XXXXX_1000 --agent-type dqn
```

---

## Complete Evaluation Workflow

### For Random Agents (No Training)

```bash
# Activate environment
source venv/bin/activate

# Evaluate random agents (no checkpoint needed, but directory must exist)
python evaluate.py --checkpoint-dir checkpoints/random_XXXXX --agent-type random --episodes 10
```

### For Trained Agents (DQN, MADDPG, MAPPO)

```bash
# Activate environment
source venv/bin/activate

# First, train agents
python train.py --agent-type dqn --episodes 1000 --experiment-name my_test

# Then evaluate (use checkpoint directory created during training)
python evaluate.py --checkpoint-dir checkpoints/my_test_1000 --agent-type dqn --episodes 100

# Optional: Render and save video
python evaluate.py --checkpoint-dir checkpoints/my_test_1000 --agent-type dqn --render --save-video
```

---

## Quick Checklist Before Running evaluate.py

- [ ] Virtual environment is activated (`source venv/bin/activate`)
- [ ] Dependencies are installed (`pip list | grep numpy torch mpe2`)
- [ ] Checkpoint directory exists and matches agent type
- [ ] Agent type flag matches the training algorithm used

---

## Testing Your Setup

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Verify setup
python test_setup.py

# 3. Train a quick test
python train.py --agent-type dqn --episodes 100 --experiment-name quick_test

# 4. Evaluate
python evaluate.py --checkpoint-dir checkpoints/quick_test_100 --agent-type dqn --episodes 10
```

---

## Still Having Issues?

1. **Check virtual environment:**

   ```bash
   which python
   # Should show: .../venv/bin/python
   ```

2. **Check installed packages:**

   ```bash
   pip list | grep -E "numpy|torch|mpe2|gymnasium"
   ```

3. **Reinstall if needed:**

   ```bash
   pip install --force-reinstall -r requirements.txt
   ```

4. **Check checkpoint directory contents:**
   ```bash
   ls -la checkpoints/your_checkpoint_dir/
   # Should show .pth files for trained agents
   ```
