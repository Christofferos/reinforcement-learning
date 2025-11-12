# 🎬 Rendering & Visualization Guide

## Issue: Nothing Shows When Using `--render`

### Problem
When you run:
```bash
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_test_final --render --episodes 100
```

You might not see a visualization window open.

### Solution ✅ FIXED!

I've updated both `evaluate.py` and `evaluate_maddpg.py` to use `render_mode='human'` which opens a pygame window showing the environment.

---

## 🎥 Rendering Options

### Option 1: Live Visualization (Human Mode) ⭐ Recommended

Opens a pygame window showing agents moving in real-time:

```bash
# For MADDPG
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_test_final --render --episodes 10

# For DQN
python evaluate.py --checkpoint-dir checkpoints/dqn_test_1000 --agent-type dqn --render --episodes 10
```

**What you'll see:**
- Red circles = Adversaries (predators)
- Green circle = Good agent (prey)
- Black circles = Obstacles
- Real-time movement and interactions

**Tips:**
- Use fewer episodes (10-20) for viewing
- Each episode runs for max 25 steps
- Window closes automatically between episodes

### Option 2: Save as Video

Records episodes and saves as MP4:

```bash
# For MADDPG
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_test_final --save-video --episodes 50

# For DQN
python evaluate.py --checkpoint-dir checkpoints/dqn_test_1000 --agent-type dqn --save-video --episodes 50
```

**Output location:**
- Saved to `videos/` directory
- Filename: `evaluation_maddpg_<checkpoint_name>.mp4`

**Requirements:**
```bash
pip install imageio[ffmpeg]
```

### Option 3: No Rendering (Fastest)

Just evaluate performance metrics:

```bash
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_test_final --episodes 100
```

---

## 🖼️ What the Visualization Shows

### Environment Layout
```
┌─────────────────────────────┐
│                             │
│    ⚫ Obstacle              │
│                             │
│  🔴 Adversary    🟢 Good    │
│                             │
│         ⚫ Obstacle         │
│                             │
│  🔴 Adversary    🔴 Adversary│
│                             │
└─────────────────────────────┘
```

### Color Coding
- 🔴 **Red** = Adversaries (trying to catch green)
- 🟢 **Green** = Good agent (trying to escape)
- ⚫ **Black** = Obstacles (blocking paths)

### What to Look For

**Good Performance:**
- Adversaries coordinate to surround good agent
- Good agent tries to maintain distance
- Strategic use of obstacles as shields

**Poor Performance (early training):**
- Random, uncoordinated movement
- Adversaries don't work together
- Good agent doesn't actively avoid

---

## 🐛 Troubleshooting Rendering

### Issue 1: "pygame.error: video system not initialized"

**Cause:** Running on a system without display (e.g., SSH, headless server)

**Solution:**
```bash
# Use video saving instead
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_test_final --save-video --episodes 50
```

### Issue 2: Window Opens But Nothing Happens

**Cause:** Episodes run too fast to see

**Solution:** Already fixed! Added 0.05 second delay per frame.

If still too fast, you can increase the delay in the code:
```python
# In evaluate_maddpg.py, line ~135
time.sleep(0.1)  # Change from 0.05 to 0.1 or higher
```

### Issue 3: "ModuleNotFoundError: No module named 'pygame'"

**Cause:** pygame not installed

**Solution:**
```bash
pip install pygame
```

### Issue 4: Window Closes Immediately

**Cause:** Episodes complete too quickly (25 steps max)

**Solution:**
- This is normal - each episode is short
- Run multiple episodes to see more: `--episodes 20`
- Episodes automatically cycle

### Issue 5: Black Screen / Frozen Window

**Cause:** macOS display issues

**Solution:**
```bash
# Set backend explicitly
export PYGAME_HIDE_SUPPORT_PROMPT=1

# Then run
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_test_final --render --episodes 10
```

---

## 📊 Comparing Rendering Modes

| Mode | Command Flag | Use Case | Speed | Output |
|------|-------------|----------|-------|--------|
| None | (no flag) | Performance eval | ⚡⚡⚡ Fast | Terminal stats only |
| Human | `--render` | Visual debugging | 🐢 Slow | Pygame window |
| Video | `--save-video` | Sharing/analysis | 🐢 Slow | MP4 file |

---

## 💡 Best Practices

### For Quick Testing
```bash
# Just check if it works
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_test_100 --render --episodes 5
```

### For Performance Evaluation
```bash
# No rendering for accurate timing
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_test_final --episodes 100
```

### For Presentations/Papers
```bash
# Save high-quality video
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_test_final --save-video --episodes 50
```

### For Debugging Agent Behavior
```bash
# Watch with human mode, fewer episodes
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_test_500 --render --episodes 10
```

---

## 🎬 Example Workflow

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Quick visual check (5 episodes)
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_test_final --render --episodes 5

# 3. Full performance evaluation (no render)
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_test_final --episodes 100

# 4. Save video for sharing
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_test_final --save-video --episodes 30
```

---

## 🔧 Advanced: Custom Rendering

If you want even slower/more controlled rendering, you can modify the evaluation scripts:

### Slower Playback
```python
# In evaluate_maddpg.py, around line 135
elif render:
    try:
        env.render()
        import time
        time.sleep(0.2)  # 5x slower (5 FPS instead of 20 FPS)
    except:
        pass
```

### Frame-by-Frame Control
```python
# Add after env.render()
input("Press Enter for next step...")  # Manual control
```

---

## ✅ Quick Fix Checklist

If rendering doesn't work:
- [ ] Virtual environment activated
- [ ] pygame installed: `pip list | grep pygame`
- [ ] Using `--render` flag
- [ ] Using fewer episodes (10-20)
- [ ] Not running over SSH/headless
- [ ] Updated evaluation scripts (run `git pull` or check timestamps)

---

## 📝 Summary

**The fix is applied!** Now when you use `--render`:
- ✅ Uses `render_mode='human'` 
- ✅ Opens pygame visualization window
- ✅ Shows agents moving in real-time
- ✅ Includes small delay for viewing (0.05s per step)

**Just run:**
```bash
python evaluate_maddpg.py --checkpoint-dir checkpoints/maddpg_test_final --render --episodes 10
```

You should now see the visualization window! 🎉
