# 🚀 Quick Setup Guide for macOS

## Step 1: Set Up Virtual Environment

You're already using `pyenv` with Python 3.10.10 - excellent choice! Now create a virtual environment:

### Option A: Using venv (Recommended - Simple)

```bash
# Make sure you're in the project directory
cd /Users/kriwer/Documents/Github/reinforcement-learning/MARL/MPE2

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate

# You should see (venv) in your prompt
```

### Option B: Using pyenv-virtualenv (Advanced)

```bash
# Install pyenv-virtualenv if you haven't
brew install pyenv-virtualenv

# Create environment
pyenv virtualenv 3.10.10 simple_tag

# Activate it
pyenv activate simple_tag

# Auto-activate when entering directory (optional)
echo "simple_tag" > .python-version
```

---

## Step 2: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Expected installation time:** 2-5 minutes

### If MPE2 installation fails:

```bash
# Try installing dependencies separately
pip install numpy torch gymnasium matplotlib

# Install MPE2 from GitHub directly
pip install git+https://github.com/Farama-Foundation/MPE2.git
```

---

## Step 3: Verify Setup

```bash
# Run the test script
python test_setup.py
```

You should see:

```
✓ MPE2 library imported successfully
✓ Config loaded successfully
✓ Agents module loaded successfully
✓ Environment created successfully
✓ Environment reset successful
✓ Environment step successful
✓ Environment closed successfully
✓ Random agent created successfully
✓ Agent action selection successful

============================================================
All tests passed! ✓
============================================================
```

---

## Step 4: Start Training!

### Quick Test (Random Agents)

```bash
python train.py --agent-type random --episodes 100
```

### Train with DQN

```bash
python train.py --agent-type dqn --episodes 5000 --experiment-name dqn_test
```

### Train with MADDPG (CTDE) 🎯

```bash
# You'll need to create train_maddpg.py (see ALGORITHMS.md)
# Or modify train.py to support MADDPG
```

### Train with MAPPO (CTDE) 🎯

```bash
# You'll need to create train_mappo.py (see ALGORITHMS.md)
# Or modify train.py to support MAPPO
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'mpe2'"

**Solution:**

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # or: pyenv activate simple_tag

# Reinstall mpe2
pip install mpe2

# Or install from source
pip install git+https://github.com/Farama-Foundation/MPE2.git
```

### Issue: "ImportError: No module named 'torch'"

**Solution:**

```bash
# Install PyTorch for macOS
pip install torch torchvision

# For Apple Silicon Mac (M1/M2):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Virtual environment not activating

**Solution:**

```bash
# Check if venv exists
ls -la venv/

# If not, recreate it
python -m venv venv --clear

# Try activating again
source venv/bin/activate
```

### Issue: "zsh: permission denied: python"

**Solution:**

```bash
# Check Python path
which python
python --version

# If using pyenv, make sure it's in PATH
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
source ~/.zshrc
```

---

## Verifying Your Setup

### Check Python version:

```bash
python --version
# Should show: Python 3.10.10
```

### Check installed packages:

```bash
pip list | grep -E 'numpy|torch|mpe2|gymnasium'
```

Expected output:

```
gymnasium         0.29.x
mpe2              0.1.x
numpy             1.21.x or higher
torch             2.0.x or higher
```

### Test import in Python:

```bash
python -c "import mpe2; import torch; import numpy; print('All imports successful!')"
```

---

## Next Steps

1. ✅ Virtual environment activated
2. ✅ Dependencies installed
3. ✅ Setup verified with `test_setup.py`
4. 📖 Read `ALGORITHMS.md` to choose your algorithm
5. 🏃 Start training!

---

## Recommended Workflow

```bash
# 1. Activate environment (do this every time you start working)
source venv/bin/activate

# 2. Run a quick test
python test_setup.py

# 3. Train your agents
python train.py --agent-type dqn --episodes 1000

# 4. Evaluate results
python evaluate.py --checkpoint-dir checkpoints/dqn_TIMESTAMP_1000

# 5. When done, deactivate
deactivate
```

---

## Pro Tips for macOS

1. **Add alias to ~/.zshrc for quick activation:**

   ```bash
   echo 'alias activate_simple_tag="cd ~/Documents/Github/reinforcement-learning/MARL/MPE2 && source venv/bin/activate"' >> ~/.zshrc
   source ~/.zshrc
   ```

2. **Use iTerm2 for better terminal experience**

3. **Monitor GPU usage (if you have GPU):**

   ```bash
   # Check if PyTorch sees MPS (Metal Performance Shaders)
   python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
   ```

4. **Keep requirements.txt updated:**
   ```bash
   pip freeze > requirements.txt
   ```

---

**Happy Training! 🚀**

If you encounter any issues, check the Troubleshooting section or open an issue on GitHub.
