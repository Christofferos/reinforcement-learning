# Reinforcement Learning 🗺️

Here I am, learning everything in the world of "Reinforcement Learning" 🦾🤖.

My journey through this jungle 🏝️ will be shared to give you a map 🗺️. A map which leads to Eldorado (or Shambala if you are an Uncharted fan). Follow along on this exhilerating journey. And maybe together we can find this city of gold 💰🏆.

I share my mistakes and what I have learnt from them listed here: `mistakes_learnings.md`.

## Get started

1. `python3 -m venv .venv` standing at `./gymnasium`
2. Activate a python virtual environment:

   - Mac: `source .venv/bin/activate`
   - Windows: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` then `.\.venv\Scripts\Activate.ps1`

3. `pip install "stable-baselines3[extra]" "gymnasium[mujoco]" "gymnasium[box2d]" shapely numpy`
4. Run a train file example: `python walk_train.py TQC -t`
5. Monitor with Tensorflow: `tensorboard --logdir logs`
6. Compare algorithms on: http://localhost:6006/

Algorithms supported by Stable-Baselines3:
https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
