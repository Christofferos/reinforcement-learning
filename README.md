<table>
  <tr>
    <td><img width="418" alt="image" src="https://github.com/user-attachments/assets/37e3f119-abd1-440f-83aa-c2ca473d7e97" /></td>
    <td><img width="418" alt="image" src="https://github.com/user-attachments/assets/65eebfd4-23e8-4efe-94d3-df0c2fbddf35" /></td>
    <td><img width="418" alt="image" src="https://github.com/user-attachments/assets/61f861b1-a9f3-4e11-867e-9caf9a1b334b" /></td>
  </tr>
</table>

Here are some of the projects I have done. Have a look:
1. https://www.youtube.com/shorts/LoGcZkZzLSA
2. https://www.youtube.com/watch?v=_F91ktb_wxU
3. https://www.youtube.com/watch?v=CYF2pNHuyC8


# Reinforcement Learning 🗺️

Here I am, learning everything in the world of "Reinforcement Learning" 🦾🤖.

My journey through this jungle 🏝️ will be shared to give you a map 🗺️. A map which leads to Eldorado (or Shambala if you are an Uncharted fan). Follow along on this exhilerating journey. And maybe together we can find this city of gold 💰🏆.

I share my mistakes and what I have learnt from them listed here: `mistakes_learnings.md`.

## Get started

1. `python3 -m venv .venv` standing at `./gymnasium`
2. Activate a python virtual environment:

   - Mac: `source .venv/bin/activate`
   - Windows: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` then `.\.venv\Scripts\Activate.ps1`

3. `pip install "stable-baselines3[extra]" "gymnasium[mujoco]" sb3_contrib "gymnasium[box2d]" shapely numpy`
4. Run a train file example: `python walk_train.py TQC -t`
5. Monitor with Tensorflow: `tensorboard --logdir logs`
6. Compare algorithms on: http://localhost:6006/

Algorithms supported by Stable-Baselines3:
https://stable-baselines3.readthedocs.io/en/master/guide/algos.html

## Mistakes and lessons learnt

#1.
Using Tensorflow was a game changer for monitoring algorithmic performance. I can compare algorithms and can see when they fizzle out in improvement.

#2.
I had forgotten to turn on friction on the ground when training the HumanoidDrunkWalk agent. And I wondered why it just kept falling for 1-2 days hahah. Until I notice during evaluation that it never gets friction on its feet. But it pushed me to research more and lead me to Tensorflow and other online learning resources.

#3.
Re-train on previously trained models instead of starting from scratch every time. The agent can learn skills in steps. Balance, walk, turning, stability during interference, jogging, running.

#4.
Make sure parameters are the same in training as in evaluation. I used a field called `frame_skip` to smooth out rendering in evaluation. Took me 2 weeks until releasing it heavily affects the model´s ability to function properly.
