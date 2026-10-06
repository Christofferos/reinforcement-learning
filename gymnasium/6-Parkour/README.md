# 6-Parkour — procedural parkour skill sequencing (MuJoCo)

A second attempt at parkour after [5-HumanoidParkour](../5-HumanoidParkour), this time
with **procedurally generated courses** instead of one hand-authored track, and with a
config-driven MDP so every design choice can be ablated.

The structure mirrors [sjtumrgx/unitree_rl_mjlab](https://github.com/sjtumrgx/unitree_rl_mjlab)
(`Flat` / `Rough` / `Parkour` task lanes, `mdp/` term modules, `train` → `play` scripts),
but it runs on plain **MuJoCo + Gymnasium + Stable-Baselines3** so it works on macOS.
`mjlab` itself needs CUDA + `mujoco-warp` and is Linux/NVIDIA only.

The robot is the same textured humanoid as `5-HumanoidParkour/parkour.xml` — identical
body tree, joint ranges, tendons, actuators and the `assets/` textures (copied over), so
the two projects are visually and dynamically comparable.

## Layout

```
g1_parkour/
  robots.py          # RobotSpec: humanoid (default) + optional Unitree G1 via Menagerie
  terrain/
    core.py          # Box / HeightField / TerrainSpec primitives
    flat.py          # flat plane lane
    rough.py         # fractal-noise heightfield + discrete obstacle blocks
    parkour.py       # 12 parkour skill modules chained into a random course
  mdp/
    observations.py  # obs terms, height scan, noise/latency
    rewards.py       # weighted reward terms
    terminations.py  # termination terms
    events.py        # domain randomisation + pushes
  scene.py           # composes robot + terrain into a compilable MJCF
  env_cfg.py         # TerrainCfg / ParkourEnvCfg dataclasses
  env.py             # ParkourEnv (gymnasium.Env)
  tasks.py           # task + ablation registry
scripts/             # list_envs.py, train.py, play.py, ablate.py, run_curriculum.sh
tests/               # smoke + invariant tests
```

## Tasks

```bash
python scripts/list_envs.py
python scripts/list_envs.py --keyword ablation
```

| Task                    | Terrain                    | Notes                                                   |
| ----------------------- | -------------------------- | ------------------------------------------------------- |
| `Flat`                  | flat plane                 | randomized turns and per-waypoint speed commands        |
| `Rough`                 | noise heightfield + blocks | speed commands, curriculum, friction/mass randomisation |
| `Procedural`            | random skill chain         | curriculum, pushes, friction/mass randomisation         |
| `Parkour-Ablation-*-v0` | random skill chain         | one MDP change each, see below                          |

All three regular presets enable 45 ground-scan debug dots. Press `2` in the
viewer to hide/show them; `4` toggles waypoint labels and the live agent speed.
Debug dots do not add policy inputs. `Flat` and `Rough` have 113 observation
values because of their speed commands; legacy 112-input checkpoints are not
directly compatible with those presets.

`Procedural` also includes three upward terrain rays at forward offsets of
0, 0.5 and 1 m, starting 0.5 m above the floor estimated near the feet.
Their distances are capped at 2 m (also the no-hit value) and follow the
configured observation delay. Cyan dots show current overhead hits; misses
are hidden, and key `2` toggles both scan marker sets. These three policy
inputs make the Procedural observation size 115 instead of 112, so existing
112-input checkpoints require an explicit adapter or a new policy. Set
`ObservationCfg(overhead_scan=False)` to retain the previous layout.
Playback with `scripts/play.py` includes the upward readings by default, matching
the 115-input Procedural training preset. Pass `--no-overhead-observation` for
an older 112-input checkpoint: cyan dots remain active but the three overhead
readings are excluded from the policy inputs. Training presets are unchanged
by this playback option.

### Procedural parkour modules

`run`, `gap`, `stepping_stones`, `hurdle`, `stairs_up`, `stairs_down`, `ramp`, `beam`,
`tunnel` (duck under), `climb`, `lateral_shift` (turn), `rubble`.

A course is `start_pad → N random modules → finish_pad`, with the per-module difficulty
ramping up along the course. `difficulty ∈ [0, 1]` scales gap widths, step heights, beam
narrowness, overhead clearance and terrain noise amplitude. Each module also emits the
route waypoints that the reward and the `waypoint_command` observation follow, so the
policy is trained to _sequence_ skills rather than to memorise one track.

Terrain is regenerated on reset (`TerrainCfg.resample_every_n_resets`) and the difficulty
is auto-adjusted from the fraction of waypoints reached (`TerrainCfg.curriculum`).

## MDP ablations

Each entry is a registered task and a `--ablations` value for `scripts/ablate.py`:

| Ablation                 | What changes                                                    |
| ------------------------ | --------------------------------------------------------------- |
| `blind`                  | remove the height scan (proprioception only)                    |
| `no-waypoints`           | remove the route command — reward-only guidance                 |
| `no-privileged-velocity` | remove base linear velocity + base height from the obs          |
| `noisy-obs`              | sensor noise, 2-step exteroceptive latency, 1-step action delay |
| `obs-history`            | stack 5 control steps                                           |
| `sparse-reward`          | waypoint/goal bonuses only, no shaping                          |
| `no-style-costs`         | drop action-rate / torque / joint-limit / lateral penalties     |
| `energy-penalty`         | add a torque-power penalty                                      |
| `no-early-termination`   | keep episodes alive after falls                                 |
| `no-domain-rand`         | no friction/mass/gear randomisation, no pushes                  |
| `reset-along-course`     | spawn at a random waypoint instead of the start                 |
| `fixed-course`           | one fixed course for the whole run                              |
| `no-curriculum`          | max difficulty from step 0                                      |

```bash
# score one checkpoint under every ablation env -> results/*.csv
python scripts/ablate.py eval --checkpoint models/<run>/final.zip --episodes 20

# train a sweep with a shared budget/seed -> results/*.csv
caffeinate -i python scripts/ablate.py train --ablations blind noisy-obs sparse-reward \
    --timesteps 2_000_000 --num-envs 8
```

## Train / play

Use the `gymnasium/.venv` interpreter (the one `5-HumanoidParkour` uses) — it already has
`mujoco`, `gymnasium`, `stable-baselines3` and `sb3-contrib`:

```bash
cd gymnasium/6-Parkour
source ../.venv/bin/activate
```

The repo also has `.venv` and `MARL/.venv`; those are missing MuJoCo and/or SB3, which is
what causes `ModuleNotFoundError: No module named 'mujoco'`.

Training runs take hours, so prefix them with `caffeinate -i` to keep macOS awake while
the process lives (the display may still sleep; use `caffeinate -di` to keep it on too):

```bash
# flat -> rough -> procedural, each stage resuming the previous checkpoint
caffeinate -i ./scripts/run_curriculum.sh 8

# or one stage at a time
caffeinate -i python scripts/train.py Parkour-Flat-v0 --num-envs 8 --timesteps 5_000_000 \
    --run-name flat
caffeinate -i python scripts/train.py Parkour-Rough-v0 --num-envs 8 --timesteps 10_000_000 \
    --run-name rough --resume models/flat/final.zip
caffeinate -i python scripts/train.py Parkour-Procedural-v0 --num-envs 8 --timesteps 30_000_000 \
    --run-name parkour --resume models/rough/final.zip

# detached, survives closing the terminal, logs to a file
caffeinate -i nohup ./scripts/run_curriculum.sh 8 > curriculum.log 2>&1 &

python scripts/play.py Parkour-Procedural-Play-v0                       # zero actions
python scripts/play.py Parkour-Procedural-v0 --checkpoint models/<run>/final.zip
python scripts/play.py Parkour-Rough-v0 --camera chase_view --episodes 3
```

The three stages are **sequential, not parallel** — each `--resume`s the previous stage's
checkpoint, and running them concurrently would just have them fight over the same cores.
All three lanes share the same observation layout (the flat lane keeps the height scan so
the actor contract matches), which is what makes resuming across them work.

`--num-envs` should stay at or below your core count (`sysctl -n hw.logicalcpu`); on an
8-core Mac use 8. Per-stage budgets can be overridden:
`FLAT_STEPS=2_000_000 ROUGH_STEPS=5_000_000 ./scripts/run_curriculum.sh 8`.

Cameras: `side_view`, `chase_view`, `front_view`, `egocentric`.

### Flat walking with turns and speed changes

`Parkour-FlatLocomotion-v0` samples a new 12 m route each episode: waypoints are
3 m apart along x with lateral offsets in [-1.25, 1.25] m. Each waypoint requests
a speed sampled independently in [0.5, 1.25] m/s. Routes and speeds reproduce under
the same reset seed. Flat resets reuse the physics model instead of recompiling it.

The policy observes the current requested speed and the existing two-waypoint
lookahead. A planar velocity-tracking reward favors the requested speed toward
the active waypoint, with small progress/waypoint bonuses and posture/action
regularization. The first stage has no obstacles, pushes or dynamics randomization;
only mild initial pose/yaw noise. Height/tilt checks retain a 0.15 s recovery window.

This task has 113 observation values instead of 112. Start a new policy, not an
old flat/rough/parkour checkpoint. Its checkpoint must be played in the same task;
direct transfer back to legacy tasks requires an explicit observation adapter.

```bash
caffeinate -i python scripts/train.py Parkour-FlatLocomotion-v0 \
  --run-name flat_locomotion --num-envs 8 --timesteps 10000000 \
  --policy-hidden-sizes 256 256 --ent-coef 0.005 --eval-every 250000

python scripts/play.py Parkour-FlatLocomotion-v0 \
  --checkpoint models/flat_locomotion/final.zip \
  --no-soft-termination --no-until-fall --episodes 5
```

Use `--hours` for a bounded pilot. TensorBoard and seeded evaluation report
`velocity_tracking_error`: mean planar velocity error in m/s over each episode,
alongside survival, completion and success. Reward improvement alone is not proof
of stable walking. Test held-out route seeds before proceeding to rough terrain.
`--policy-hidden-sizes` applies only to fresh PPO policies; `--ent-coef` also applies
on resume. Training still saves only `final.zip` or `interrupted.zip`.

### Rough terrain speed commands

`Rough` samples independent waypoint speed commands in [0.75, 2.0] m/s on
each reset. The current command is included in the policy observation and a
planar velocity-tracking reward trains speed and direction matching. No extra
speed flags are required. Progress rewards are capped and an overspeed penalty
discourages exceeding the command; neither physically clamps actual velocity.

The episode limit is 70 seconds so the 28 m route is achievable at the lowest
requested speed. Terrain curriculum and friction/mass randomization remain enabled.
This preset has 113 observation values, matching the current `Flat` preset;
legacy 112-input rough checkpoints cannot be resumed directly.

```bash
python scripts/train.py Rough --run-name rough_speed --num-envs 8 --timesteps 10000000
```

### Controlled progression training

The procedural task now starts at difficulty 0.15 with a 60-second time limit.
A goal means reaching the final route waypoint with adequate clearance above the
finish platform and upright posture, not merely advancing to it or passing below
it. Completion counts reached waypoints, including the final one;
for mid-course resets it measures the remaining route.

With curriculum enabled, each training worker promotes difficulty by 0.05 only
after collecting 20 episodes at that level with at least 80% successful full-course
finishes. Falls, timeouts, and mid-course starts do not count as successes.
Success history survives normal resets and terrain regeneration, but clears after
a difficulty change. New terrain uses the updated difficulty on the next reset.
The range is [0.1, 1.0]. The existing demotion rule remains: an episode with at most
25% completion lowers difficulty by 0.05, so progression is not necessarily monotonic.
Evaluation disables curriculum and keeps its configured test difficulty fixed.

Progress rewards metres travelled toward the active waypoint rather than giving
a speed reward every step. The procedural configuration uses 20 reward per metre
(progress capped at 1.2 m/s), 20 per intermediate waypoint, and 200 for completion.
Movement above 1.2 m/s receives an additional penalty. Survival/style weights are
smaller so standing still or rushing is not the intended objective. These weights
are experimental; judge them using completion and success, not old reward totals.

Height and tilt violations have a 20-step (0.3-second) recovery window. Torso
clearance may drop to 0.6 m for the humanoid before the height counter starts.
Pit and boundary checks remain active. This is not a fully contact-aware climbing
detector: prolonged low climbing/crouching poses can still terminate. Timeouts now
also update curriculum difficulty, and stall detection uses `stall_distance`.

PPO training defaults to learning rate `1e-4`, `gamma=0.999`, batch size 256,
5 update epochs, and `target_kl=0.02`. The discount horizon is roughly 15 seconds
at this task's control timestep. These options also override checkpoint settings
on resume; CLI flags can change them without editing the script.

```bash
caffeinate -i python scripts/train.py Parkour-Procedural-v0 \
  --resume models/<previous-run>/final.zip \
  --run-name procedural_progression --timesteps 20000000 \
  --eval-every 1000000
```

For a wall-clock budget, add `--hours 10` and use a sufficiently large step ceiling
(for example `--timesteps 1000000000`). The first limit reached stops training.
The timer includes initial and periodic evaluation; an in-progress PPO update or
evaluation can finish after the deadline, followed by final evaluation and saving.
Normal deadline completion writes `final.zip` and `final.state.json`.
Training saves models only on completion (`final.zip`) or Ctrl+C
(`interrupted.zip`), with matching curriculum-state files. No periodic or
best-model checkpoints are saved; `--save-every` is no longer supported.

Each run must use a new folder name. TensorBoard includes `progress/` completion,
success and difficulty, `termination/` failure fractions, and `reward_terms/`
episode totals. Initial, periodic, and final evaluations use 20 fixed seeded
courses at difficulty 0.45 (override with `--eval-difficulty`); results go into
`evaluations.jsonl`. The evaluation keeps training termination rules, not play
overrides.

Checkpoints have matching `.state.json` files that preserve each training worker's
difficulty. Keep these alongside their `.zip` files when resuming; the worker count
must match. Older checkpoints lack this state and start at the configured initial
difficulty. These files do not preserve the full physics state or RNG streams.
`run_config.json` records the task configuration and CLI settings for comparison.

For follow-up experiments, first compare completion on the fixed evaluation set.
Then test single-obstacle courses (steps, gaps, beams), mix start-of-course and
mid-course practice, and add contact-aware fall detection for prolonged climbing.
Do not interpret a short validation rollout as evidence of learned improvement.

## Using the Unitree G1 instead

`robots.py` also ships a `g1` spec. It needs the MuJoCo Menagerie model (meshes are not
vendored here):

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie
export MENAGERIE_PATH=$PWD/mujoco_menagerie
```

Then set `ParkourEnvCfg.robot = "g1"`. The terrain, MDP terms, curriculum and ablations
are robot-agnostic; only the spawn height, healthy height range and foot bodies differ.
The reward/termination limits were tuned for the humanoid, so expect to retune
`nominal_base_height` and `RewardCfg` for G1.

## Tests

```bash
source ../.venv/bin/activate
pip install pytest && python -m pytest tests -q
```

## Reumse training on procedural

```
python scripts/train.py Parkour-Procedural-v0 \
  --num-envs 8 \
  --timesteps 20000000 \
  --run-name procedural_extended \
  --resume models/curriculum_20260930_142501_Parkour-Procedural-v0/final.zip
```
