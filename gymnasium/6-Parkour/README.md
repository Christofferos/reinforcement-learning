## Train / play

```bash
cd gymnasium/6-Parkour
source ../.venv/bin/activate
```

One stage at a time:

```bash
1.
caffeinate -i python scripts/train.py Flat --hours 1 --run-name flat_0.0.1

2.
caffeinate -i python scripts/train.py Rough --hours 1 --run-name rough_0.0.1 --resume models/flat_0.0.1/final.zip

3.
caffeinate -i python scripts/train.py Procedural --hours 1 --run-name procedural_0.0.1 --resume models/rough_0.0.1/final.zip

4.
python scripts/play.py Flat                       # random actions
python scripts/play.py Procedural --checkpoint models/<run>/final.zip
python scripts/play.py Procedural --checkpoint models/<run>/final.zip --difficulty 0.3
```

Playback runs at one fixed difficulty with the curriculum off. It uses `--difficulty` when
given, otherwise the mean level saved in the checkpoint's `.state.json`, otherwise the task's
starting difficulty, and prints which one it picked.

Optional (all three at once):

```bash
# flat -> rough -> procedural, each stage resuming the previous checkpoint
caffeinate -i ./scripts/all_curriculum.sh 8
```

NOTE:

```bash
V0 trained using all_curriculum & below timesteps, result = walking:
A)
Flat --timesteps 5_000_000
B)
Rough --timesteps 10_000_000
C)
Procedural --timesteps 30_000_000
```

### Tensorflow monitoring:

```bash
python -m tensorboard.main --logdir logs --host 127.0.0.1 --port 6006
```

Cameras: `side_view`, `chase_view`, `front_view`, `egocentric`.

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
the two projects are visually and dynamically comparable. Two joint values are equalised so
the robot is exactly left/right symmetric, which the mirror symmetry training relies on: the
right `hip_y` armature is 0.01 like the left (0.008 in project 5), and the left knee has no
spring, like the right (stiffness 1 in project 5).

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
  symmetry.py        # left/right mirror maps for observations and actions
  symmetric_ppo.py   # PPO with a mirror loss and/or mirrored minibatches
  tasks.py           # task + ablation registry
scripts/             # list_envs.py, train.py, play.py, ablate.py, repair_state.py, all_curriculum.sh
tests/               # smoke + invariant tests
```

## Tasks

```bash
python scripts/list_envs.py
python scripts/list_envs.py --keyword ablation
```

| Task                    | Terrain                    | Notes                                                   |
| ----------------------- | -------------------------- | ------------------------------------------------------- |
| `Flat`                  | flat plane                 | curriculum: faster commands, wider turns, longer routes |
| `Rough`                 | noise heightfield + blocks | speed commands, curriculum, friction/mass randomisation |
| `Procedural`            | random skill chain         | curriculum, pushes, friction/mass randomisation         |
| `Parkour-Ablation-*-v0` | random skill chain         | one MDP change each, see below                          |

All three regular presets enable 45 ground-scan debug dots. Press `2` in the
viewer to hide/show them; `4` toggles waypoint labels and the live agent speed.
Debug dots do not add policy inputs. `Flat` and `Rough` have 113 observation
values because of their speed commands; legacy 112-input checkpoints are not
directly compatible with those presets.

`Procedural` also includes three upward terrain rays at forward offsets of
0, 0.2 and 0.4 m, starting 0.5 m above the floor estimated near the feet.
Their distances are capped at 2 m (also the no-hit value) and follow the
configured observation delay. The spacing is below the 0.24 m thickness of the tunnel bar,
so the bar is always seen from 0.52 m ahead until it is 0.12 m behind the torso. The earlier
0 / 0.5 / 1 m layout looked further ahead but missed the bar entirely while its centre was
0.13-0.38 m or 0.63-0.88 m ahead. A ray that starts inside a terrain box (a hurdle or climb
taller than 0.5 m, or stairs rising more than that within the ray's reach) reads as a miss
instead of reporting the box's top face as a low ceiling. That false ceiling also lowered the
`base_height` target, asking the robot to crouch before climbs. The layout keeps three
readings, so the observation size is unchanged, but a Procedural policy trained on the old
offsets sees different values and needs to adapt. Flat and Rough have no ceilings or
obstacles taller than 0.5 m, so their readings stay at 2 m either way. Cyan dots show current overhead hits; misses
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

### Flat walking with turns and speed changes

#### Control, gait and normalisation (v0.1 — breaks old checkpoints)

Diagnosis of `flat_0.0.4/0.0.5`: 100 % `bad_height` endings after ~2 s, ~85 % of the
return from `velocity_tracking` earned while leaning forward on one foot. The fixes:

- **Joint-position PD control** (`ParkourEnvCfg.control_mode="position"`, default). Actions
  are target offsets (`position_action_scale` rad) around a standing pose
  (`RobotSpec.default_joint_pos`, hips −18.5°, knees −30°, CoM over the feet). A per-substep
  PD loop (`kp = 2 × torque limit`, `kd = 0.05 kp`) converts them to motor commands clipped
  at the gear limits, so the zero action means "stand" instead of "no torque". Resets use
  that pose and drop the robot until the lowest foot is 1 cm above the terrain.
  `control_mode="torque"` restores the old mapping.
- **Gait rewards**: `feet_air_time` (biped single-stance time, capped at 0.4 s, only when a
  speed is commanded), `feet_step_ahead` (touchdown reward for alternating steps that land
  ahead of the stance foot along the direction to the waypoint, negative when landing behind,
  nothing for a repeated hop on one foot; a skip that always leads with the same leg nets zero
  per stride), `feet_step_symmetry` (penalty per metre of difference between consecutive left
  and right step lengths, so a gallop or limp pays on every step) and `feet_slide` (planar
  speed of feet in contact). Steps used to be measured on the torso heading, and `flat_0.0.9`
  learned to exploit that: it galloped with the left foot leading 0.56 m and the right foot
  catching up 0.19 m behind, torso turned 33° so the catch-up step projected to about zero.
  `flat_0.0.10` (resumed from 0.0.9 with the travel-direction steps and the mirror loss)
  narrowed that to a skip: left steps 0.30 m ahead, right steps land level (−0.02 m). With
  `feet_step_ahead` saturating at a 0.15 m step and `feet_step_symmetry` at −5, that skip
  still netted about +1.1 per stride. The shared weights now saturate at 0.3 m
  (`feet_step_ahead_margin`) and charge −10 per metre of asymmetry, so the skip nets −1.7, the
  0.0.9 gallop −13.2 and an even 0.3 m walk +10 per stride.
- **Support gating** (`RewardCfg.gate_on_support`): `progress` and `velocity_tracking` pay
  nothing while height or tilt is outside the termination band, so a slow fall earns nothing.
- **Zero-command episodes** (`zero_command_prob`, Flat 0.2, Rough 0.1): all waypoint speeds
  are 0, standing still is the task, stall termination is suspended and the episode does not
  affect the curriculum.
- **Recovery window** on Flat/Rough: `bad_height_scale=0.75` (torso ≥ 0.71 m) and
  `recovery_grace_steps=20`, matching Procedural.
- **Training defaults**: `VecNormalize` on observations and rewards (`--no-normalize` to
  disable; stats saved as `<checkpoint>.vecnormalize.pkl` and restored on `--resume`, used
  automatically by `play.py`/`ablate.py eval`), fresh policies get `256 256` layers and
  `--log-std-init -1`, `--ent-coef` defaults to 0.002, and PPO trains with a left/right
  mirror loss plus mirrored minibatches (`--symmetry both`, see
  [Left/right mirror symmetry](#leftright-mirror-symmetry)).

Action semantics changed, so every stage needs a **fresh** policy; do not `--resume` the
`flat_0.0.x` torque-control checkpoints. Suggested first run:

```bash
caffeinate -i python scripts/train.py Flat --run-name flat_0.1.0 --hours 3 --gamma 0.99
```

Watch `reward_terms_per_step/feet_air_time`, `termination/bad_height` and
`rollout/ep_len_mean`; stepping has started when air-time reward rises while episode
length grows beyond ~150 steps.

`Flat` samples a new route each episode. Difficulty `d` in [0, 1] sets the
course length to `12 + 6*d` metres and lateral offsets within
`+/- (1.25 + 1.25*d)` metres. Waypoints remain approximately 3 m apart along x.
Each waypoint requests a speed sampled independently from
`[0.5 + 1.0*d, 1.5 + 1.0*d]` m/s: a fixed 1 m/s window, from [0.5, 1.5]
at difficulty 0 to [1.5, 2.5] at difficulty 1. Episodes allow 40 seconds.
Routes and speeds reproduce under the same reset seed and difficulty. Resets
reuse the physics model at the same difficulty; level changes rebuild the track.

Flat reward weights are balanced so no single term dominates: at a good 1 m/s walk the
per-step budget is roughly `progress` 1, `velocity_tracking` 1, gait (`feet_air_time` +
`feet_step_ahead` + `feet_step_symmetry`) 0.5 and posture (`alive`, `upright`, `heading`)
0.8, with 20 per waypoint and 200 for the goal on top. `heading` 0.3 and `lateral_velocity`
-0.2 make walking sideways cost about 0.17 per step.

Rough and Procedural use the same progress, bonus, fall, gait and heading weights
(`LOCOMOTION_REWARDS` in `tasks.py`); only tracking, posture and smoothness weights differ
per stage. Keeping one reward scale matters for `--resume` across stages: the restored
`VecNormalize` return statistics are a cumulative average over every step so far, so after
tens of millions of Flat steps they would take about as long again to adapt to a stage paying
5x more per step, and the value function would start 5x off. Before this, Rough and
Procedural paid `progress` 5, which at 1.1 m/s is about 5.5 per step against a gallop-vs-walk
difference of about 0.45 per step from the gait terms.

Flat also applies randomised pushes (`push_robot`): a planar velocity kick in a uniform
direction every 100-300 control steps, with magnitude drawn from 0.5-1x a maximum that
ramps with difficulty (`push_velocity` 0.25 + `push_velocity_difficulty_shift` 0.35·d):
0.12-0.25 m/s at difficulty 0 up to 0.3-0.6 m/s at difficulty 1. In the viewer a red ball
flies toward the torso during the 20 steps before the kick and an arrow labelled with the
kick speed shows its direction for 20 steps afterwards.

Difficulty starts at 0. Both curriculum decisions use the same 15-episode window per worker
and the same 80 % threshold: promote by 0.05 when at least 80 % of the window are
full-course successes, demote by 0.05 when at least 80 % end with completion of 25 % or
less. A single early fall no longer resets a level; a mixed window holds the level. The
window clears on every level change. Promotion measures completion, not speed-tracking
accuracy.

The policy observes the current requested speed and the existing two-waypoint
lookahead. A planar velocity-tracking reward favors the requested speed toward
the active waypoint, with small progress/waypoint bonuses and posture/action
regularization. The first stage has no obstacles or dynamics randomization; only mild
initial pose/yaw noise plus the randomised pushes above. Height/tilt checks retain a 0.15 s
recovery window.

All three presets use `velocity_tracking=1.5` as the maximum tracking reward,
not a desired speed. The reward is `weight * exp(-velocity_error_squared / sigma^2)`,
with `velocity_tracking_sigma=0.35` m/s. It measures planar velocity toward the
waypoint, so both speed and direction matter. Each action is scored against the
waypoint, speed and direction active before that action, even when a waypoint is
crossed. Progress saturates at that command, with a 4 m/s safety cap.
There is no separate excess-speed penalty or `preferred_speed` fallback. Setting
`command_speed_range=None` disables the tracking reward.

All three presets enable soft base-height shaping. It applies only with supporting
ground contact, allowing airborne motion without a height cost. Upward rays lower
the target under overhead obstacles, allowing necessary crouching with a 0.05 m
clearance margin. This is shaping, not a change to fall-termination thresholds.

The observation layout is unchanged, so current Flat checkpoints can resume with
this curriculum. Legacy 112-input checkpoints are incompatible with this task's
116 observations (including three overhead distances). Angular velocity now uses
the correct torso-local frame; policies trained with the old rotation need
adaptation even though input size is unchanged. A running training process keeps
its already-loaded settings; restart or resume to apply these changes.

```bash
caffeinate -i python scripts/train.py Flat \
  --run-name flat_locomotion --num-envs 8 --timesteps 10000000 \
  --policy-hidden-sizes 256 256 --ent-coef 0.0 --eval-every 250000

python scripts/play.py Flat \
  --checkpoint models/flat_locomotion/final.zip \
  --no-soft-termination --no-until-fall --episodes 5
```

Use `--hours` for a bounded pilot. TensorBoard and seeded evaluation report
`velocity_tracking_error`: mean planar velocity error in m/s over each episode,
alongside survival, completion and success. Reward improvement alone is not proof
of stable walking. Test held-out route seeds before proceeding to rough terrain.
`--policy-hidden-sizes` applies only to fresh PPO policies; `--ent-coef` also applies
on resume. Training saves checkpoints every ten minutes, plus `final.zip` or
`interrupted.zip` when it stops.

### Left/right mirror symmetry

Without a symmetry prior, RL policies usually settle on one-sided gaits: `flat_0.0.9`
galloped with the left foot always leading. `train.py` therefore trains PPO through
`SymmetricPPO`, which uses mirror maps of the observation and action
(`ParkourEnv.observation_mirror()` / `action_mirror()`). The mirror reflects the world in the
vertical plane along the robot's heading, so lateral components flip, roll and yaw rates flip,
left and right joints swap with the sign their axes imply, and the height scan swaps its
lateral columns.

| `--symmetry` | What it does | Evidence |
|---|---|---|
| `loss` | Adds `--symmetry-coef` (1.0) x MSE between the policy mean at each mirrored state and the mirrored mean at the original state | Yu, Turk & Liu 2018 (asymmetric gaits without it); Abdolhosseini et al. 2019 (the most consistent of four methods); van Marum et al. 2024 (Digit) |
| `augment` | Trains every minibatch on its mirrored copy too, with the original advantages, returns and old log-probabilities, and averages each action's log std with its mirror partner's after every optimizer step | Mittal et al. 2024 (fastest PPO convergence, from a near-symmetric policy) |
| `both` (default) / `none` | Both mechanisms, or plain PPO | |

The mirror loss works from any policy, including an asymmetric one being resumed.
Augmentation only teaches much once the policy is close to symmetric: a mirrored sample counts
inside the PPO clip range only if the policy would have taken the mirrored action about as
likely as the original. On the `flat_0.0.9` gallop (mirror loss about 2.8) the median ratio of a
mirrored sample is about 1e-33 and only 0.2% fall inside the clip range. `flat_0.0.10` ended at
a mirror loss of about 0.03, yet only 8% fall inside (median ratio 0.30), for two reasons:

- The ratio multiplies over all 17 actions, so a median mean mismatch of 0.23 std per joint
  adds up to a log-ratio of about −1.5 ± 2.3. A low mirror loss is not enough; judge by
  `train/mirrored_clip_fraction`.
- The action std is one learned value per action, the same in every state, and the mirror loss
  only matches means, so left and right std drift apart (up to 0.055). That alone keeps 60%
  of mirrored samples outside the clip range. Augmentation therefore averages each action's log
  std with its mirror partner's after every optimizer step.

`both` is the default: the mirror loss keeps pulling the means together, and augmentation
learns faster as they converge, at little cost when it cannot help yet. Augmentation adds
about 17% to a PPO update, about 4% of total training time at 1450 steps/s. The flag is not
saved with the checkpoint; pass `--symmetry loss` to any `train.py` run, including `--resume`,
to train without augmentation. KL early stopping uses the original samples only. The mirror is applied in raw observation units (un-normalise, mirror,
normalise again), so asymmetric `VecNormalize` statistics do not break it. Checkpoints still
load with plain `PPO.load`.

Watch `train/symmetry_loss` (it should fall toward 0) and, with augmentation,
`train/mirrored_clip_fraction`. A symmetric policy can still produce an asymmetric gait cycle,
because a gallop that leads with either foot is itself mirror-symmetric. The reward's
`feet_step_symmetry` term handles that case. Reset noise keeps the robot from starting in an
exactly symmetric pose, which a symmetric policy could never break out of. The humanoid is
exactly mirror-symmetric, so a mirrored transition is one the simulator itself produces
(`tests/test_symmetry.py` checks this to machine precision).

### Rough terrain curriculum and speed commands

Difficulty `d` in [0, 1] changes the Rough course like this (means over 20 seeds, measured
along the route):

| `d` | Course | Hills / pits | Mean / p95 slope | Bumps per 10 m | Rubble on route per 10 m | Tallest block step |
|---|---|---|---|---|---|---|
| 0 | 16 m | ±0.03 m | 1° / 2° | 3.8 | 0 | — |
| 0.5 | 20 m | ±0.07 m | 4° / 7° | 4.6 | 2.2 | 0.23 m |
| 1 | 25 m | ±0.11 m | 8° / 16° | 5.7 | 4.9 | 0.44 m |

- **Length**: `course_length` 16 m plus `course_length_growth` 9 m × `d`, the same pattern
  as Flat. Waypoints stay 2.5 m apart; lateral offsets of up to ±`d` m make the walked path
  about 5% longer than the course at `d` = 1.
- **Hills and pits**: the fractal heightfield is centred on the spawn height, so the ground
  rises above and dips below the start level. The total relief is 0.35 m × (0.25 + 0.75 `d`).
- **Bumpiness**: the coarsest noise feature shrinks from 3.0 m to 1.5 m
  (`base_wavelength`, `base_wavelength_growth`), so bumps come more often as well as higher.
- **Rubble**: `rubble_per_metre` 0.5 × `d` blocks per metre of course, placed within 1 m of
  the route (`rubble_corridor`), 0.3-0.8 m wide and 0.05 to 0.05 + 0.25 `d` m above the
  highest ground under them. Before this, blocks were scattered over the full 12 m width with
  their base at z = 0: at `d` = 1 only 2.5 of 12 were on the route, and about 8 sat partly or
  fully inside hills.

A curriculum level change rebuilds the course at the next reset, even between the
5-reset resample interval.

`Rough` samples independent waypoint speed commands from `[0.5 + d, 2.0 + d]` m/s on each
reset, so difficulty also raises the requested speed. The current command is included in the
policy observation and a planar velocity-tracking reward trains speed and direction matching.
Progress rewards are capped and velocity tracking favors matching the command; neither
physically clamps actual velocity. The episode limit is 60 seconds, enough for the 16 m
starting course at the lowest requested speed. Friction/mass randomization remains enabled.
This preset has 116 observation values, matching the current `Flat` preset;
legacy 112-input rough checkpoints cannot be resumed directly.

```bash
python scripts/train.py Rough --run-name rough_speed --num-envs 8 --timesteps 10000000
```

### Controlled progression training

The procedural task starts at difficulty 0.05 with a 60-second time limit.
A goal means reaching the final route waypoint with adequate clearance above the
finish platform and upright posture, not merely advancing to it or passing below
it. Completion counts reached waypoints, including the final one;
for mid-course resets it measures the remaining route.

With curriculum enabled, each training worker promotes difficulty by 0.05 only
after collecting 15 episodes at that level with at least 80% successful full-course
finishes. Falls, timeouts, and mid-course starts do not count as successes.
Success history survives normal resets and terrain regeneration, but clears after
a difficulty change. New terrain uses the updated difficulty on the next reset.
The range is [0.0, 1.0]. Demotion mirrors promotion: difficulty drops by 0.05 only when at
least 80% of a full 20-episode window end with 25% completion or less, so a single early
fall does not undo a promotion. Evaluation disables curriculum and pins its difficulty.

Progress rewards metres travelled toward the active waypoint rather than giving
a progress-rate reward every step. The procedural configuration uses 20 reward per metre
(progress capped at the waypoint's command), 20 per intermediate waypoint, and 200 for completion.
Commands range from `[1.0 + 1.5*d, 2.5 + 1.5*d]` m/s, at most 4 m/s.
The tracking reward encourages matching them, rewarding deviations above and below
the command less without an additional excess-speed penalty. Survival/style weights are
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
caffeinate -i python scripts/train.py Procedural \
  --resume models/<previous-run>/final.zip \
  --run-name procedural_progression --timesteps 20000000 \
  --eval-every 1000000
```

For a wall-clock budget, add `--hours 10` and use a sufficiently large step ceiling
(for example `--timesteps 1000000000`). The first limit reached stops training.
The timer includes initial and periodic evaluation; an in-progress PPO update or
evaluation can finish after the deadline, followed by final evaluation and saving.
Normal deadline completion writes `final.zip` and `final.state.json`.
Training also saves `checkpoint_<timesteps>_steps.zip` every ten minutes of
wall-clock time, with matching `.state.json` curriculum-state files. Saves happen
at the next training step after the interval, so a long evaluation or update can
delay them. Completion (`final.zip`) and Ctrl+C (`interrupted.zip`) saves remain.
No best-model checkpoints are saved; `--save-every` is not supported. Restart an
already-running training process to pick up this checkpoint behavior.

Each run must use a new folder name. Initial, periodic, and final evaluations use 20 fixed seeded
courses at the workers' mean training difficulty, snapped to the 0.05 grid; results go into
`evaluations.jsonl` with the `difficulty` used.
The evaluation keeps training termination rules, not play overrides.

### TensorBoard and persistent logs

TensorBoard is the monitoring dashboard; the policies still train with PyTorch.
TensorFlow itself is not required. From this project's directory, start it in a
separate terminal:

```bash
../.venv/bin/python -m tensorboard.main --logdir logs --host 127.0.0.1 --port 6006
```

Open http://localhost:6006. Existing event files are readable too; a running
training process does not need restarting just to view its existing metrics.
If the port is occupied, choose another, such as `--port 6007`.

New training runs write into `logs/<run-name>/`: TensorBoard event files,
`progress.csv`, and `progress.json` (one JSON object per logged update).
Use `--log-dir /path/to/logs` to change the parent directory. Each resumed run
uses its new run name and retains the checkpoint's timestep counter, so curves
can be compared without appending to the original run. Older SB3 log folders
with numbered suffixes remain visible under the same parent directory.

In the Scalars tab, monitor these groups:

| Group                    | Meaning                                                                             |
| ------------------------ | ----------------------------------------------------------------------------------- |
| `rollout/`               | Episode return, length and success rate                                             |
| `progress/`              | Completion, success, curriculum difficulty and velocity error in m/s                |
| `termination/`           | Fraction ending for each reason                                                     |
| `reward_terms/`          | Mean accumulated reward contribution per episode                                    |
| `reward_terms_per_step/` | Mean of each episode's reward contribution divided by its length                    |
| `eval/`                  | Deterministic fixed-seed evaluation, independent of training episode averages       |
| `train/`                 | PPO KL, clipping, losses, explained variance, entropy and policy standard deviation |
| `time/`                  | Training throughput and timesteps                                                   |

Custom training summaries use the latest 100 completed episodes. Per-step reward
summaries help distinguish a stronger reward from merely a longer episode.
`progress/avg_speed` averages each episode's mean planar torso speed in m/s;
`progress/avg_height` averages each episode's mean torso height above terrain in
metres, excluding missing terrain-ray hits. These metrics are telemetry only:
observation/action dimensions and reward values are unchanged, so checkpoints
compatible before this logging change remain compatible.
Use TensorBoard's run selector to compare experiments; inspect curves with
smoothing at zero before relying on smoothed trends. Initial/final evaluations
are explicitly flushed; Ctrl+C also flushes pending metrics and closes writers.
New logging options and normalized metrics require restarting/resuming training.

Checkpoints have matching `.state.json` files that preserve each training worker's
difficulty. Keep these alongside their `.zip` files when resuming; the worker count
must match. Older checkpoints lack this state and start at the configured initial
difficulty. These files do not preserve the full physics state or RNG streams.
`play.py` also reads them to pick its default playback difficulty.

Runs resumed before the restore fix never applied the saved difficulty, and every
state file they wrote repeats the restored value instead of the workers' real level.
Rewrite those from the training log, which was always correct:

```bash
python scripts/repair_state.py <run> --dry-run
python scripts/repair_state.py <run>
```
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
