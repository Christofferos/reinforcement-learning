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

Training options worth knowing:

- Keep Mac running in backpack: `sudo pmset -a disablesleep 1` before the run and `sudo pmset -a disablesleep 0` afterwards.
- Keep Mac running when plugged in: `caffeinate -ims`
- `--subproc` (on by default; `--no-subproc` turns it off, for debugging) steps the
  environments in parallel worker processes instead of one after another in the main process. Rough on an 8-core Mac: 919 → 2250 environment steps/s
- `--max-std` caps the learned action noise, also on a resumed policy before its first
  rollout. Actions are clipped to [−1, 1], so once the noise overshoots the limits nothing in
  the reward pulls it back while the entropy bonus pushes it up: `rough_0.0.2` drifted to a
  std of 2.32 with 66% of its training actions clipped, and reached 4/40 goals at `d` = 1
  with that noise against 10/40 without. Keep `--ent-coef` above 0 with a cap, so the std
  stays near it instead of shrinking. The default is now 0.35: `procedural_0.0.3` and `0.0.4`
  sat at a 0.5 cap for their whole runs, even with the entropy bonus halved.
- `--bounds-coef` (default 0.02, from `procedural_0.0.8`; 0 turns it off) weights an
  action-bounds loss: the squared amount by which each action's mean lies outside [−1, 1].
  The clip made such means free and noise-proof, and the policy had drifted there:
  `procedural_0.0.4`-`0.0.7` kept a median |mean| of 2.2, 88% of hip and knee means outside the
  bounds and the leg targets at the ±57° edge on ~90% of steps, bang-bang control on flat
  ground and at obstacles alike, which is why less noise did not help hurdles, climbs or the
  crumbling bridge. Pulling a mean back to the bound leaves the clipped action unchanged, so
  the noise reaches it again and graded control can be learned. Resuming `procedural_0.0.7`,
  25 updates took the loss from 78 to 78, 57 and 47 per sample at 0.001, 0.02 and 0.1, with
  the KL unchanged; at 0.1 it would outweigh the policy loss for a while. TensorBoard:
  `train/bounds_loss`, `train/mean_beyond_bounds`.

```bash
caffeinate -i python scripts/train.py Rough --run-name rough_0.0.3 \
  --resume models/rough_0.0.2/checkpoint_180958824_steps.zip \
  --max-std 0.5 --hours 8
```

Playback runs at fixed difficulty with the curriculum off. It uses `--difficulty` for every
module when given, otherwise the levels saved in the checkpoint's `.state.json` (for
Procedural, each skill's level averaged over the training workers, so courses look as they
did in training), otherwise the task's starting difficulty, and prints which one it picked. A
skill the checkpoint never trained, such as a newly added module, starts at the task's
starting difficulty (0.05 for Procedural).
Procedural playback courses draw from every module, the experimental ones included, each
equally likely, so watching or evaluating a model shows them all; `--training-mix` draws only
the training skills, also each equally likely, and `--skills <module> ...` names the modules.
Training and the evaluation it runs keep training's practice-weighted mix.

### Tensorflow monitoring:

```bash
python -m tensorboard.main --logdir logs --host 127.0.0.1 --port 6007
```

### Test a specific module skill

```bash
python scripts/play.py Procedural --checkpoint models/procedural_0.0.3/final.zip --skills water_tank
```

### NOTE:

```bash
V0 trained using all_curriculum & below timesteps, result = walking:
A)
Flat --timesteps 5_000_000
B)
Rough --timesteps 10_000_000
C)
Procedural --timesteps 30_000_000
```

**Evaluating models trained with weaker legs** (everything up to `procedural_0.0.5`).
Their observations and actions are unchanged, so they load and run, but their robot was weaker:

| Model | Leg strength (`torque_ceiling_scale`) | Arm strength | Procedural speed commands (`command_speed_range`) |
|---|---|---|---|
| `rough_*`, `procedural_0.0.1`-`0.0.3` | 1.0 | 1.0 | 0.75-1.5 m/s |
| `procedural_0.0.4` | 1.0 | 1.0 | 1.0-1.75 m/s |
| `procedural_0.0.5` | 1.25 | 1.0 | 1.0-1.75 m/s |
| `procedural_0.0.6` | 2.0 | 1.0 | 1.0-1.75 m/s |
| `procedural_0.0.7` onwards | 2.0 | 1.5, then 2.0, then 3.0 | 1.0-1.75 m/s |

To see an old model as it trained, set its strength back before playing or evaluating it, and
restore the current value afterwards:

1. In `g1_parkour/robots.py`, set the four leg `torque_ceiling_scale` values of `HUMANOID`
   (`hip_x`, `hip_y`, `hip_z`, `knee`) and the three arm ones (`shoulder1`, `shoulder2`,
   `elbow`) to the model's values from the table. With 1.25x, `procedural_0.0.4` scored 56% instead of 60% on
   full `d` = 0.65 courses, and 31% at 2x: its full-effort moves overshoot.
2. For `procedural_0.0.3` and older, also set `command_speed_range=(0.75, 1.5)` in
   `parkour_cfg()` (`g1_parkour/tasks.py`), the speeds they trained with.

In a script instead, before creating the environment:

```python
from g1_parkour import robots
robots.HUMANOID.torque_ceiling_scale.update(hip_x=1.0, hip_y=1.0, hip_z=1.0, knee=1.0)  # 1.25 for procedural_0.0.5
```

Changes that make courses easier or harder but leave the robot alone apply to old models too:
lower gap landings, denser rubble, the 1.25 m tunnel cap and the relaxed height check under
ceilings. The `feet_clearance` reward changes the reward an evaluation reports, not behaviour.

### Optional (all three at once):

```bash
# flat -> rough -> procedural, each stage resuming the previous checkpoint
caffeinate -i ./scripts/all_curriculum.sh 8
```

### Versioning

`Example: flat_0.0.1, rough_0.0.1, procedural_0.0.1`

- x.y.z:
- x is major breaking changes,
- y is minor breaking changes (different obs dim),
- z is patch / iteration (same observation dimensions).

All models with identical x and y versions share observational dim.

Cameras: `side_view`, `chase_view`, `front_view` and `egocentric`, in that order when TAB cycles
the viewer's cameras. `egocentric` is a first-person view that moves with the body like a head
(`g1_parkour/camera.py`, render-only): a model camera on a mocap mount (no geoms, no joints)
that the environment moves to the eye every frame it renders, whichever camera is shown, so TAB
finds it in place. It replaced a camera fixed to the torso; `--camera egocentric_stable` still
works as an alias. The eye sits 2 cm ahead of and 10 cm below the torso's centre, at the chest;
the body never shows, as MuJoCo draws nothing nearer a camera than about 0.6 m here (1% of the
scene's size), which also hides an obstacle closer than that. It looks along the course (from
the previous waypoint to the first one 3 m beyond it) turned 18% of the way towards where the
torso faces, 30° down plus 48% of the torso's pitch, and tilts with 48% of its sideways lean. It
follows forward motion exactly and the torso's height almost directly, with the bounce
exaggerated to 1.56x against the torso's slow height so each footstep and landing reads as
up-and-down; sideways it keeps about 88% of the sway. On flat ground (the scanned ground
within 5 cm of level, nothing clipped, no ceiling) it glances 10-15° up every 3-8 s for 0.5-2 s,
and a glance once started runs to its end before anything else turns the view. It looks 15°
down while the ground scan has a magenta (clipped) reading, from about 1.9 m before a gap, or
sees ground more than 0.3 m above its floor (a hurdle, climb or flight of stairs, from about
1.5 m before); and 35° up while the upward scan sees a ceiling, from about 0.4 m before a tunnel
roof (the upward rays reach 0.4 m ahead) until it no longer sees one, which wins over looking
down. Every shift eases
in and out over 0.3 s. Glances draw on the camera's own random generator, so watching a run
never changes it. It never
leaves the head by more than 0.15 m up or down or 0.25 m sideways. Walking on flat ground the
course 2 m ahead moves 29 cm/s across the view, 22 of it up and down. A camera fixed to the
torso made it 616 cm/s, nearly all from turning with the torso's ±9° twist every stride; the
steadiest version (no twist, 30% of the bob) made it 2.6.

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
| `Procedural`            | random skill chain         | curriculum, friction/mass/gear randomisation, no pushes |
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
0.13-0.38 m or 0.63-0.88 m ahead. A ray that starts inside a terrain box (stairs rising more than 0.5 m within the ray's
reach; hurdles and climbs now stop at 0.5 and 0.45 m) reads as a miss
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

**Per-skill difficulty** (`TerrainCfg.per_skill_difficulty`, Procedural): every skill has its
own level, and every module on a course is built at its skill's level, the same from the
first module to the last. A module is passed when the robot reaches its last waypoint and
failed when the episode ends (a fall, stall or timeout) short of it. Clearing a tunnel and then
falling at the next gap is a pass for the tunnel and a failure for the gap. Every module ends
in a flat run-out of at least 1.8 m (`EXIT_RUNOUT`) with its last waypoint at the end, so with
the 0.7 m waypoint radius it counts as passed only 1.1 m past its last hazard: a fall while
recovering, such as standing up after a tunnel roof, counts against that module.
The tunnel's run-out is 0.4 m longer still (1.5 m to the pass after the roof): its old
run-out passed d = 1 roofs 0.55 m past the end, and rubble counted as passed while the robot
was still in the blocks. `tests/test_skill_curriculum.py` checks the margin for every module. Each worker keeps each
skill's last 20 attempts at its level (`skill_window`): 75% passed or more promotes that skill
by 0.05, under 50% demotes it by 0.05, and the window clears on every change. The lines were
85% and 60% until `procedural_0.0.6`, but levels are judged on noisy training attempts, which
pass 16-28 points less often than the noise-free policy (hurdle at 0.6: 97% vs 69%; climb at
0.7: 100% vs 75%), so they held every skill a step or two below what it could do.

**Practice around the levels** (training only, as the curriculum runs): a fifth of modules
are built at a random easier level (`skill_review_share` 0.2, uniform below the skill's) and a
tenth up to 0.1 above it (`skill_preview_share`, `skill_preview_range`); the rest are at the
level. `TerrainSpec.module_kinds` records which, and only modules at the level count toward
promotion. Review keeps easy instances from being forgotten: training only at its level,
`procedural_0.0.6` dropped d = 0.5 gaps from 16 to 12 of 16 while its d = 0.9 gaps went from 5
to 15. Preview shows what promotion brings, such as hurdles up to 0.04 m taller. Evaluation,
playback and fixed-difficulty sweeps (curriculum off) build every module exactly at its level. The course length
(4 modules, up to 8) follows the skills' mean level, weighted as the course samples them.
Before, one level for the whole course was promoted on 80% of whole courses finished, and
courses ramped from 0.35 × that level at the start to the level at the end (since the first
commit). So only the last module was at the stated level, and the weakest skill held back
all the others: `procedural_0.0.5` at course level 0.70 ran clean at `d` = 0.9 on ramps,
rubble and lateral shifts, while its hurdles stopped at 0.31 m because no taller ones
appeared. Expect lower whole-course success than before: most skills now sit where they pass
50-75% of the time. Judge a run by its `skill_difficulty/` curves instead.
Practice follows the levels (`skill_maintain_share` = 0.25): each skill's weight is scaled by
`1 - 0.75 × level`, so a skill at level 1 gets a quarter of the modules it would at 0 and is
kept up rather than drilled. With `procedural_0.0.6`'s final levels that moves stepping
stones (0.38) from 9% of modules to 16% and hurdles (0.61) from 14% to 18%, and run, ramp,
rubble, stairs down and lateral shifts (all about 1.0) from 2-7% each to 1-4%. The mean level
that sets the course length uses the unscaled weights. `skill_share/` in TensorBoard shows
each skill's share of module attempts.

**Action noise per module** (`TerrainCfg.skill_action_noise`, Procedural
`{"stepping_stones": 0.6}`): while the active waypoint belongs to a listed skill's module,
training samples actions with the policy's std times that scale (0.35 → 0.21 on stones). The
environment reports it as `info["action_noise_scale"]`; `SymmetricPPO.collect_rollouts` samples
with it and stores it with the step, and the update evaluates log-probabilities and entropy
(mirrored copies included) under the same scaled distribution, so the PPO ratio stays exact.
With every scale 1, training runs stable-baselines3's own code; checkpoints load as before.
`train/noise_scaled_share` is the share of steps sampled at a reduced scale. Evaluation and
`play.py` use the noise-free action. Why stones: `procedural_0.0.6` passed d = 0.5 stones 84%
noise-free but 66% with its sampled noise, so their level sank from 0.58 to 0.38 against the
60% demotion line. The noise costs every skill about as much at its trained level, though
(32 tries each, noise-free → sampled): climb 0.7 100% → 75%, hurdle 0.6 97% → 69%, gap 0.85
97% → 69%, stairs up 0.65 94% → 78%, beam 0.9 94% → 78%. Less noise on one module trades its
practice at recovering from disturbances for cleaner crossings; check it after a run by
testing stones noise-free under pushes or full noise.
`procedural_0.0.5` and older checkpoints saved one level per worker; resuming one starts
every skill at that worker's level (0.70-0.75), and the windows sort them out within minutes.

- **Climb**: a step up of 0.15 + 0.3 `d`² m (0.15 m at `d` = 0, 0.23 m at 0.5, 0.45 m at 1), rising straight from the end
  of the floor so the robot can stand against the riser before stepping up. The block reaches
  down to the floor's underside, so no slot opens under a riser taller than the 0.4 m platform
  thickness. The humanoid's knee is 0.43 m and its hip 0.79 m above the floor and it has no
  hands; the earlier 0.3-0.75 m rise after a 0.2-0.6 m gap topped out at 95% of hip height, and
  a 0.1-0.3 m gap before the riser remained until now.
- **Tunnel bar**: marked `Box(ceiling=True)`. Downward ground rays continue below it, so a
  standing robot's ground scan reads the tunnel floor; before, from `d` = 0.25 up, the bar
  spanned the rays' start height (0.1 m above the torso) and up to five rays read it as a
  1.3 m wall. The upward overhead scan still sees it. Its covered stretch grows from a 0.24 m bar at
  `d` = 0 to a 1 m roof at `d` = 1 along the course, which has to be crouch-walked. Clearance is 1.45 m at `d` = 0 to 1.25 m
  at `d` = 1, a 0.28 m duck below the standing head top (1.53 m); it went down to 1.05 m, then
  1.2 m, where `procedural_0.0.3` cleared none of 16 bars at `d` = 0.9. While the upward rays see
  a ceiling, and for 60 steps (0.9 s) after, the height termination and the progress reward's
  support gate allow the torso down to 0.475 m above the floor instead of 0.665 m
  (`TerminationCfg.ceiling_bad_height_scale` = 0.5; Procedural only). With that, the same policy
  passed 15/16 bars at `d` = 0.6 (11/16 before) and 4/16 at 0.9 (0/16); most remaining failures
  are falls 1.6-3.5 m past the bar, standing back up from a crouch it never got to practise.
  The upward rays stay at 0, 0.2 and 0.4 m ahead: shifting them 0.2 m forward would see the
  bar 0.2 m earlier but lose it while it is still over the head, and the crouch target in
  `base_height` comes from the same rays. `ceiling_clearance` (−6 per metre per step, −3 until
  `procedural_0.0.5`) charges for every metre the robot's top is above the lowest ceiling those
  rays see, less 5 cm: −1.98 per step standing upright before a `d` = 1 roof, where the
  quadratic `base_height` target charged −0.072. Under the roof the robot's top can be no
  higher than the roof, so `ceiling_contact` (−1 per step) charges for any part of the robot
  touching it: `procedural_0.0.5` scraped every `d` = 0.9 roof with torso and arms, wedged at
  the front edge on 3 of 12 and pitched over past the end on 6, for about −7 per episode. The
  pillars holding the roof are 0.12 m thick and at most 0.175 m long (`TUNNEL_POST_WIDTH`,
  `TUNNEL_POST_LENGTH`), standing on the floor's edges (they used to hang half over the pit
  beside it), not side walls as long as the roof. A tunnel bar touched from below no longer counts as the ground
  the robot stands on, which had read the floor at bar height and asked it to stand up.
- **Hurdle**: 0.2 + 0.3 `d`² m (0.2 m at `d` = 0, 0.28 m at 0.5, 0.5 m at 1; 63% of hip height);
  it went to 0.65 m. Hurdle and climb grow with `d`² because `procedural_0.0.2` stepped over or
  onto 0.20 m every time and failed from 0.24 m: linear growth put every obstacle at
  `d` = 0.38 past that line, so it practised only obstacles it always failed.
- **Swing-foot clearance** (`feet_clearance` 0.5, every stage): while the ground scan shows
  terrain at least 0.1 m above the floor the robot stands on, from 0.2 m behind to 0.6 m ahead
  and within 0.2 m to either side, each foot in the air earns its height above that floor as a
  share of the obstacle plus 5 cm (at most 1). Nothing on flat ground, for a planted foot, or
  without forward motion (0.2 m/s), so it never pays for high-stepping in general or for lifting
  a foot in place in front of a hurdle. `procedural_0.0.2`-`0.0.4` stalled at hurdles and climbs
  of 0.24, 0.30 and 0.33 m. On `procedural_0.0.4`'s own runs it would have been 1.6-4.5% of the
  return on climb, hurdle and rubble modules and nothing on a flat run.
- **Posture and step-ups** (from `procedural_0.0.8`, every stage):
  - `base_height_low` (1.0) charges, every step, each metre the torso is below standing
    height (1.25 m over the ground holding it up, lowered under a roof) beyond 3 cm.
    `procedural_0.0.7` walked flat ground crouched, its torso at 1.15 m (the default stance
    is 1.27 m, straight legs 1.30 m) and its knees at a median −83° of the −87° the actions
    reach, leaving no bend for a step-up; the quadratic `base_height` charged that 0.004 a
    step, this about 0.04-0.07. Crouching under tunnel and vent roofs costs nothing.
  - `vertical_progress` (1.0) is `progress`'s vertical twin: it pays per m/s the torso closes
    its gap to standing height over the active waypoint, measured against the same target
    across a step. `base_height` measures against the ground under the torso, so with a foot
    on a climb's top and the torso still over the floor it asked for nothing more, where
    `procedural_0.0.6`-`0.0.7` failed their climbs. A 0.15 m deadband keeps gait bobbing and
    a low posture out of it. It is a potential, the change of one height measure with rises
    and drops alike, so any crouch, stumble or jump and its recovery nets zero; an ending
    repays the drops charged and not climbed back, so a fall costs nothing extra; and the gap
    a higher waypoint opens is never charged, so a failed climb keeps only what it rose.
    Replaying `procedural_0.0.7` without training on it: flat runs, gaps, hurdles and stairs
    down net 0.0; d = 0.6 climbs +13; d = 0.8 climbs +17 passed and +13 failed; d = 0.8
    stairs up +61 and +47; d = 0.9 tunnels +7, for standing back up after the roof. Paying
    rises only with a foot down had netted −9 to −17 on every module (each running stride's
    landing charged), and repaying the whole open gap at an ending had paid failed climbs as
    passed ones.
- **Leg strength** (`RobotSpec.torque_ceiling_scale`, humanoid): hip and knee torque ceilings
  are 2x (hip 240 Nm forward/back and 80 Nm sideways and twist, knee 160 Nm): 1.25x for
  `procedural_0.0.5`, then 2x from `procedural_0.0.6`, skipping the planned 1.5x step. They apply to the robot everywhere, every stage
  and every module, set after the position-control gains are taken from the old limits, so the
  joints are exactly as stiff and push the same up to the old limit. As modelled (2.9 Nm/kg at the hip, 1.9 at the knee, no ankles)
  its best standing jump rose 7 cm, against 30-40 cm for an untrained adult; through the
  position controller it rises 20 cm at 2x. `procedural_0.0.4` was at its hip and knee limits on a
  quarter of its steps even on flat ground, so its maximum-effort moves now overshoot: on full
  `d` = 0.65 courses it scored 60% at 1x, 56% at 1.25x, 42% at 1.5x and 31% at 2x (hurdles at
  `d` = 0.6: 11, 13, 8 and 2 of 16). Resuming it needs relearning at 2x; stepping 1.25 -> 1.5 -> 2
  over successive runs keeps most of its performance. `procedural_0.0.5` (trained at 1.25x)
  scored 65%, 60% and 48% at 1.25x, 1.5x and 2x on the same courses, so `procedural_0.0.6`
  starts with that dip to relearn.
- **Arm strength** (same mechanism, `shoulder1`, `shoulder2`, `elbow`): 1.5x (15 Nm) from
  `procedural_0.0.7`, then 2x, then 3x, a step per run like the legs. The arms were 10 Nm per
  joint on a 42 kg robot: on hands and knees about 165 N rests on the hands, which with bent
  arms needs ~20 Nm at each elbow; a push up off the ground or the air vent's hand support
  needs more. `procedural_0.0.6` had an arm at its limit on 96% of steps, swinging them at full
  force, so stronger arms make the same swings overshoot: on full `d` = 0.6 courses it scored
  58% at 1x, 46% at 1.5x, 27% at 2x and 29% at 3x.
- **Run** (a sprint): its waypoints ask for the course's sampled speed plus 0.5 m/s at `d` = 0
  to 1.0 m/s at `d` = 1 (`ModuleOutput.speed_boost`, added at reset), so 1.5-2.25 m/s at `d` = 0
  and 2.0-2.75 m/s at `d` = 1 on Procedural's 1.0-1.75 m/s commands. The policy sees it through
  its speed-command input as the run's waypoint becomes active, and the existing terms pay for
  it: velocity tracking against the higher command, and a progress cap raised to it. Its
  normaliser's spread for that input (mean 1.22, std 0.60 in `procedural_0.0.4`) puts even
  2.75 m/s only 2.5 std out, so the boosted commands need no rescaling.
  A clock is built but off (`RUN_CLOCK_SHARE` = None; (0.5, 0.65) turns it on): with it, from
  entering the module the robot has the time
  its waypoint speeds ask for, divided by 50% (`d` = 0) to 65% (`d` = 1) of that speed
  (`RUN_CLOCK_SHARE`), plus 0.3 s to speed up, or the episode ends `too_slow` (no fall
  penalty; for the curriculum, a failed run). `procedural_0.0.6` averaged 64%, 59% and 55% of
  the command at `d` = 0, 0.5 and 1 (1.20-1.32 m/s against 1.90-2.41 m/s), whatever the boost,
  so the clock turns the run's level into a speed curriculum: it settles where the robot keeps
  up and rises as it gets faster. Zero-command episodes have no clock. When rendering, a black
  box 2.2 m over the middle of the module shows the remaining seconds in red seven-segment
  digits on all four sides. The rewards themselves barely pay for speed: progress pays the
  robot's speed each step, which over a fixed distance totals the same at any speed; the
  per-step terms pay a slower crossing slightly more; and velocity tracking
  (`exp(-error² / 0.5²)`) gives about 1% of its maximum at a 1.1 m/s shortfall.
- **Rubble**: 8 blocks at `d` = 0 to 24 at `d` = 1 over the module's 3 m × 2 m field (4 to 12
  before, which `procedural_0.0.4` crossed 16/16 at `d` = 0.6), up to 0.2 m high. Back at weight
  1.0 with `ramp`, after both slipped at weight 0.5 (16 -> 13 and 16 -> 12 of 16 at `d` = 0.9).
- **Gap**: 0.3 m at `d` = 0 to 0.8 m at `d` = 1, ±10%, up to the humanoid's ~0.9 m longest
  stride with each foot 5 cm off the edges. It went from 0.45 m to 1.5 m ±15%, up to a 1.73 m
  running long jump for a walker stepping 0.3-0.45 m. The landing is never higher than the
  take-off once the gap is long: up to 0.15 `d` higher below `d` = 0.5 (tapering to level at
  0.5), and from there the longer the gap the lower, 0.20-0.35 m down at `d` = 1. Stepping off
  with no upward speed, 0.3 m of drop is 0.25 s more in the air, about 0.3 m more reach at a
  1.3 m/s take-off. It stops at 0.35 m, above the ground scan's −0.5 m clip (0.4 m below the
  floor), so a landing never reads like falling off the course. It was ±0.15 `d` at every
  level, so a `d` = 1 gap could land 15 cm up; the humanoid's best standing jump rose 7 cm.
- **Beam**: 1.0 m wide at `d` = 0 to 0.28 m at `d` = 1. At 1.0 m every ray of Procedural's
  ±0.4 m ground scan lands on it, so it starts out reading like a walkway; the outer rays hang
  over the pit from about `d` = 0.3. It started at 0.6 m, with 18 of 45 rays over the pit.
- **Stepping stones**: 3-6 stones from 0.8 m to 0.5 m square, with one edge-to-edge gap
  before the first, between stones and after the last: 0.1 m at `d` = 0 to 0.3 m at `d` = 1,
  so the longest step needed is 0.4 m. Neighbours sit at most 0.4 m apart sideways and
  overlap by at least 0.1 m. Stone centres used to be evenly spaced from the previous section
  with the landing reaching 0.6 m back, so at `d` = 1 the first gap was a 1.07 m jump and at
  `d` = 0 the last stone overlapped the landing.
- **Speed commands and rewards**: Procedural samples Rough's `[0.3, 1.2]` m/s with no
  difficulty shift. It used `[0.5 + d, 2.0 + d]`, up to 3 m/s against a ~1 m/s walker. Its
  posture and smoothness weights (`alive` 0.05, `action_rate` and `angular_velocity` −0.01)
  and tracking sigma (0.35) now come from `LOCOMOTION_REWARDS` like Flat's and Rough's.
- **Resuming from another stage** (`--resume` a Rough checkpoint into Procedural) starts at
  the new task's initial difficulty; the saved curriculum level is only restored within a task.
- **Viewer**: rebuilding the course keeps the viewer window and points it at the new model.
  Closing it on every rebuild (every Procedural reset) terminated GLFW, which could leave the
  screen black after a reset.

- **Water tank** (`water_tank`, in the Procedural mix since `procedural_0.0.7`; watch it with
  `python scripts/play.py Procedural --skills water_tank --difficulty 1`). Its difficulty is
  eased (`WATER_TANK_EASE`): since 2026-10-17 `d` = 0.9 builds what `d` = 0.3 did, every value a
  third as far along, so `d` = 1 is the old 0.33 (a 0.35 m drop, banks to 0.12 m);
  `procedural_0.0.7`-`0.0.8` had reached only levels 0.22-0.3 and passed 5 of 16 at `d` = 0.3.
  The ranges below are the uneased ones. An 0.8 m deck of its
  own (one block down to the tank's base, so nothing overlaps the previous module and its end
  is the riser), then a 0.3-0.45 m drop into water standing 0.1 m below the rim, over a soft sand floor with 2-4 sand banks
  0.05-0.25 m high, out by a 20° sand ramp back to rim height. The drop stops at 0.45 m so
  the floor stays above the ground scan's −0.5 m clip. Same 116 observations and 17 actions.
  - _Sand_: `Box.solref`/`solimp` (0.05, 1.5) / (0.3, 0.95, 0.05): a standing foot sinks
    ~3.5 cm (0.06 cm on concrete) and nothing bounces. Terrain priority makes the terrain's
    values decide the contact. Geoms without them write no attributes, so other courses are
    unchanged.
  - _Sand banks_ are wide cylinders lying across the tank, buried so only the crest shows and
    the flanks meet the floor at 35°, below what `feet_edge` counts as a drop. A heightfield
    averaged 12-17 contacts on the robot against 1.7 on boxes and ran 3× slower; the upward
    rays also treat a ray starting inside a cylinder as a miss, as they do for boxes.
  - _Water_ (`WaterVolume`) has no collision geom, so contacts and the ground scan read the
    floor under it. `WaterPhysics` applies, once per control step through `xfrc_applied`,
    buoyancy (ρ g × the submerged part of each robot geom's volume) and quadratic drag
    (½ ρ C<sub>d</sub> A |v| v on the submerged part), so a foot swung through the water is
    slowed and one lifted clear of it is not. The humanoid's geoms are as dense as water.
    It is skipped while the robot is more than 1.5 m from every tank, and off with
    `ParkourEnvCfg.water_physics = False`.
  - _Rendering_: the water body (see-through, reflective) exists only when the env renders,
    and `WaterEffects` adds wave crests, ripples around wading legs and splashes from fast
    entries as viewer markers. Crests appear at random (about two at a time in a 5 m tank),
    each with its own length, slant, speed and bob, and rise out of and sink back into the
    surface over 2.5-5 s. The water body is drawn 1 cm inside the walls and above the floor,
    and the sand banks end inside the walls, so no two faces coincide and flicker. Training
    builds neither.
  - _Throughput_: see the table under the experimental modules below. The tank's cost is the
    soft sand, not the water: it runs as fast with water physics off.

### Moving and special modules

Same 116 observations and 17 actions. From `procedural_0.0.7` (2026-10-17) these are in the
Procedural mix (`INTRODUCED_SKILLS`), at weight 0.3 each: `water_tank`, `icy_slide`,
`spinning_barrel`, `wind_turbine`, `crumbling_bridge`, `sliding_elevator` and `seesaw`, joined
later by `tilt_passage` and `ball_balance`; `trampoline` went back out (level 0.11 after
`procedural_0.0.8`), leaving it and `air_vent` in `EXPERIMENTAL_SKILLS`. Each starts at level 0.05, where level-scaled practice gives it nearly its full
weight while mastered skills get a quarter of theirs, so together they take about 21% of
modules at first (at 0.5 it was 30%, with hurdles down from 18% to 13%), shrinking as they
level up.
Known gap: the sliding elevator's command keeps asking for walking speed while the robot
should wait at the boarding edge (only the stall clock pauses there).
Watch one with `python scripts/play.py Procedural --skills <module> --difficulty 1`; every
module of the course is then at exactly that difficulty.

- **`spinning_barrel`**: 1-3 barrels (0.35 m radius; 2 from `d` = 0.25, 3 from 0.75) across a
  pit on free axles. A foot on
  a barrel's side turns it, so the robot has to step onto the crest and off. They get
  narrower (2.0 to 1.2 m) and spin more freely (axle damping 25 to 4) with `d`.
- **`icy_slide`**: an icy decline (6° to 9°, friction 0.30 to 0.22) onto a run-out with a
  snowman (carrot nose, top hat) in a random corner, off the robot's line and drawn only
  when rendering. Friction is the main lever, as it limits every step's grip; the decline
  went from 15° with 0.12 friction (no grip even on the flat, 3.1 m/s at the bottom) to 12°
  with 0.18, and now 9° with friction never below 0.22. At every level the friction is well
  above the decline's tangent, so the ice is slippery underfoot but holds still (a crate
  settles 3 cm in 2 s at `d` = 1).
  The ice gets glassier as friction drops: five materials `MatIceSlide0`-`4`, reflectance
  0.1 to 0.5, picked from the friction. Ice is invisible to the scans; the robot only feels it.
- **`seesaw`**: a plank on a damped hinge (8° to 14° each way), resting near end down at the
  deck, with its pivot past the plank's middle. With a 42 kg load it tips from 0.52 m past the
  pivot in 0.94 s at `d` = 0, and from 0.16 m in 0.49 s at `d` = 1 (a lighter plank, a 0.12 to
  0.07 m pivot offset, damping 800 to 190), setting its far end at the exit.
- **`trampoline`**: drop 0.25-0.35 m onto a 1 m bed tilted 5° down towards a full-width open
  slot (0.2 m at `d` = 0 to 0.25 m at `d` = 1), and bounce over it onto a landing that ramps
  10° back up. The landing's near edge is 0.1 m (`d` = 0) to 0.4 m (`d` = 1) above the bed's
  low end, and it rises another 0.14 m over 0.8 m. This V with a pit at the bottom is the
  trampoline's signature in the height scan, found in no other module. Flat, mid-bed it read
  like a climb. The slot is at least one 0.2 m scan row wide, so a ray always lands in it, and
  narrower than the shortest `gap` (0.27 m). The bed is a 10 kg body on a spring (5000 N/m, a
  slide joint along the bed's normal), not a soft contact: soft contacts either barely gave
  or, underdamped, launched the robot (one test throw reached 11 m). The spring alone returned
  about half a landing's energy, so `TrampolineActor` launches the robot: a landing faster than
  1.5 m/s along the normal arms the bed, and its rebound gives the base 1.4 × the landing speed
  along the normal, at most 3.5 m/s. The tilt turns about 0.3 m/s of that into forward speed.
  Dropped from the deck while just holding its pose, the robot clears the slot and lifts its
  feet 0.32 m (`d` = 0) to 0.20 m (`d` = 1) above the landing's near edge on the first
  bounce. A bounce that falls short of the slot lands on the bed and is launched again; one
  that reaches the slot can drop into it. Settling onto the bed does not arm it. The net is
  drawn on the bed and moves with it.
- **`wind_turbine`**: a walkway past a cannon on the left or right that, only while the robot
  is on the walkway, fires a ball every 1.6 to 0.8 s kicking its base sideways by up to 0.3 to
  0.8 m/s (Rough's strongest random push is 0.75), and blows a steady crosswind of 0 to 10 N
  (2.4% of the humanoid's weight) away from the cannon. At `d` = 1 that deals about 1.0 m/s
  of sideways velocity per second; 0.6 s shots of up to 1.0 m/s with a 30 N wind (2.0 m/s per
  second) were far too much. Painted arrows show the wind; blades
  spin in the cannon's mouth when rendering.
- **`sliding_elevator`**: a platform shuttling 0.5-0.875 m across a pit at an average
  0.375-0.44 m/s (peaking at π/2 times that mid-crossing) with a 1.5-1.35 s stop at each end;
  it shrinks from 1.4 × 1.6 m to 1.3 × 1.45 m. A full cycle takes 5.7-6.7 s. The top end is
  capped at what used to be `d` = 0.5 (it went to 1.25 m of travel at 0.5 m/s), so every value
  changes half as fast along the curriculum. Eased again since 2026-10-17
  (`SLIDING_ELEVATOR_EASE`): `d` = 0.9 builds what `d` = 0.5 did, so `d` = 1 is the old 0.56
  (`procedural_0.0.8` had reached only 0.34); the ranges given are the uneased ones. It is a 2000 kg body on a slide joint whose
  position and velocity are set every control step, so friction carries a rider. It runs only
  while the robot is within 3 m of the module, from a random point of its cycle. Waiting at the
  boarding edge with the platform away pauses stall termination.
- **`crumbling_bridge`**: a sagging rope bridge of 0.3 m planks with 4-15 cm gaps over a pit;
  each plank drops 1 s (`d` = 0) to 0.5 s (`d` = 1) after the robot first touches it, drifting
  and tumbling its own random way. Planks are mocap bodies (no joints).
- **`tilt_passage`**: a corridor between two 1 m walls that fill the floor to its edges,
  narrowing from 0.8 m (`d` = 0, walked facing forward) to 0.28 m (`d` = 1) over 1.5-3 m. Without
  touching the walls the humanoid needs 0.32-0.34 m facing forward in any leg pose, 0.33 m
  sideways in its bent-knee stance and 0.24 m sideways with straight legs, so the narrowest
  makes it turn sideways and straighten up; the width narrows smoothly, so every few degrees
  of turn help. The walls sit below the shoulders (arms out to 0.7 m at 1.2-1.3 m pass over
  them) and are `Box.wall`: a ground ray that starts inside one, from a crouching robot, now
  reads it as a wall, where `mj_ray` skipped it and read the floor beside it. Turned sideways
  the scan sees only 0.4 m along the corridor, so it is flat and empty. The heading and
  lateral-speed terms cost a sideways walk at most 0.3 and about 0.1 per step, well under
  what progress pays. Before it trains, its scan readings (a wall reads up to +1.5 m, 8-12
  standard deviations out on the current normalisation) want the normaliser's scan statistics
  widened without changing the policy, a generalisation of `adopt_constant_inputs`.
- **`ball_balance`**: a row of fixed balls over a pit, their crests level with the deck: step
  crest to crest, balancing on a curved patch each time. Two big balls (0.675 m radius) at
  `d` = 0 to four smaller ones (0.4 m) at `d` = 1, their crests 0.6 m to 0.8 m apart: big balls
  overlap into a bumpy walkway with 0.07 m dips and the smallest just touch, with a sideways
  jitter up to 0.12 m. The crests stay a stride apart however big the balls; the 0.2 m scan
  puts two to four rays on every ball.
- **`air_vent`**: a low duct to crawl up through, then stand up and walk on. Its roof funnels
  down from 1.45 m over 1 m to 1.2 m (`d` = 0, a deep duck-walk) to 0.95 m (`d` = 1, hands and
  knees). From the end of that inlet the floor ramps up 0.15 m (`d` = 0) to 0.25 m (`d` = 1)
  over the duct's 1.5-3 m, the roof following it at the same height above the floor, and the
  duct opens where the ramp ends; the run-out after it is 0.6 m longer than `EXIT_RUNOUT` for
  standing up. Under
  a ceiling at most 1.22 m above the floor (`TerminationCfg.crawl_ceiling_clearance`; tunnel
  roofs are 1.25 m or more, so nothing else changes) and for 60 steps after, the height check
  allows the torso down to 0.24 m (`crawl_bad_height_scale` 0.25) and the tilt check up to
  110° (`crawl_orientation_limit` −0.35), for the progress reward's support gate too. The
  deepest kneeling pose the actions reach (hips at 76° and knees at 87°, the most the ±1 rad
  action range allows; shins and forearms on the floor) holds at 0.31 m and an upright of
  −0.16, which the earlier 0.285 m and −0.17 left no room to rock in. The arms
  are being strengthened a step per run (see Arm strength) for the hand support a crawl needs.

How moving terrain fits in without changing the observation:

- **Robot-only joints.** The observation, its left/right mirror, reset noise and the
  joint-limit penalty use the robot's own joints, found from its kinematic tree, instead of
  `qpos[7:]`. Terrain bodies are written after the robot so its free joint keeps
  `qpos[0:7]`. On Flat, Rough and Procedural the observations, rewards, terminations and
  mirror are bit-identical to before, checked over fixed-seed rollouts.
- **`TerrainBody`**: a hinge or slide joint for physics to drive (barrel, seesaw, trampoline,
  elevator) or a mocap body an actor places (bridge planks). Jointed bodies have their
  contacts with the static terrain excluded, so they only meet the robot; a trampoline bed
  rubbing on its deck jammed.
- **Actors** (`terrain/actors.py`): per-module runtime behaviour, called at bind, reset, once
  per control step and when rendering. Each returns at once while the robot is away.
- **Decoration** (`Box(visual=True)`: ropes, nets, arrows, barrel staves) has no contacts or
  mass and is left out of the scene unless it renders, like the drawn water.

Throughput (`python scripts/benchmark_modules.py --render`, one env, random actions, robot
placed on the module after every reset; physics at `d` = 0.5):

| Course | steps/s | vs Procedural | frames/s (640×360) |
|---|---|---|---|
| Procedural | 939 | | 70-77 |
| `water_tank` (water physics off) | 810 (785) | −14% (−16%) | 60 |
| `spinning_barrel` | 959 | +2% | 69 |
| `icy_slide` | 851 | −9% | 70 |
| `seesaw` | 941 | 0% | 78 |
| `trampoline` | 966 | +3% | 75 |
| `wind_turbine` | 960 | +2% | 77 |
| `sliding_elevator` | 965 | +3% | 80 |
| `crumbling_bridge` | 929 | −1% | 62 |
| `tilt_passage` | 721 vs 712 | +1% | |
| `ball_balance` | 743 vs 712 | +4% | |
| `air_vent` | 689 vs 712 | −3% | |

Run renders one per process: on macOS, offscreen renderers made after the first in a process
can return black frames.

MuJoCo draws a reflection on only one surface per scene, the first reflective one, which was
a 2 mm lane line. Lane lines, the finish checker, metal ramps (now brushed grey) and Rough's
icy rubble are not reflective, so only the icy slide's ice or the water tank's surface gets it
(whichever comes first on a course). A metal ramp coming first had mirrored the scene in white.

Procedural-only settings (same 116-value observation, so Rough checkpoints still resume):

- **Ground scan at 0.2 m**: the 9 × 5 rays span 0.2 m behind to 1.4 m ahead and ±0.4 m, so
  any gap or crack 0.2 m or wider is always under a ray (0.2 m stone cracks were missed 22%
  of the time at 0.25 m) and edges are placed to within ±0.1 m. Cracks narrower than the
  spacing still flicker; at 0.1 m no foot fits in them.
- **Off-course rays read −0.5**, not −1.0. Normalised on Rough's scale (std ~0.12 m), −1.0
  was −7 std and saturated ~30% of the first tanh layer on beams, stones and gaps (12-26% at
  −0.5). `procedural_0.0.1` failed every beam (0/20) and passed 4/20 with misses at −0.5 and
  14/20 at −0.3; −0.5 keeps pits clearly apart from steps down. Readings are taken from the
  torso (1.25 m minus the drop from 0.1 m above it), so flat ground under a robot at its
  nominal height reads −0.1 and the −0.5 clip sits 0.4 m below the floor it stands on
  (0.55 m when crouched to 1.1 m). In the viewer, a clipped ray's dot stops at that reading
  and turns magenta instead of falling into the pit.
- **Extra practice where the last run failed** (`skill_weights`): hurdle, climb and stepping
  stones 3×, gap 1.5×, the rest 1× (~16% of modules each for the weighted three, ~5% for the
  rest). `procedural_0.0.1` failed beam, stones and gap at `d` = 0, which capped full-course
  success near 22%; 3× practice on those let `procedural_0.0.2` climb to `d` = 0.38 with beams
  at 0% and gaps at 6-20% failure, leaving hurdles (100%), climbs (36-71%) and stones (22-55%).

A course is `start_pad → N random modules → finish_pad`, N growing from 4 at `d` = 0 to 8
at `d` = 1 (`num_modules` + `num_modules_growth`), with the per-module difficulty
ramping up along the course. `difficulty ∈ [0, 1]` scales gap widths, step heights, beam
narrowness, overhead clearance and terrain noise amplitude. Each module also emits the
route waypoints that the reward and the `waypoint_command` observation follow, so the
policy is trained to _sequence_ skills rather than to memorise one track.

Terrain is regenerated on reset (`TerrainCfg.resample_every_n_resets`) and each skill's
difficulty is auto-adjusted from how often its modules are passed (`TerrainCfg.curriculum`,
`per_skill_difficulty`; see Per-skill difficulty above).

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
  and right step lengths, so a gallop or limp pays on every step), `feet_slide` (planar
  speed of feet in contact) and `feet_stumble` (−1 per step on which a foot or shin presses
  on a terrain face steeper than 60°, such as a rubble block's side; never on flat ground)
  and `feet_edge` (−0.5 per foot per step whose support centre, the mean of its ground
  contacts, is within 5 cm of a drop deeper than 5 cm: a block-top edge, stair nose, gap or
  beam side; found by sampling the ground at eight points around that centre, so a step up
  beside the foot or a seam between flush platforms does not count). On the shallowest
  Procedural stairs (0.32 m treads) that is the 16% of each tread before its drop; a flat
  sole is judged by its middle, so a toe resting on a stair nose is not penalised.
  Steps used to be measured on the torso heading, and `flat_0.0.9`
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

| `--symmetry`              | What it does                                                                                                                                                                                              | Evidence                                                                                                                                          |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `loss`                    | Adds `--symmetry-coef` (1.0) x MSE between the policy mean at each mirrored state and the mirrored mean at the original state                                                                             | Yu, Turk & Liu 2018 (asymmetric gaits without it); Abdolhosseini et al. 2019 (the most consistent of four methods); van Marum et al. 2024 (Digit) |
| `augment`                 | Trains every minibatch on its mirrored copy too, with the original advantages, returns and old log-probabilities, and averages each action's log std with its mirror partner's after every optimizer step | Mittal et al. 2024 (fastest PPO convergence, from a near-symmetric policy)                                                                        |
| `both` (default) / `none` | Both mechanisms, or plain PPO                                                                                                                                                                             |                                                                                                                                                   |

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
| --- | ------ | ------------ | ---------------- | -------------- | ------------------------ | ------------------ |
| 0   | 16 m   | ±0.03 m      | 1° / 2°          | 3.8            | 0                        | —                  |
| 0.5 | 20 m   | ±0.07 m      | 4° / 7°          | 4.6            | 2.2                      | 0.23 m             |
| 1   | 25 m   | ±0.11 m      | 8° / 16°         | 5.7            | 4.9                      | 0.44 m             |

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

`Rough` samples independent waypoint speed commands from `[0.3, 1.2]` m/s on each reset,
whatever the difficulty, and tracks them with `velocity_tracking_sigma=0.5`. `rough_0.0.1`
sampled `[0.5 + d, 1.25 + d]` with sigma 1: every command was faster than the ~0.9 m/s it
could walk, so it learned to ignore the command and walked 0.85-0.95 m/s whether told 0.5 or
1.75. The current command is included in the
policy observation and a planar velocity-tracking reward trains speed and direction matching.
Progress rewards are capped and velocity tracking favors matching the command; neither
physically clamps actual velocity. The episode limit is 60 seconds, enough for the 16 m
starting course at the lowest requested speed. Friction/mass randomization remains enabled.
Every terrain geom has contact `priority="1"` above the robot's 0, so a contact's friction
comes from the terrain alone. With equal priorities MuJoCo takes the larger of the two, and
the feet's 1.0 hid every slippery ground draw, ice section and 0.6 metal ramp below it.
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

With curriculum enabled on Flat and Rough (Procedural moves each skill instead; see
Per-skill difficulty), each training worker promotes difficulty by 0.05 only
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
clearance may drop to 0.71 m for the humanoid (0.95 m × `bad_height_scale` 0.75) before
the height counter starts. Hands on a 0.45 m climb or 0.5 m hurdle need the torso at
about 1.0 m (shoulder 0.06 m above the torso, arm reach ~0.59 m), and the `d` = 1 tunnel
crouch target is about 0.87 m, so none of them forces the torso below that line.
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
| `skill_difficulty/`      | Procedural: each skill's level (mean over workers) and `mean`, the course's level   |
| `skill_pass_rate/`       | Procedural: the share of each skill's modules passed in the latest 100 episodes    |
| `skill_share/`           | Procedural: each skill's share of the module attempts in the latest 100 episodes   |
| `skill_review_pass_rate/`, `skill_preview_pass_rate/` | Procedural: pass rates of the easier and harder practice modules (not counted for promotion) |
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
difficulty (for Procedural, each skill's level per worker). Keep these alongside their `.zip` files when resuming; the worker count
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
