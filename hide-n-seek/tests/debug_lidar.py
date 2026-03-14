"""Quick diagnostic: what do lidar rays actually hit?"""
import sys, os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', 'src')))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import mujoco
from env import HideAndSeekEnv, AGENT_NAMES, _compute_lidar

env = HideAndSeekEnv(procedural=True, horizon=240)
obs, info = env.reset()
acts = {n: np.array([0, 0, 0], dtype=np.float32) for n in AGENT_NAMES}
env.step(acts)

print(f"Total geoms in scene: {env.model.ngeom}")
print(f"Total bodies in scene: {env.model.nbody}")
print()

# ── Classify every geom ──
categories = {}
for gid in range(env.model.ngeom):
    gnm = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    bid = int(env.model.geom_bodyid[gid])
    bnm = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, bid)
    label = gnm or bnm or "unknown"

    if "grid" in label.lower():
        cat = "GRID_LINE"
    elif "terrain" in label.lower():
        cat = "TERRAIN"
    elif "floor" in label.lower():
        cat = "FLOOR"
    elif "wall" in label.lower():
        cat = "WALL"
    elif "box" in label.lower():
        cat = "BOX"
    elif "ramp" in label.lower():
        cat = "RAMP"
    elif "hider" in label.lower() or "seeker" in label.lower():
        cat = "AGENT"
    else:
        cat = f"OTHER({label})"
    categories[gid] = cat

cat_counts = {}
for c in categories.values():
    cat_counts[c] = cat_counts.get(c, 0) + 1
print("Geom categories:")
for c, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"  {c:20s}: {n}")

# ── Cast lidar for each agent and classify hits ──
print("\n--- LIDAR RAY ANALYSIS ---")
for ai, name in enumerate(AGENT_NAMES):
    body_id = env._agent_body_ids[name]
    pos = env._get_body_pos(body_id)
    angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)

    hit_cats = {}
    sample_hits = []
    for i, angle in enumerate(angles):
        d = np.array([np.cos(angle), np.sin(angle), 0.0])
        gid_arr = np.array([-1], dtype=np.int32)
        hd = mujoco.mj_ray(
            env.model, env.data,
            pnt=pos.astype(np.float64), vec=d,
            geomgroup=None, flg_static=1,
            bodyexclude=body_id, geomid=gid_arr,
        )
        gid = int(gid_arr[0])
        if gid >= 0 and 0 <= hd <= 18.0:
            cat = categories.get(gid, "UNKNOWN")
            hit_cats[cat] = hit_cats.get(cat, 0) + 1
            if len(sample_hits) < 3:
                sample_hits.append((i, np.degrees(angle), cat, hd, gid))
        else:
            hit_cats["MISS"] = hit_cats.get("MISS", 0) + 1

    print(f"\n  {name} at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.2f}):")
    for c, n in sorted(hit_cats.items(), key=lambda x: -x[1]):
        print(f"    {c:20s}: {n:2d}/30 rays")
    for ray_i, deg, cat, dist, gid in sample_hits:
        print(f"    Sample: ray {ray_i} ({deg:.0f} deg) -> {cat} (gid={gid}) at {dist:.2f}m")

# ── Check grid line Z vs agent Z ──
print("\n--- GRID LINE Z CHECK ---")
for gid in range(env.model.ngeom):
    cat = categories.get(gid, "")
    if cat == "GRID_LINE":
        p = env.data.geom_xpos[gid]
        s = env.model.geom_size[gid]
        top_z = p[2] + s[2]  # pos_z + half_height
        print(f"  Grid geom {gid}: pos_z={p[2]:.4f}, half_h={s[2]:.4f}, top_z={top_z:.4f}")
        break  # just one sample

agent_z = env._get_body_pos(env._agent_body_ids["hider_0"])[2]
print(f"  Agent Z: {agent_z:.4f}")
print(f"  Ray Z == Agent Z, so grid lines at Z<0.01 should NOT be hit by horizontal rays")

# ── Verify obs vector structure ──
print("\n--- OBS VECTOR SANITY CHECK ---")
obs_h0 = obs["hider_0"]
print(f"  hider_0 obs shape: {obs_h0.shape}")
print(f"  obs[0:3] (pos): {obs_h0[0:3]}")
print(f"  obs[3:6] (vel): {obs_h0[3:6]}")
print(f"  obs[6:9] (is_hider, in_prep, prep_remaining): {obs_h0[6:9]}")
# Other agents: 3 others x (3 rel_pos + 3 rel_vel + 2 flags) = 24
idx = 9
for j in range(3):
    rp = obs_h0[idx:idx+3]
    rv = obs_h0[idx+3:idx+6]
    flags = obs_h0[idx+6:idx+8]
    print(f"  Other agent {j}: rel_pos={rp}, rel_vel={rv}, visible={flags[0]:.0f}, is_hider={flags[1]:.0f}")
    idx += 8
# Boxes: 5 x (3 rel_pos + 2 flags) = 25
print(f"  Box obs start at idx {idx}:")
for bi in range(5):
    rp = obs_h0[idx:idx+3]
    flags = obs_h0[idx+3:idx+5]
    print(f"    Box {bi}: rel_pos={rp}, grabbed={flags[0]:.0f}, exists={flags[1]:.0f}")
    idx += 5
# Ramps: 2 x (3 rel_pos + 2 flags) = 10
print(f"  Ramp obs start at idx {idx}:")
for ri in range(2):
    rp = obs_h0[idx:idx+3]
    flags = obs_h0[idx+3:idx+5]
    print(f"    Ramp {ri}: rel_pos={rp}, grabbed={flags[0]:.0f}, exists={flags[1]:.0f}")
    idx += 5
# Lidar: 30 values
lidar = obs_h0[idx:idx+30]
print(f"  Lidar start at idx {idx}, shape={lidar.shape}")
print(f"    min={lidar.min():.4f}, max={lidar.max():.4f}, mean={lidar.mean():.4f}")
print(f"    num hitting (<1.0): {np.sum(lidar < 1.0)}/30")
unique_vals = len(np.unique(np.round(lidar, 4)))
print(f"    unique values: {unique_vals} (should be >>1, was 1 before bugfix)")
idx += 30
print(f"  Total consumed: {idx}, obs length: {len(obs_h0)}")
assert idx == len(obs_h0), f"OBS LENGTH MISMATCH: consumed {idx} != {len(obs_h0)}"
print(f"  [OK] Obs vector length matches expected structure")

env.close()
print("\n[OK] Full vision pipeline diagnostic complete!")
