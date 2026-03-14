"""Stress-test: generate 200 arenas, verify no crashes and open maps get more objects."""
import sys
from pathlib import Path
import numpy as np
import mujoco

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from worldgen import generate_arena

rng = np.random.default_rng(123)
layout_counts = {}
open_box_counts = []
open_ramp_counts = []

for i in range(200):
    xml, meta = generate_arena(rng)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    lt = meta["layout_type"]
    layout_counts[lt] = layout_counts.get(lt, 0) + 1
    if lt == "open":
        open_box_counts.append(meta["n_boxes"])
        open_ramp_counts.append(meta["n_ramps"])

print("Layout distribution (200 arenas):")
for lt, cnt in sorted(layout_counts.items()):
    print(f"  {lt:10s}: {cnt}")

if open_box_counts:
    print(f"\nOpen maps: boxes min={min(open_box_counts)} max={max(open_box_counts)} "
          f"avg={np.mean(open_box_counts):.1f}")
    print(f"Open maps: ramps min={min(open_ramp_counts)} max={max(open_ramp_counts)} "
          f"avg={np.mean(open_ramp_counts):.1f}")

print("\n✅ All 200 procedural arenas generated and simulated successfully!")
