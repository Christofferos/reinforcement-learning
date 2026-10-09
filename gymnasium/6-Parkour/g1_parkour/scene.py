"""Compose a MuJoCo scene (robot + generated terrain) into a compilable MJCF file."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .robots import RobotSpec
from .terrain.core import TerrainSpec

SHARED_ASSETS = """
    <texture type="skybox" builtin="gradient" rgb1=".4 .6 .8" rgb2="0 0 0" width="800" height="800"/>
    <texture name="grid_tex" type="2d" builtin="checker" rgb1="0 0 0" rgb2="1 1 1" width="300" height="300"/>
    <material name="grid" texture="grid_tex" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    <!-- Grass + running track from 4-LVL9-HumanoidDrunkWalk/walk.xml. Grass uses texuniform so the
         ~15 m checker squares keep the same scale on the plane and the smaller field boxes. -->
    <texture type="2d" name="texgrass" builtin="checker" rgb1="0.1 0.4 0.1" rgb2="0.2 0.5 0.2" width="100" height="100"/>
    <material name="MatGrass" texture="texgrass" texrepeat="0.0333 0.0333" texuniform="true" specular="0" shininess="0" reflectance="0"/>
    <texture type="2d" name="textrack_light" builtin="flat" rgb1="0.6 0.3 0.2" width="100" height="100"/>
    <texture type="2d" name="textrack_dark" builtin="flat" rgb1="0.45 0.2 0.1" width="100" height="100"/>
    <material name="MatTrackLight" texture="textrack_light" specular="0.1" shininess="0.1" reflectance="0"/>
    <material name="MatTrackDark" texture="textrack_dark" specular="0.1" shininess="0.1" reflectance="0"/>
    <material name="MatLine" rgba="1 1 1 1" specular="0.5" shininess="0.5" reflectance="0.2"/>
    <texture name="SimulatedWood" type="cube" builtin="gradient" width="128" height="128" rgb1="0.6 0.4 0.2" rgb2="0.35 0.2 0.1"/>
    <material name="Wood" texture="SimulatedWood" texrepeat="5 5" emission="0.1" specular="0.2" shininess="0.3" rgba="0.7 0.5 0.3 1"/>
    <texture name="ConcreteTex" type="2d" builtin="flat" width="300" height="300" rgb1="0.4 0.4 0.4" rgb2="0.5 0.5 0.5" random="0.03"/>
    <material name="Concrete" texture="ConcreteTex" texrepeat="1 1" specular="0.1" shininess="0.1"/>
    <texture name="MetalTex" type="cube" builtin="flat" width="128" height="128" rgb1="0.9 0.9 0.95" rgb2="0.9 0.9 0.95"/>
    <material name="Metal" texture="MetalTex" specular="0.1" shininess="0.1" reflectance="0.1"/>
    <texture name="PlatformTex" type="2d" builtin="checker" rgb1="0.15 0.2 0.5" rgb2="0.15 0.2 0.5" width="300" height="300" mark="edge" markrgb="1 1 1"/>
    <material name="PlatformMat" texture="PlatformTex" texrepeat="1 1" texuniform="true" specular="0.1" shininess="0.1"/>
    <material name="geom" rgba="0.8 0.6 .4 1"/>
    <!-- Friction-tinted rubble appearances, assigned per geom at reset by
         mdp.events.randomize_rubble_sections. rgba stays neutral white so geom_rgba carries
         the ice/mud colour; these materials only add the glossy vs matte shading.
         MatIce gets a little emission so the pale colour survives the scene lighting. -->
    <material name="MatIce" rgba="1 1 1 1" emission="0.2" specular="0.9" shininess="0.9" reflectance="0.4"/>
    <material name="MatMud" rgba="1 1 1 1" specular="0.0" shininess="0.02" reflectance="0"/>
"""


def _track_and_field_xml(ground_z: float, course_length: float) -> list[str]:
    """walk.xml-style floor: a 5-lane running track along +x set into a grass field.

    A full-size grass plane sits 0.1 m below the surface as a base (and safety net beyond
    the field); the grass boxes and lane boxes are laid side by side with flush tops at
    ``ground_z``, so the running surface is flat apart from walk.xml's 2 mm line ridges.
    """
    x_mid = course_length / 2.0
    half_x = course_length / 2.0 + 5.0  # 5 m margin behind the spawn and past the finish
    box_z = ground_z - 0.05
    line_z = ground_z + 0.001
    geoms = [
        f'    <geom name="ground" type="plane" size="120 120 0.1" pos="0 0 {ground_z - 0.1}" '
        'material="MatGrass" condim="3" group="1" friction="1 .1 .1"/>',
    ]
    for side, y in (("left", -10.0), ("right", 10.0)):
        geoms.append(
            f'    <geom name="grass_{side}" type="box" size="{half_x:.3f} 5 0.05" '
            f'pos="{x_mid:.3f} {y} {box_z}" material="MatGrass" condim="3" group="1" friction="1 .1 .1"/>'
        )
    for i, y in enumerate((-4.0, -2.0, 0.0, 2.0, 4.0)):
        material = "MatTrackLight" if i % 2 == 0 else "MatTrackDark"
        geoms.append(
            f'    <geom name="lane_{i + 1}" type="box" size="{half_x:.3f} 1 0.05" '
            f'pos="{x_mid:.3f} {y} {box_z}" material="{material}" condim="3" group="1" friction="1 .1 .1"/>'
        )
    for i, y in enumerate((-3.0, -1.0, 1.0, 3.0)):
        geoms.append(
            f'    <geom name="lane_line_{i + 1}" type="box" size="{half_x:.3f} 0.05 0.001" '
            f'pos="{x_mid:.3f} {y} {line_z}" material="MatLine" condim="3" group="1" friction="1 .1 .1"/>'
        )
    for name, x in (("start_line", 1.0), ("finish_line", course_length)):
        geoms.append(
            f'    <geom name="{name}" type="box" size="0.05 5 0.001" '
            f'pos="{x:.3f} 0 {line_z}" material="MatLine" condim="3" group="1" friction="1 .1 .1"/>'
        )
    return geoms


def build_scene_xml(
    robot: RobotSpec,
    terrain: TerrainSpec,
    build_dir: Path,
    num_scan_markers: int = 0,
    timestep: float = 0.003,
    num_overhead_markers: int = 0,
) -> Path:
    """Write the composed MJCF for ``robot`` on ``terrain`` and return its path."""
    build_dir.mkdir(parents=True, exist_ok=True)

    hfield_assets, hfield_geoms = [], []
    for hfield in terrain.heightfields:
        hfield_assets.append(hfield.asset_xml())
        hfield_geoms.append(hfield.geom_xml())

    world: list[str] = [
        '    <light pos="0 0 8" dir="0 0 -1" diffuse="1 1 1" specular="0.1 0.1 0.1" castshadow="true"/>',
        '    <light pos="20 0 8" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>',
    ]
    if terrain.add_ground_plane:
        world.extend(_track_and_field_xml(terrain.ground_plane_z, terrain.course_length))
    world.extend(hfield_geoms)
    world.extend(box.to_xml() for box in terrain.boxes)
    world.append('    <site name="target_marker" type="sphere" size="0.15" rgba="0 1 1 0.7"/>')
    world.append('    <site name="target_marker_next" type="sphere" size="0.12" rgba="0 1 1 0.3"/>')
    for i in range(num_scan_markers):
        world.append(
            f'    <geom name="scan_viz_{i}" type="sphere" size="0.035" pos="0 0 -50" '
            'rgba="1 0.2 0.2 0.6" contype="0" conaffinity="0" group="2"/>'
        )
    for index in range(num_overhead_markers):
        world.append(
            f'    <geom name="overhead_viz_{index}" type="sphere" size="0.035" pos="0 0 -50" '
            'rgba="0.1 0.9 1 0.8" contype="0" conaffinity="0" group="2"/>'
        )

    spawn = (terrain.spawn_pos[0], terrain.spawn_pos[1], terrain.spawn_pos[2] + robot.spawn_height)
    if robot.body_xml:
        world.append(robot.body_xml.format(x=spawn[0], y=spawn[1], z=spawn[2]))

    includes = "\n".join(f'  <include file="{path}"/>' for path in robot.includes)
    xml = f"""<mujoco model="parkour-{terrain.name}-{robot.name}">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option integrator="RK4" iterations="50" solver="PGS" timestep="{timestep}"/>
  <size nkey="5" nuser_geom="1"/>
{includes}
{robot.default_xml}
  <visual>
    <map znear="0.01" zfar="60"/>
    <quality shadowsize="4096"/>
    <global offwidth="1920" offheight="1080"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
{SHARED_ASSETS}
{robot.asset_xml}
{chr(10).join(hfield_assets)}
  </asset>
  <worldbody>
{chr(10).join(world)}
  </worldbody>
{robot.extra_xml}
</mujoco>
"""
    path = build_dir / f"scene_{terrain.name}_{robot.name}.xml"
    path.write_text(xml)
    return path


def default_build_dir(tag: str = "parkour") -> Path:
    return Path(tempfile.mkdtemp(prefix=f"{tag}_scene_"))
