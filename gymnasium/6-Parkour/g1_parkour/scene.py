"""Compose a MuJoCo scene (robot + generated terrain) into a compilable MJCF file."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .camera import EYE_MOUNT, STABLE_EGOCENTRIC
from .robots import RobotSpec
from .terrain.core import TERRAIN_GEOM_ATTRS, TerrainSpec
from .terrain.parkour import ICE_SLIDE_REFLECTANCE

SCAN_MARKER_RGBA = (1.0, 0.2, 0.2, 0.6)
"""Ground-scan dot: red, drawn at the height the policy reads."""
SCAN_CLIPPED_RGBA = (1.0, 0.2, 1.0, 0.9)
"""Ground-scan dot whose reading was clipped (an off-course ray read as ``scan_clip``): magenta."""

WATER_GEOM_GROUP = 2
"""Geom group of the drawn water: visible by default, and not group 1, which the scans hit."""
WATER_INSET = 0.01
"""Gap in metres between the drawn water body and the tank's walls and floor."""

SHARED_ASSETS = """
    <texture type="skybox" builtin="gradient" rgb1=".4 .6 .8" rgb2="0 0 0" width="800" height="800"/>
    <texture name="grid_tex" type="2d" builtin="checker" rgb1="0 0 0" rgb2="1 1 1" width="300" height="300"/>
    <!-- MuJoCo draws one reflective surface per scene, the first in it: lane lines and the
         finish checker are not reflective, so that one is the water or the ice. -->
    <material name="grid" texture="grid_tex" texrepeat="1 1" texuniform="true"/>
    <!-- Grass + running track from 4-LVL9-HumanoidDrunkWalk/walk.xml. Grass uses texuniform so the
         ~15 m checker squares keep the same scale on the plane and the smaller field boxes. -->
    <texture type="2d" name="texgrass" builtin="checker" rgb1="0.1 0.4 0.1" rgb2="0.2 0.5 0.2" width="100" height="100"/>
    <material name="MatGrass" texture="texgrass" texrepeat="0.0333 0.0333" texuniform="true" specular="0" shininess="0" reflectance="0"/>
    <texture type="2d" name="textrack_light" builtin="flat" rgb1="0.6 0.3 0.2" width="100" height="100"/>
    <texture type="2d" name="textrack_dark" builtin="flat" rgb1="0.45 0.2 0.1" width="100" height="100"/>
    <material name="MatTrackLight" texture="textrack_light" specular="0.1" shininess="0.1" reflectance="0"/>
    <material name="MatTrackDark" texture="textrack_dark" specular="0.1" shininess="0.1" reflectance="0"/>
    <material name="MatLine" rgba="1 1 1 1" specular="0.5" shininess="0.5"/>
    <texture name="SimulatedWood" type="cube" builtin="gradient" width="128" height="128" rgb1="0.6 0.4 0.2" rgb2="0.35 0.2 0.1"/>
    <material name="Wood" texture="SimulatedWood" texrepeat="5 5" emission="0.1" specular="0.2" shininess="0.3" rgba="0.7 0.5 0.3 1"/>
    <texture name="ConcreteTex" type="2d" builtin="flat" width="300" height="300" rgb1="0.4 0.4 0.4" rgb2="0.5 0.5 0.5" random="0.03"/>
    <material name="Concrete" texture="ConcreteTex" texrepeat="1 1" specular="0.1" shininess="0.1"/>
    <!-- Brushed grey, a little glossier than concrete, and never reflective: it was near-white with
         reflectance 0.1, and once nothing earlier in a course reflected, metal ramps mirrored the
         scene like the ice. Only the icy slide's ice (and the water tank's surface) reflect. -->
    <texture name="MetalTex" type="cube" builtin="flat" width="128" height="128" rgb1="0.5 0.51 0.54" rgb2="0.5 0.51 0.54"/>
    <material name="Metal" texture="MetalTex" specular="0.35" shininess="0.35"/>
    <texture name="PlatformTex" type="2d" builtin="checker" rgb1="0.15 0.2 0.5" rgb2="0.15 0.2 0.5" width="300" height="300" mark="edge" markrgb="1 1 1"/>
    <material name="PlatformMat" texture="PlatformTex" texrepeat="1 1" texuniform="true" specular="0.1" shininess="0.1"/>
    <material name="geom" rgba="0.8 0.6 .4 1"/>
    <!-- Friction-tinted rubble appearances, assigned per geom at reset by
         mdp.events.randomize_rubble_sections. rgba stays neutral white so geom_rgba carries
         the ice/mud colour; these materials only add the glossy vs matte shading.
         MatIce gets a little emission so the pale colour survives the scene lighting. -->
    <material name="MatIce" rgba="1 1 1 1" emission="0.2" specular="0.9" shininess="0.9"/>
    <material name="MatMud" rgba="1 1 1 1" specular="0.0" shininess="0.02" reflectance="0"/>
    <texture name="SandTex" type="2d" builtin="flat" width="256" height="256" rgb1="0.76 0.66 0.45" rgb2="0.7 0.6 0.4" random="0.08"/>
    <material name="MatSand" texture="SandTex" texrepeat="4 4" specular="0.05" shininess="0.05"/>
    <!-- Drawn only when rendering (see build_scene_xml's visuals). -->
    <material name="MatWater" rgba="0.25 0.5 0.8 0.45" emission="0.1" specular="0.9" shininess="0.95" reflectance="0.35"/>
"""


# The icy slide's ice, glassier the less friction it has (terrain.parkour.ice_slide_material).
SHARED_ASSETS += "\n".join(
    f'    <material name="MatIceSlide{k}" rgba="1 1 1 1" emission="0.08" specular="0.9" '
    f'shininess="0.95" reflectance="{reflectance:g}"/>'
    for k, reflectance in enumerate(ICE_SLIDE_REFLECTANCE)
) + "\n"


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
        f'material="MatGrass" condim="3" {TERRAIN_GEOM_ATTRS} friction="1 .1 .1"/>',
    ]
    for side, y in (("left", -10.0), ("right", 10.0)):
        geoms.append(
            f'    <geom name="grass_{side}" type="box" size="{half_x:.3f} 5 0.05" '
            f'pos="{x_mid:.3f} {y} {box_z}" material="MatGrass" condim="3" {TERRAIN_GEOM_ATTRS} friction="1 .1 .1"/>'
        )
    for i, y in enumerate((-4.0, -2.0, 0.0, 2.0, 4.0)):
        material = "MatTrackLight" if i % 2 == 0 else "MatTrackDark"
        geoms.append(
            f'    <geom name="lane_{i + 1}" type="box" size="{half_x:.3f} 1 0.05" '
            f'pos="{x_mid:.3f} {y} {box_z}" material="{material}" condim="3" {TERRAIN_GEOM_ATTRS} friction="1 .1 .1"/>'
        )
    for i, y in enumerate((-3.0, -1.0, 1.0, 3.0)):
        geoms.append(
            f'    <geom name="lane_line_{i + 1}" type="box" size="{half_x:.3f} 0.05 0.001" '
            f'pos="{x_mid:.3f} {y} {line_z}" material="MatLine" condim="3" {TERRAIN_GEOM_ATTRS} friction="1 .1 .1"/>'
        )
    for name, x in (("start_line", 1.0), ("finish_line", course_length)):
        geoms.append(
            f'    <geom name="{name}" type="box" size="0.05 5 0.001" '
            f'pos="{x:.3f} 0 {line_z}" material="MatLine" condim="3" {TERRAIN_GEOM_ATTRS} friction="1 .1 .1"/>'
        )
    return geoms


def build_scene_xml(
    robot: RobotSpec,
    terrain: TerrainSpec,
    build_dir: Path,
    num_scan_markers: int = 0,
    timestep: float = 0.003,
    num_overhead_markers: int = 0,
    visuals: bool = False,
) -> Path:
    """Write the composed MJCF for ``robot`` on ``terrain`` and return its path.

    ``visuals`` adds decoration: a see-through body for each water volume and every geom
    marked ``visual`` (ropes, nets, painted arrows). None has contacts or mass, and all sit
    outside the scans' geom group, so they only change the picture; leave them out when
    nothing renders, as in training.
    """
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
    world.extend(box.to_xml() for box in terrain.boxes if visuals or not box.visual)
    if visuals:
        for water in terrain.water:
            # Drawn WATER_INSET inside the tank's walls and above its floor: faces lying exactly
            # on theirs flicker as the renderer alternates between the two.
            half_z = (water.surface_z - water.floor_z - WATER_INSET) / 2.0
            world.append(
                f'    <geom name="{water.name}" type="box" material="MatWater" contype="0" conaffinity="0" '
                f'group="{WATER_GEOM_GROUP}" '
                f'pos="{water.center[0]:.4f} {water.center[1]:.4f} {water.surface_z - half_z:.4f}" '
                f'size="{water.half_size[0] - WATER_INSET:.4f} {water.half_size[1] - WATER_INSET:.4f} {half_z:.4f}"/>'
            )
    world.append('    <site name="target_marker" type="sphere" size="0.15" rgba="0 1 1 0.7"/>')
    world.append('    <site name="target_marker_next" type="sphere" size="0.12" rgba="0 1 1 0.3"/>')
    for i in range(num_scan_markers):
        world.append(
            f'    <geom name="scan_viz_{i}" type="sphere" size="0.035" pos="0 0 -50" '
            f'rgba="{" ".join(str(c) for c in SCAN_MARKER_RGBA)}" contype="0" conaffinity="0" group="2"/>'
        )
    for index in range(num_overhead_markers):
        world.append(
            f'    <geom name="overhead_viz_{index}" type="sphere" size="0.035" pos="0 0 -50" '
            'rgba="0.1 0.9 1 0.8" contype="0" conaffinity="0" group="2"/>'
        )

    spawn = (terrain.spawn_pos[0], terrain.spawn_pos[1], terrain.spawn_pos[2] + robot.spawn_height)
    if robot.body_xml:
        world.append(robot.body_xml.format(x=spawn[0], y=spawn[1], z=spawn[2]))
    # The first-person camera, on a mocap mount the environment moves to a steadied eye every
    # frame it renders (camera.StableEyeCamera). No geoms, no joints: nothing in the physics.
    # Right after the robot's cameras, so the viewer's TAB reaches it after them.
    world.append(
        f'    <body name="{EYE_MOUNT}" mocap="true" pos="{spawn[0]} {spawn[1]} {spawn[2]}">'
        f'<camera name="{STABLE_EGOCENTRIC}" mode="fixed" xyaxes="0 -1 0 0 0 1"/></body>'
    )
    # After the robot, so its free joint keeps qpos[0:7] and its hinges come next.
    world.extend(body.to_xml(visuals) for body in terrain.bodies)

    includes = "\n".join(f'  <include file="{path}"/>' for path in robot.includes)
    # A jointed terrain part only meets the robot: rubbing on the static terrain it sits in
    # (a trampoline bed against its deck, an elevator between its platforms) would jam it.
    exclusions = "\n".join(
        ["  <contact>"]
        + [f'    <exclude body1="world" body2="{body.name}"/>' for body in terrain.bodies if body.joint]
        + ["  </contact>"]
    ) if any(body.joint for body in terrain.bodies) else ""
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
{exclusions}
{robot.extra_xml}
</mujoco>
"""
    path = build_dir / f"scene_{terrain.name}_{robot.name}.xml"
    path.write_text(xml)
    return path


def default_build_dir(tag: str = "parkour") -> Path:
    return Path(tempfile.mkdtemp(prefix=f"{tag}_scene_"))
