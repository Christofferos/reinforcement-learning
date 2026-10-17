"""Render one panel per Procedural terrain module at easy, medium and hard difficulty.

Run from ``gymnasium/6-Parkour``::

    python scripts/render_terrain_sections.py

Each panel is rendered in its own process because repeated offscreen MuJoCo renderers can
produce black frames on macOS.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from g1_parkour.terrain.parkour import EXPERIMENTAL_SKILLS, MODULES  # noqa: E402

TERRAIN_DIR = Path(__file__).resolve().parents[1] / "g1_parkour" / "terrain"
DIFFICULTIES = (("easy", 0.15), ("medium", 0.5), ("hard", 1.0))
SECTIONS_PER_ROW = 3
SECTION_WIDTH = 720
VIEW_HEIGHT = 400
LABEL_HEIGHT = 50
TITLE_HEIGHT = 66
BACKGROUND = (241, 243, 245)


@dataclass(frozen=True)
class Section:
    skill: str
    title: str
    ramp_direction: str | None = None


def sections() -> list[Section]:
    result = []
    for skill in MODULES:
        if skill == "ramp":
            result.extend((Section(skill, "Ramp Up", "up"), Section(skill, "Ramp Down", "down")))
        else:
            result.append(Section(skill, skill.replace("_", " ").title()))
    return result


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def render_section(
    section: Section, difficulty: float, output: Path, seed: int,
) -> tuple[int, float, float]:
    import mujoco

    from g1_parkour import tasks
    from g1_parkour.env import ParkourEnv

    cfg = tasks.parkour_cfg()
    cfg.terrain.params = {"num_modules": 1, "skills": (section.skill,)}
    cfg.terrain.difficulty = difficulty
    cfg.terrain.curriculum = False
    cfg.terrain.resample_every_n_resets = 1
    cfg.events.push_robot = False
    cfg.events.reset_noise_scale = 0.0
    cfg.events.reset_yaw_range = 0.0
    cfg.debug_scan_markers = False

    env = ParkourEnv(cfg, render_mode="rgb_array", width=SECTION_WIDTH, height=VIEW_HEIGHT)
    try:
        selected_seed = seed
        for selected_seed in range(seed, seed + 256):
            env.reset(seed=selected_seed)
            exit_height = float(env._waypoints[-1, 2] - env.terrain.spawn_pos[2])
            if section.ramp_direction is None or (
                exit_height > 0 if section.ramp_direction == "up" else exit_height < 0
            ):
                break
        else:
            raise RuntimeError(f"Could not find a {section.ramp_direction} ramp seed")

        length = env.terrain.course_length
        y_min = min(box.pos[1] - box.size[1] for box in env.terrain.boxes)
        y_max = max(box.pos[1] + box.size[1] for box in env.terrain.boxes)

        env.render()
        viewer = env._mujoco_renderer._get_viewer("rgb_array")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = (length / 2.0, (y_min + y_max) / 2.0, 1.0)
        viewer.cam.distance = max(8.0, length * 0.8)
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0
        for waypoint in env._waypoints:
            viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=waypoint + (0.0, 0.0, 0.3),
                size=(0.08, 0.08, 0.08),
                rgba=(0.0, 0.8, 1.0, 0.9),
                emission=0.6,
                label="",
            )
        Image.fromarray(env.render()).save(output)
        return selected_seed, length, exit_height
    finally:
        env.close()


def compose(difficulty: float, rendered: list[tuple[Section, Path, int, float, float]], output: Path) -> None:
    rows = math.ceil(len(rendered) / SECTIONS_PER_ROW)
    image = Image.new(
        "RGB",
        (SECTION_WIDTH * SECTIONS_PER_ROW, TITLE_HEIGHT + rows * (LABEL_HEIGHT + VIEW_HEIGHT)),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(image)
    title_font, subtitle_font = font(27), font(15)
    label_font, details_font = font(19), font(14)
    draw.text((18, 7), f"Procedural sections | Difficulty {difficulty:.2f}", fill=(28, 37, 46), font=title_font)
    draw.text(
        (18, 39),
        "Actual MuJoCo geometry. Cyan dots: route waypoints. Humanoid: scale reference. Experimental modules marked.",
        fill=(65, 79, 92),
        font=subtitle_font,
    )

    for index, (section, path, seed, course_length, exit_height) in enumerate(rendered):
        column, row = index % SECTIONS_PER_ROW, index // SECTIONS_PER_ROW
        x, y = column * SECTION_WIDTH, TITLE_HEIGHT + row * (LABEL_HEIGHT + VIEW_HEIGHT)
        draw.text((x + 12, y + 2), section.title, fill=(28, 37, 46), font=label_font)
        if section.skill in EXPERIMENTAL_SKILLS:
            draw.text(
                (x + 12 + int(draw.textlength(section.title, font=label_font)) + 12, y + 5),
                "EXPERIMENTAL",
                fill=(150, 55, 35),
                font=details_font,
            )
        draw.text(
            (x + 12, y + 27),
            f"Seed {seed} | Course {course_length:.2f} m | Exit height {exit_height:+.2f} m",
            fill=(65, 79, 92),
            font=details_font,
        )
        with Image.open(path) as panel:
            image.paste(panel.convert("RGB"), (x, y + LABEL_HEIGHT))
    image.save(output)
    print(f"Saved {output} ({len(rendered)} sections)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=TERRAIN_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-one", choices=tuple(MODULES), help=argparse.SUPPRESS)
    parser.add_argument("--difficulty", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--ramp-direction", choices=("up", "down"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.render_one:
        if args.difficulty is None or args.output is None:
            parser.error("--render-one requires --difficulty and --output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        section = Section(args.render_one, args.render_one.replace("_", " ").title(), args.ramp_direction)
        selected_seed, length, exit_height = render_section(section, args.difficulty, args.output, args.seed)
        args.output.with_suffix(".txt").write_text(
            f"{selected_seed} {exit_height:.8f} {length:.8f}\n",
            encoding="ascii",
        )
        print(f"{args.render_one}: seed {selected_seed}, exit height {exit_height:+.2f} m")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    section_list = sections()
    with tempfile.TemporaryDirectory(prefix="terrain-sections-") as temp_dir:
        temp = Path(temp_dir)
        for difficulty_name, difficulty in DIFFICULTIES:
            rendered = []
            for index, section in enumerate(section_list):
                panel_path = temp / f"{difficulty_name}-{index}.png"
                print(f"Rendering {difficulty_name}: {section.title}", flush=True)
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--render-one",
                    section.skill,
                    "--difficulty",
                    str(difficulty),
                    "--seed",
                    str(args.seed),
                    "--output",
                    str(panel_path),
                ]
                if section.ramp_direction is not None:
                    command.extend(("--ramp-direction", section.ramp_direction))
                subprocess.run(command, check=True)
                selected_seed, exit_height, length = parse_worker_result(section.skill, panel_path)
                rendered.append((section, panel_path, selected_seed, length, exit_height))
            compose(
                difficulty,
                rendered,
                args.output_dir / f"terrain-sections-{difficulty_name}.png",
            )


def parse_worker_result(skill: str, panel_path: Path) -> tuple[int, float, float]:
    result_path = panel_path.with_suffix(".txt")
    if not result_path.exists():
        raise RuntimeError(f"Renderer did not report metadata for {skill}")
    seed_text, height_text, length_text = result_path.read_text(encoding="ascii").split()
    return int(seed_text), float(height_text), float(length_text)


if __name__ == "__main__":
    main()
