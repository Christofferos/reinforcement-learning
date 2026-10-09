"""Rewrite stale curriculum state files of a resumed run from its training log.

Runs resumed before the difficulty-restore fix in ``train.py`` saved the difficulty that was
restored at startup in every later ``.state.json``, frozen for the whole run, while the
workers' real difficulty kept moving. The training log's ``progress/difficulty`` (mean over
recent episodes) was always correct. Every state file whose saved mean is more than one
difficulty step away from the logged value at the same timestep is rewritten with that logged
level, snapped to the difficulty grid, for all workers. Per-worker differences are not logged,
so they cannot be recovered.

Examples::

    python scripts/repair_state.py flat_0.0.8 --dry-run
    python scripts/repair_state.py flat_0.0.8
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def logged_difficulties(progress_csv) -> list[tuple[int, float]]:
    """(timestep, mean training difficulty) rows from a run's ``progress.csv``."""
    rows = []
    with open(progress_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            timestep, difficulty = row.get("time/total_timesteps"), row.get("progress/difficulty")
            if timestep and difficulty:
                rows.append((int(float(timestep)), float(difficulty)))
    return rows


def repair_run(model_dir, progress_csv, step: float = 0.05,
               dry_run: bool = False) -> list[tuple[Path, float, float]]:
    """Rewrite state files that disagree with the log; returns (path, saved mean, new level)."""
    log = logged_difficulties(progress_csv)
    if not log:
        raise ValueError(f"no progress/difficulty rows in {progress_csv}")
    repaired = []
    for path in sorted(Path(model_dir).glob("*.state.json")):
        state = json.loads(path.read_text())
        _, logged = min(log, key=lambda entry: abs(entry[0] - state["num_timesteps"]))
        saved = sum(state["difficulties"]) / len(state["difficulties"])
        if abs(saved - logged) <= step:
            continue
        level = round(round(logged / step) * step, 6)
        if not dry_run:
            state["difficulties"] = [level] * len(state["difficulties"])
            path.write_text(json.dumps(state, indent=2))
        repaired.append((path, saved, level))
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run name under models/, or a path to a run's model folder")
    parser.add_argument("--log", default=None,
                        help="progress.csv to read (default: the log folder recorded in run_config.json)")
    parser.add_argument("--dry-run", action="store_true", help="list the files that would change")
    args = parser.parse_args()

    model_dir = Path(args.run) if Path(args.run).is_dir() else ROOT / "models" / args.run
    if not model_dir.is_dir():
        parser.error(f"no model folder at {model_dir}")
    if args.log:
        progress_csv = Path(args.log)
    else:
        config_path = model_dir / "run_config.json"
        log_dir = json.loads(config_path.read_text()).get("log_directory") if config_path.exists() else None
        progress_csv = Path(log_dir) / "progress.csv" if log_dir else ROOT / "logs" / model_dir.name / "progress.csv"
    if not progress_csv.exists():
        parser.error(f"no training log at {progress_csv}; pass --log")

    repaired = repair_run(model_dir, progress_csv, dry_run=args.dry_run)
    verb = "would rewrite" if args.dry_run else "rewrote"
    for path, saved, level in repaired:
        print(f"{verb} {path.name}: saved {saved:.2f} -> logged {level:.2f}")
    print(f"{verb} {len(repaired)} state file(s) in {model_dir}")


if __name__ == "__main__":
    main()
