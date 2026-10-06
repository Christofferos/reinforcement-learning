"""List every registered parkour task and its key config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_parkour import tasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyword", default="", help="substring filter on the task id")
    args = parser.parse_args()

    rows = []
    for task_id, cfg_fn in tasks.TASKS.items():
        if args.keyword.lower() not in task_id.lower():
            continue
        cfg = cfg_fn()
        rows.append(
            (
                task_id,
                cfg.terrain.kind,
                f"{cfg.terrain.difficulty:.2f}",
                "yes" if cfg.terrain.curriculum else "no",
                "yes" if cfg.observation.height_scan else "no",
                str(cfg.observation.history_length),
            )
        )

    header = ("task", "terrain", "difficulty", "curriculum", "height-scan", "history")
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    print(f"\n{len(rows)} task(s). Ablations: {', '.join(sorted(tasks.ABLATIONS))}")


if __name__ == "__main__":
    main()
