"""Run a single debug shot to verify the pipeline."""

from __future__ import annotations

import pooltool as pt

from main import _scaled_positions, run_shot
from shot_utils import config


def main() -> None:
    config.BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    reference_table = pt.Table.default()
    safe_position = _scaled_positions(reference_table, num=1)[0]
    entry = run_shot("debug_shot", safe_position[0], safe_position[1], 1.8, 0.0)
    print(f"Debug shot stored in {entry['paths']['directory']}")


if __name__ == "__main__":
    main()
