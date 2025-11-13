"""Dataset generator for pooltool shots."""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from functools import partial
from multiprocessing import Pool, cpu_count

import pooltool as pt
from shot_utils import config
from shot_utils.rendering import encode_video, render_frames
from shot_utils.simulation import (build_system_one_ball_hit_cushion,
                                   extract_trajectories, simulate_shot)
from shot_utils.summary import summarize_system


def run_shot(
    shot_id: str,
    x: float,
    y: float,
    velocity: float,
    phi: float,
) -> dict[str, object]:
    outdir = config.BASE_OUTPUT / shot_id
    outdir.mkdir(parents=True, exist_ok=True)

    system = build_system_one_ball_hit_cushion(x, y, velocity, phi)
    simulate_shot(system, config.DURATION, config.FPS)

    df = extract_trajectories(system)
    df = df[df["t"] <= config.DURATION].copy()

    trajectory_path = outdir / f"trajectory_{shot_id}.csv"
    df.to_csv(trajectory_path, index=False)

    system_path = outdir / f"system_{shot_id}.json"
    system.save(system_path)

    metadata = {
        "shot_id": shot_id,
        "cue_start": {"x": x, "y": y},
        "velocity": velocity,
        "phi": phi,
        "fps": config.FPS,
        "duration": config.DURATION,
    }

    summary = summarize_system(system, metadata=metadata)
    summary_path = outdir / f"summary_{shot_id}.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    frames_dir = render_frames(system, outdir, config.FPS)
    video_path = outdir / f"video_{shot_id}.mp4"
    encode_video(frames_dir, config.FPS, video_path)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    return {
        "shot_id": shot_id,
        "cue_start": metadata["cue_start"],
        "velocity": velocity,
        "phi": phi,
        "paths": {
            "directory": str(outdir),
            "trajectory": str(trajectory_path),
            "system": str(system_path),
            "summary": str(summary_path),
            "video": str(video_path),
        },
    }


def _scaled_positions(table: pt.Table, num: int) -> list[tuple[float, float]]:
    num = max(1, num)
    grid = int(num**0.5)
    while grid * grid < num:
        grid += 1
    positions: list[tuple[float, float]] = []
    margin_x = 0.05 * table.w
    margin_y = 0.05 * table.l
    usable_w = table.w - 2 * margin_x
    usable_l = table.l - 2 * margin_y
    for i in range(grid):
        for j in range(grid):
            if len(positions) >= num:
                break
            fx = (i + 0.5) / grid
            fy = (j + 0.5) / grid
            x = margin_x + fx * usable_w
            y = margin_y + fy * usable_l
            positions.append((x, y))
    return positions


def _segment_values(low: float, high: float, num: int) -> list[float]:
    if num <= 1:
        return [low]
    step = (high - low) / (num - 1)
    return [low + i * step for i in range(num)]


def _segment_angles(num: int) -> list[float]:
    if num <= 1:
        return [0.0]
    step = 360.0 / num
    return [(i * step) % 360.0 for i in range(num)]


def main(processes: int | None = None) -> None:
    config.BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

    reference_table = pt.Table.default()
    positions = _scaled_positions(reference_table, num=10)
    velocities = _segment_values(1, 6, num=10)
    phis = _segment_angles(num=10)

    combos = list(itertools.product(positions, velocities, phis))
    tasks = [
        (f"shot_{idx:02d}", x, y, velocity, phi)
        for idx, ((x, y), velocity, phi) in enumerate(combos, start=1)
    ]

    worker = partial(_run_shot_from_tuple)
    proc_count = processes or cpu_count()
    with Pool(processes=proc_count) as pool:
        results = pool.map(worker, tasks)

    index_path = config.GLOBAL_INDEX_PATH
    with open(index_path, "w", encoding="utf-8") as fp:
        json.dump({"shots": results}, fp, indent=2)

    print(f"Global index written to {index_path}")
    print(f"Generated {len(results)} shots using {proc_count} processes")


def _run_shot_from_tuple(args: tuple[str, float, float, float, float]):
    shot_id, x, y, velocity, phi = args
    result = run_shot(shot_id, x, y, velocity, phi)
    print(
        f"Generated {shot_id}: position=({x:.2f}, {y:.2f}) velocity={velocity:.2f} phi={phi:.1f}",
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate poolshot dataset")
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of worker processes (defaults to CPU count)",
    )
    main(parser.parse_args().processes)
