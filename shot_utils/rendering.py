from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pooltool.ani.animate import FrameStepper
from pooltool.ani.camera import CameraState, camera_states
from pooltool.ani.image import ImageExt, ImageZip, save_images

from . import config


def render_frames(system, outdir: Path, fps: int) -> Path:
    frames_dir = outdir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    interface = FrameStepper()
    exporter = ImageZip(path=frames_dir, ext=ImageExt.PNG,
                        prefix=config.FRAME_PREFIX, compress=False)

    try:
        camera_state = camera_states[config.CAMERA_NAME]
        offset = getattr(config, "CAMERA_DISTANCE_OFFSET", 0.0)
        if offset:
            cam_pos = (
                camera_state.cam_pos[0] + offset,
                camera_state.cam_pos[1],
                camera_state.cam_pos[2],
            )
            camera_state = CameraState(
                cam_hpr=camera_state.cam_hpr,
                cam_pos=cam_pos,
                fixation_hpr=camera_state.fixation_hpr,
                fixation_pos=camera_state.fixation_pos,
            )

        save_images(
            exporter=exporter,
            system=system,
            interface=interface,
            size=config.FRAME_SIZE,
            fps=fps,
            camera_state=camera_state,
            gray=False,
            show_hud=False,
        )
    finally:
        interface.destroy()

    return frames_dir


def encode_video(frames_dir: Path, fps: int, video_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / config.FRAME_PATTERN),
        "-pix_fmt", "yuv420p",
        "-crf", "20",                      # optional: slightly higher CRF for smaller 240p files
        "-vf", "scale=-2:240:flags=lanczos,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags", "+faststart",         # optional: better web playback
        str(video_path),
    ]

    subprocess.run(cmd, check=True)
