"""Color grading presets applied via FFmpeg filters.

Presets: cinematic, bright, moody, vintage, clean.
Applied non-destructively to clip output.
"""
import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PRESETS = {
    "cinematic": "eq=contrast=1.1:saturation=0.95:brightness=0.0,colorbalance=rs=0.02:gs=-0.01:bs=0.05,vignette=PI/4",
    "bright": "eq=brightness=0.05:saturation=1.15:contrast=1.05",
    "moody": "eq=brightness=-0.05:saturation=0.85:contrast=1.15,colorbalance=rs=-0.03:gs=-0.02:bs=0.04",
    "vintage": "eq=saturation=0.7:contrast=1.1,colorbalance=rs=0.06:gs=0.02:bs=-0.04,curves=vintage",
    "clean": "eq=contrast=1.02:saturation=1.05:brightness=0.02",
    "none": None,
}


def apply_color_grade(video_path: Path, preset: str = "none") -> Path:
    """Apply a color grading preset to a video clip."""
    if preset == "none" or preset not in PRESETS or PRESETS[preset] is None:
        return video_path
    
    vf_filter = PRESETS[preset]
    output = video_path.parent / f"{video_path.stem}_{preset}{video_path.suffix}"
    
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vf", vf_filter,
             "-c:a", "copy", str(output)],
            capture_output=True, check=True, timeout=120,
        )
        if output.exists() and output.stat().st_size > 0:
            logger.info(f"Color grade '{preset}' applied: {output}")
            return output
    except Exception as e:
        logger.warning(f"Color grading failed ({preset}): {e}")
    
    return video_path


def list_presets() -> list[str]:
    return [k for k in PRESETS if k != "none"]
