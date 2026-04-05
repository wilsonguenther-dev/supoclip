"""Brand bumper management — prepend/append intro/outro to clips.

Supports configurable 1-3 second video bumpers via env vars or per-task config.
"""
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_intro_bumper() -> Optional[Path]:
    path = os.getenv("INTRO_BUMPER_PATH")
    if path:
        p = Path(path)
        if p.exists():
            return p
        logger.warning(f"Intro bumper not found: {path}")
    return None


def get_outro_bumper() -> Optional[Path]:
    path = os.getenv("OUTRO_BUMPER_PATH")
    if path:
        p = Path(path)
        if p.exists():
            return p
        logger.warning(f"Outro bumper not found: {path}")
    return None


def apply_bumpers(
    clip_path: Path,
    intro: Optional[Path] = None,
    outro: Optional[Path] = None,
) -> Path:
    """Concatenate intro + clip + outro using ffmpeg concat demuxer."""
    intro = intro or get_intro_bumper()
    outro = outro or get_outro_bumper()
    
    if not intro and not outro:
        return clip_path
    
    parts = []
    if intro:
        parts.append(intro)
    parts.append(clip_path)
    if outro:
        parts.append(outro)
    
    if len(parts) == 1:
        return clip_path
    
    # Create concat list file
    concat_file = clip_path.parent / f"{clip_path.stem}_concat.txt"
    output = clip_path.parent / f"{clip_path.stem}_bumpered{clip_path.suffix}"
    
    try:
        with open(concat_file, "w") as f:
            for part in parts:
                f.write(f"file '{part}'\n")
        
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_file), "-c", "copy", str(output)],
            capture_output=True, check=True, timeout=60,
        )
        
        concat_file.unlink(missing_ok=True)
        
        if output.exists() and output.stat().st_size > 0:
            logger.info(f"Bumpers applied: {output}")
            return output
    except Exception as e:
        logger.warning(f"Bumper application failed: {e}")
        concat_file.unlink(missing_ok=True)
    
    return clip_path
