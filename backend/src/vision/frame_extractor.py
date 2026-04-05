"""Extract key frames from video for visual intelligence analysis.

Strategies:
- uniform: one frame every N seconds
- scene_change: detect scene transitions via frame difference
- combined: uniform sampling + scene change detection (default)
"""
import logging
import subprocess
import json
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

MAX_FRAMES = 50
UNIFORM_INTERVAL = 10  # seconds


def extract_key_frames(
    video_path: Path,
    strategy: str = "combined",
    max_frames: int = MAX_FRAMES,
    interval: int = UNIFORM_INTERVAL,
) -> List[Tuple[float, Path]]:
    """Extract key frames and return list of (timestamp_seconds, frame_path)."""
    output_dir = video_path.parent / f"{video_path.stem}_frames"
    output_dir.mkdir(exist_ok=True)

    duration = _get_duration(video_path)
    if duration is None or duration < 1:
        logger.warning(f"Cannot determine video duration: {video_path}")
        return []

    if strategy == "uniform":
        return _extract_uniform(video_path, output_dir, duration, interval, max_frames)
    elif strategy == "scene_change":
        return _extract_scene_change(video_path, output_dir, max_frames)
    else:  # combined
        uniform = _extract_uniform(video_path, output_dir, duration, interval, max_frames // 2)
        scenes = _extract_scene_change(video_path, output_dir, max_frames // 2)
        combined = _merge_and_dedupe(uniform + scenes, min_gap=3.0)
        return combined[:max_frames]


def _get_duration(video_path: Path) -> Optional[float]:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def _extract_uniform(
    video_path: Path, output_dir: Path, duration: float,
    interval: int, max_frames: int,
) -> List[Tuple[float, Path]]:
    frames = []
    timestamps = [i * interval for i in range(int(duration // interval) + 1)]
    timestamps = timestamps[:max_frames]

    for ts in timestamps:
        frame_path = output_dir / f"frame_{ts:.1f}.jpg"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(ts), "-i", str(video_path),
                 "-vframes", "1", "-q:v", "3", str(frame_path)],
                capture_output=True, timeout=10,
            )
            if frame_path.exists() and frame_path.stat().st_size > 0:
                frames.append((ts, frame_path))
        except Exception as e:
            logger.debug(f"Frame extraction failed at {ts}s: {e}")

    logger.info(f"Extracted {len(frames)} uniform frames")
    return frames


def _extract_scene_change(
    video_path: Path, output_dir: Path, max_frames: int,
) -> List[Tuple[float, Path]]:
    """Detect scene changes using ffmpeg's select filter."""
    frames = []
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "frame=pts_time",
             "-of", "csv=p=0",
             "-f", "lavfi",
             f"movie={str(video_path)},select='gt(scene\\,0.3)'"],
            capture_output=True, text=True, timeout=60,
        )
        timestamps = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    timestamps.append(float(line))
                except ValueError:
                    pass

        timestamps = timestamps[:max_frames]
        for ts in timestamps:
            frame_path = output_dir / f"scene_{ts:.1f}.jpg"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(ts), "-i", str(video_path),
                     "-vframes", "1", "-q:v", "3", str(frame_path)],
                    capture_output=True, timeout=10,
                )
                if frame_path.exists() and frame_path.stat().st_size > 0:
                    frames.append((ts, frame_path))
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Scene change detection failed: {e}")

    logger.info(f"Extracted {len(frames)} scene-change frames")
    return frames


def _merge_and_dedupe(
    frames: List[Tuple[float, Path]], min_gap: float = 3.0,
) -> List[Tuple[float, Path]]:
    if not frames:
        return []
    sorted_frames = sorted(frames, key=lambda x: x[0])
    result = [sorted_frames[0]]
    for ts, path in sorted_frames[1:]:
        if ts - result[-1][0] >= min_gap:
            result.append((ts, path))
    return result


def cleanup_frames(video_path: Path):
    output_dir = video_path.parent / f"{video_path.stem}_frames"
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
