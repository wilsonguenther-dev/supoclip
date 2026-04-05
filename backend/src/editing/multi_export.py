"""Parallel multi-format export — render once, transcode in parallel.

Exports TikTok (9:16), Reels (9:16), Shorts (9:16), LinkedIn (1:1)
simultaneously from a single source clip.
"""
import asyncio
import logging
import subprocess
import os
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MAX_CONCURRENT_EXPORTS = int(os.getenv("MAX_CONCURRENT_EXPORTS", "4"))


@dataclass
class ExportPreset:
    name: str
    width: int
    height: int
    max_duration: int
    fps: int = 30
    bitrate: str = "5M"
    codec: str = "libx264"


PRESETS: Dict[str, ExportPreset] = {
    "tiktok": ExportPreset("TikTok", 1080, 1920, 180, bitrate="6M"),
    "instagram_reels": ExportPreset("Instagram Reels", 1080, 1920, 90, bitrate="5M"),
    "youtube_shorts": ExportPreset("YouTube Shorts", 1080, 1920, 60, bitrate="8M"),
    "linkedin": ExportPreset("LinkedIn", 1080, 1080, 600, bitrate="5M"),
    "x_twitter": ExportPreset("X / Twitter", 1080, 1920, 140, bitrate="5M"),
    "landscape": ExportPreset("Landscape", 1920, 1080, 600, bitrate="8M"),
}


async def export_all_formats(
    source_clip: Path,
    presets: List[str] | None = None,
    output_dir: Path | None = None,
) -> Dict[str, Path]:
    """Export a clip to all requested formats in parallel."""
    if presets is None:
        presets = ["tiktok", "instagram_reels", "youtube_shorts"]
    
    out = output_dir or source_clip.parent
    out.mkdir(parents=True, exist_ok=True)
    
    sem = asyncio.Semaphore(MAX_CONCURRENT_EXPORTS)
    
    async def _export_one(preset_name: str) -> tuple[str, Path | None]:
        if preset_name not in PRESETS:
            return preset_name, None
        preset = PRESETS[preset_name]
        output = out / f"{source_clip.stem}_{preset_name}{source_clip.suffix}"
        
        async with sem:
            return preset_name, await _transcode(source_clip, output, preset)
    
    results = await asyncio.gather(*[_export_one(p) for p in presets])
    
    exports = {}
    for name, path in results:
        if path:
            exports[name] = path
            logger.info(f"Exported {name}: {path}")
    
    return exports


async def _transcode(source: Path, output: Path, preset: ExportPreset) -> Path | None:
    """Transcode a clip to the target preset using ffmpeg."""
    try:
        vf = f"scale={preset.width}:{preset.height}:force_original_aspect_ratio=decrease,pad={preset.width}:{preset.height}:(ow-iw)/2:(oh-ih)/2"
        
        cmd = [
            "ffmpeg", "-y", "-i", str(source),
            "-vf", vf,
            "-c:v", preset.codec,
            "-b:v", preset.bitrate,
            "-r", str(preset.fps),
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(output),
        ]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        
        if proc.returncode == 0 and output.exists() and output.stat().st_size > 0:
            return output
        else:
            logger.warning(f"Transcode failed for {preset.name}: {stderr.decode()[:200]}")
    except asyncio.TimeoutError:
        logger.warning(f"Transcode timed out for {preset.name}")
    except Exception as e:
        logger.warning(f"Transcode error for {preset.name}: {e}")
    
    return None
