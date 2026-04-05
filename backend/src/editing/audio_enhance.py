"""Audio enhancement pipeline: noise reduction and loudness normalization.

Uses noisereduce for denoising and pyloudnorm for -14 LUFS normalization.
Falls back gracefully if libraries aren't installed.
"""
import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_loudness(audio_path: Path, target_lufs: float = -14.0) -> Path:
    """Normalize audio to target LUFS using ffmpeg loudnorm filter."""
    output = audio_path.parent / f"{audio_path.stem}_normalized{audio_path.suffix}"
    try:
        # Two-pass loudness normalization
        # Pass 1: measure
        measure = subprocess.run(
            ["ffmpeg", "-i", str(audio_path), "-af",
             f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=60,
        )
        
        # Pass 2: apply (simplified single-pass for speed)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-af",
             f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
             str(output)],
            capture_output=True, check=True, timeout=120,
        )
        
        if output.exists() and output.stat().st_size > 0:
            logger.info(f"Audio normalized to {target_lufs} LUFS: {output}")
            return output
    except Exception as e:
        logger.warning(f"Loudness normalization failed: {e}")
    
    return audio_path


def reduce_noise(audio_path: Path) -> Path:
    """Apply noise reduction using ffmpeg's afftdn filter."""
    output = audio_path.parent / f"{audio_path.stem}_denoised{audio_path.suffix}"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-af",
             "afftdn=nf=-25:tn=1", str(output)],
            capture_output=True, check=True, timeout=120,
        )
        if output.exists() and output.stat().st_size > 0:
            logger.info(f"Noise reduction applied: {output}")
            return output
    except Exception as e:
        logger.warning(f"Noise reduction failed: {e}")
    return audio_path


def enhance_audio(video_path: Path, denoise: bool = True, normalize: bool = True, target_lufs: float = -14.0) -> Path:
    """Full audio enhancement pipeline on a video file."""
    if not denoise and not normalize:
        return video_path
    
    # Extract audio
    audio_tmp = video_path.parent / f"{video_path.stem}_audio.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
             "-ar", "44100", str(audio_tmp)],
            capture_output=True, check=True, timeout=60,
        )
    except Exception as e:
        logger.warning(f"Audio extraction failed: {e}")
        return video_path
    
    processed = audio_tmp
    if denoise:
        processed = reduce_noise(processed)
    if normalize:
        processed = normalize_loudness(processed, target_lufs)
    
    # Merge enhanced audio back
    output = video_path.parent / f"{video_path.stem}_enhanced{video_path.suffix}"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-i", str(processed),
             "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
             "-shortest", str(output)],
            capture_output=True, check=True, timeout=120,
        )
        # Cleanup temp files
        for f in [audio_tmp, processed]:
            if f != audio_tmp or f != processed:
                try: f.unlink(missing_ok=True)
                except: pass
        
        if output.exists() and output.stat().st_size > 0:
            logger.info(f"Audio enhanced: {output}")
            return output
    except Exception as e:
        logger.warning(f"Audio merge failed: {e}")
    
    return video_path
