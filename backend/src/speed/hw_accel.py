"""Hardware acceleration detection for FFmpeg."""
import platform
import shutil
import subprocess
import logging
import os

logger = logging.getLogger(__name__)

def detect_hw_accel() -> str | None:
    override = os.getenv("HW_ACCEL", "auto")
    if override != "auto":
        return override if override != "none" else None

    system = platform.system()
    if system == "Darwin":
        return "videotoolbox"
    if system == "Linux" and shutil.which("nvidia-smi"):
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True, timeout=5)
            return "cuda"
        except Exception:
            pass
    return None

def get_ffmpeg_hw_args(accel: str | None) -> list[str]:
    if accel == "videotoolbox":
        return ["-hwaccel", "videotoolbox"]
    if accel == "cuda":
        return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    return []

_detected = None
def get_accel():
    global _detected
    if _detected is None:
        _detected = detect_hw_accel()
        logger.info(f"Hardware acceleration: {_detected or 'none'}")
    return _detected
