"""Select Whisper model size based on video duration and processing mode."""
import logging

logger = logging.getLogger(__name__)

def select_whisper_model(duration_seconds: float, processing_mode: str = "balanced") -> str:
    if processing_mode == "quality":
        selected = "medium"
    elif processing_mode == "fast":
        selected = "base"
    elif duration_seconds < 300:  # under 5 min
        selected = "base"
    elif duration_seconds < 1800:  # under 30 min
        selected = "small"
    else:
        selected = "medium"

    logger.info(f"Whisper model selected: {selected} (duration={duration_seconds:.0f}s, mode={processing_mode})")
    return selected
