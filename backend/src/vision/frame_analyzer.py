"""Analyze video frames with vision AI to detect energy peaks, emotions, and hooks.

Uses Gemini Vision or GPT-4o Vision to analyze frame grids.
Results merge with transcript analysis for better clip selection.
"""
import base64
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VisualAnalysis:
    energy_peaks: List[float] = field(default_factory=list)  # timestamps of high energy
    emotion_peaks: List[float] = field(default_factory=list)  # timestamps of high emotion
    text_segments: List[Dict[str, Any]] = field(default_factory=list)  # {timestamp, has_text}
    hook_frames: List[float] = field(default_factory=list)  # timestamps of visual hooks
    speaker_count: int = 1
    has_broll: bool = False
    avg_cut_frequency: float = 0.0  # cuts per minute
    face_timestamps: List[Dict[str, float]] = field(default_factory=list)  # [{timestamp, confidence}]

    def get_energy_boost(self, clip_start: float, clip_end: float) -> float:
        overlaps = sum(1 for t in self.energy_peaks if clip_start <= t <= clip_end)
        return min(overlaps * 0.05, 0.15)

    def get_emotion_boost(self, clip_start: float, clip_end: float) -> float:
        overlaps = sum(1 for t in self.emotion_peaks if clip_start <= t <= clip_end)
        return min(overlaps * 0.03, 0.10)

    def has_face_at(self, timestamp: float, tolerance: float = 2.0) -> bool:
        return any(abs(f["timestamp"] - timestamp) < tolerance for f in self.face_timestamps)


VISION_ANALYSIS_PROMPT = """Analyze these video frames extracted at the timestamps shown.

For each frame, identify:
1. ENERGY LEVEL (1-10): gesture intensity, motion blur, facial expression intensity
2. EMOTION: what emotion is being expressed (neutral, excited, surprised, angry, sad, thoughtful)
3. FACES: how many faces visible, are they looking at camera?
4. TEXT: any on-screen text, slides, or graphics?
5. HOOK POTENTIAL: would this frame make someone stop scrolling? (yes/no + why)
6. B-ROLL: is this a speaker shot or B-roll/cutaway?

Return JSON:
{
  "frames": [
    {
      "timestamp": 0.0,
      "energy": 7,
      "emotion": "excited",
      "face_count": 1,
      "looking_at_camera": true,
      "has_text": false,
      "hook_potential": true,
      "hook_reason": "speaker pointing directly at camera with intense expression",
      "is_broll": false
    }
  ],
  "summary": {
    "speaker_count": 1,
    "has_broll": false,
    "avg_energy": 6.5,
    "recommended_clip_style": "fast_cuts"
  }
}"""


async def analyze_frames_batch(
    frames: List[tuple[float, Path]],
    provider: str = "google",
) -> VisualAnalysis:
    if not frames:
        return VisualAnalysis()

    try:
        if provider == "google":
            return await _analyze_with_gemini(frames)
        elif provider == "openai":
            return await _analyze_with_openai(frames)
        else:
            logger.warning(f"Unknown vision provider: {provider}, falling back to google")
            return await _analyze_with_gemini(frames)
    except Exception as e:
        logger.error(f"Frame analysis failed: {e}")
        return VisualAnalysis()


async def _analyze_with_gemini(frames: List[tuple[float, Path]]) -> VisualAnalysis:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not set, skipping frame analysis")
        return VisualAnalysis()

    import httpx

    image_parts = []
    timestamp_map = []
    for ts, path in frames[:20]:  # limit to 20 frames for API
        if path.exists():
            data = path.read_bytes()
            b64 = base64.b64encode(data).decode()
            image_parts.append({
                "inline_data": {"mime_type": "image/jpeg", "data": b64}
            })
            image_parts.append({"text": f"[Frame at {ts:.1f}s]"})
            timestamp_map.append(ts)

    if not image_parts:
        return VisualAnalysis()

    body = {
        "contents": [{
            "parts": [{"text": VISION_ANALYSIS_PROMPT}] + image_parts,
        }],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            json=body,
        )
        resp.raise_for_status()
        result = resp.json()

    text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    return _parse_vision_response(text, timestamp_map)


async def _analyze_with_openai(frames: List[tuple[float, Path]]) -> VisualAnalysis:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set, skipping frame analysis")
        return VisualAnalysis()

    import httpx

    content = [{"type": "text", "text": VISION_ANALYSIS_PROMPT}]
    timestamp_map = []
    for ts, path in frames[:10]:  # OpenAI limit
        if path.exists():
            data = path.read_bytes()
            b64 = base64.b64encode(data).decode()
            content.append({
                "type": "text", "text": f"[Frame at {ts:.1f}s]"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
            })
            timestamp_map.append(ts)

    if len(timestamp_map) == 0:
        return VisualAnalysis()

    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4096,
        "temperature": 0.2,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        result = resp.json()

    text = result["choices"][0]["message"]["content"]
    return _parse_vision_response(text, timestamp_map)


def _parse_vision_response(text: str, timestamp_map: List[float]) -> VisualAnalysis:
    analysis = VisualAnalysis()

    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
        else:
            return analysis

        frames_data = data.get("frames", [])
        summary = data.get("summary", {})

        for frame in frames_data:
            ts = frame.get("timestamp", 0)
            energy = frame.get("energy", 5)
            emotion = frame.get("emotion", "neutral")
            hook = frame.get("hook_potential", False)
            face_count = frame.get("face_count", 0)
            is_broll = frame.get("is_broll", False)

            if energy >= 7:
                analysis.energy_peaks.append(ts)
            if emotion in ("excited", "surprised", "angry"):
                analysis.emotion_peaks.append(ts)
            if hook:
                analysis.hook_frames.append(ts)
            if frame.get("has_text"):
                analysis.text_segments.append({"timestamp": ts, "has_text": True})
            if face_count > 0:
                analysis.face_timestamps.append({"timestamp": ts, "confidence": 0.9})
            if is_broll:
                analysis.has_broll = True

        analysis.speaker_count = summary.get("speaker_count", 1)
        analysis.avg_cut_frequency = len(frames_data) / max(
            (max(timestamp_map) - min(timestamp_map)) / 60, 1
        ) if timestamp_map and len(timestamp_map) > 1 else 0

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning(f"Failed to parse vision response: {e}")

    logger.info(
        f"Vision analysis: {len(analysis.energy_peaks)} energy peaks, "
        f"{len(analysis.emotion_peaks)} emotion peaks, "
        f"{len(analysis.hook_frames)} hooks, "
        f"speakers={analysis.speaker_count}"
    )
    return analysis
