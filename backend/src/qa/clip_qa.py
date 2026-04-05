"""Automated clip quality assurance checks."""
import asyncio
import logging
import subprocess
import json
from pathlib import Path
from typing import List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class QAIssue:
    check: str
    severity: str  # "error", "warning", "info"
    message: str
    auto_fixed: bool = False

@dataclass
class QAResult:
    passed: bool
    issues: List[QAIssue] = field(default_factory=list)
    qa_score: float = 1.0  # 0.0 to 1.0

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

class ClipQA:
    SILENCE_RMS_THRESHOLD = 0.001
    BLACK_FRAME_THRESHOLD = 10  # luminance 0-255
    MIN_LUMINANCE = 20  # percentage
    MIN_DURATION = 3.0
    MAX_OVERLAP_RATIO = 0.7

    async def run(self, clip_path: Path, transcript_segment: dict | None = None) -> QAResult:
        checks = await asyncio.gather(
            self._check_audio(clip_path),
            self._check_black_frames(clip_path),
            self._check_luminance(clip_path),
            self._check_duration(clip_path),
            self._check_aspect_ratio(clip_path),
            return_exceptions=True,
        )

        issues: List[QAIssue] = []
        for result in checks:
            if isinstance(result, Exception):
                logger.warning(f"QA check failed: {result}")
                issues.append(QAIssue("internal", "warning", f"Check error: {str(result)[:100]}"))
            elif result:
                issues.extend(result)

        if transcript_segment:
            boundary = await self._check_sentence_boundary(transcript_segment)
            if boundary:
                issues.extend(boundary)

        has_errors = any(i.severity == "error" for i in issues)
        penalty = sum(0.15 if i.severity == "error" else 0.05 for i in issues)
        qa_score = max(0.0, 1.0 - penalty)

        return QAResult(passed=not has_errors, issues=issues, qa_score=qa_score)

    async def _check_audio(self, clip_path: Path) -> List[QAIssue]:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=codec_type",
                 "-of", "csv=p=0", str(clip_path)],
                capture_output=True, text=True, timeout=10,
            )
            if not result.stdout.strip():
                return [QAIssue("audio", "error", "No audio stream detected")]

            rms_result = subprocess.run(
                ["ffmpeg", "-i", str(clip_path), "-af", "volumedetect",
                 "-f", "null", "-"],
                capture_output=True, text=True, timeout=30,
            )
            stderr = rms_result.stderr
            if "mean_volume:" in stderr:
                vol_line = [l for l in stderr.split("\n") if "mean_volume:" in l]
                if vol_line:
                    mean_vol = float(vol_line[0].split("mean_volume:")[1].split("dB")[0].strip())
                    if mean_vol < -50:
                        return [QAIssue("audio", "warning", f"Very quiet audio ({mean_vol:.1f} dB)")]
        except Exception as e:
            logger.debug(f"Audio check error: {e}")
        return []

    async def _check_black_frames(self, clip_path: Path) -> List[QAIssue]:
        try:
            result = subprocess.run(
                ["ffmpeg", "-i", str(clip_path), "-vf",
                 f"blackdetect=d=0.3:pix_th={self.BLACK_FRAME_THRESHOLD / 255:.3f}",
                 "-f", "null", "-"],
                capture_output=True, text=True, timeout=30,
            )
            if "black_start" in result.stderr:
                lines = [l for l in result.stderr.split("\n") if "black_start" in l]
                if lines:
                    return [QAIssue("video", "warning", f"Black frames detected ({len(lines)} segment(s))")]
        except Exception:
            pass
        return []

    async def _check_luminance(self, clip_path: Path) -> List[QAIssue]:
        try:
            sig = subprocess.run(
                ["ffmpeg", "-i", str(clip_path), "-vf",
                 "signalstats,metadata=print:key=lavfi.signalstats.YAVG",
                 "-f", "null", "-"],
                capture_output=True, text=True, timeout=30,
            )
            if "lavfi.signalstats.YAVG" in sig.stderr:
                vals = []
                for line in sig.stderr.split("\n"):
                    if "lavfi.signalstats.YAVG" in line:
                        try:
                            vals.append(float(line.split("=")[-1].strip()))
                        except ValueError:
                            pass
                if vals:
                    avg_lum = sum(vals) / len(vals)
                    pct = (avg_lum / 255) * 100
                    if pct < self.MIN_LUMINANCE:
                        return [QAIssue("video", "warning", f"Poor lighting ({pct:.0f}% luminance)")]
        except Exception:
            pass
        return []

    async def _check_duration(self, clip_path: Path) -> List[QAIssue]:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(clip_path)],
                capture_output=True, text=True, timeout=10,
            )
            dur = float(result.stdout.strip())
            if dur < self.MIN_DURATION:
                return [QAIssue("duration", "error", f"Clip too short ({dur:.1f}s, minimum {self.MIN_DURATION}s)")]
        except Exception:
            pass
        return []

    async def _check_aspect_ratio(self, clip_path: Path) -> List[QAIssue]:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-of", "json", str(clip_path)],
                capture_output=True, text=True, timeout=10,
            )
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if streams:
                w, h = streams[0]["width"], streams[0]["height"]
                ratio = h / w if w > 0 else 0
                if abs(ratio - (16/9)) > 0.1 and abs(ratio - 1.0) > 0.1:
                    return [QAIssue("aspect_ratio", "warning", f"Unexpected aspect ratio {w}x{h}")]
        except Exception:
            pass
        return []

    async def _check_sentence_boundary(self, segment: dict) -> List[QAIssue]:
        text = segment.get("text", "").strip()
        if text and not text[-1] in ".!?\"'":
            return [QAIssue("boundary", "info", "Clip ends mid-sentence")]
        return []

    @staticmethod
    async def check_duplicates(clips: List[dict], overlap_threshold: float = 0.7) -> List[int]:
        remove_indices = []
        for i in range(len(clips)):
            for j in range(i + 1, len(clips)):
                s1, e1 = clips[i].get("start", 0), clips[i].get("end", 0)
                s2, e2 = clips[j].get("start", 0), clips[j].get("end", 0)
                overlap_start = max(s1, s2)
                overlap_end = min(e1, e2)
                if overlap_end > overlap_start:
                    overlap = overlap_end - overlap_start
                    shorter = min(e1 - s1, e2 - s2)
                    if shorter > 0 and overlap / shorter > overlap_threshold:
                        worse = j if clips[j].get("virality_score", 0) <= clips[i].get("virality_score", 0) else i
                        if worse not in remove_indices:
                            remove_indices.append(worse)
        return remove_indices
