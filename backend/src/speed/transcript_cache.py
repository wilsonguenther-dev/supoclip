"""Redis-backed transcript and analysis caching."""
import hashlib
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_TTL_DAYS = 30
CACHE_TTL = CACHE_TTL_DAYS * 86400

def compute_video_hash(video_url: str, file_size: int = 0) -> str:
    raw = f"{video_url}:{file_size}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]

class TranscriptCache:
    def __init__(self, redis_pool=None):
        self._redis = redis_pool

    async def get_transcript(self, video_hash: str) -> Optional[str]:
        if not self._redis:
            return None
        try:
            val = await self._redis.get(f"supoclip:transcript:{video_hash}")
            if val:
                logger.info(f"Transcript cache HIT: {video_hash[:8]}")
                return val.decode() if isinstance(val, bytes) else val
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        return None

    async def set_transcript(self, video_hash: str, transcript: str):
        if not self._redis:
            return
        try:
            await self._redis.set(f"supoclip:transcript:{video_hash}", transcript, ex=CACHE_TTL)
            logger.info(f"Transcript cached: {video_hash[:8]}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    async def get_analysis(self, video_hash: str, mode: str) -> Optional[str]:
        if not self._redis:
            return None
        try:
            val = await self._redis.get(f"supoclip:analysis:{video_hash}:{mode}")
            if val:
                logger.info(f"Analysis cache HIT: {video_hash[:8]}:{mode}")
                return val.decode() if isinstance(val, bytes) else val
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        return None

    async def set_analysis(self, video_hash: str, mode: str, analysis_json: str):
        if not self._redis:
            return
        try:
            await self._redis.set(f"supoclip:analysis:{video_hash}:{mode}", analysis_json, ex=CACHE_TTL)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
