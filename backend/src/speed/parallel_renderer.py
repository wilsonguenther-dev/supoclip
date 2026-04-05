"""Parallel clip rendering with semaphore-based concurrency control."""
import asyncio
import logging
import os
from typing import List, Dict, Any, Callable, Optional, Awaitable
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_CONCURRENT_RENDERS = int(os.getenv("MAX_CONCURRENT_RENDERS", "3"))

class ParallelRenderer:
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_RENDERS):
        self._sem = asyncio.Semaphore(max_concurrent)
        self._completed = 0
        self._total = 0

    async def render_all(
        self,
        segments: List[Dict[str, Any]],
        render_fn: Callable[[Dict[str, Any]], Awaitable[Path]],
        progress_callback: Optional[Callable[[int, str, str], Awaitable[None]]] = None,
        clip_ready_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> List[Path]:
        self._total = len(segments)
        self._completed = 0

        async def _render_one(seg: Dict[str, Any]) -> Path:
            async with self._sem:
                result = await render_fn(seg)
                self._completed += 1
                pct = int(70 + (self._completed / self._total) * 25)
                if progress_callback:
                    await progress_callback(pct, f"Rendered clip {self._completed}/{self._total}", "processing")
                if clip_ready_callback:
                    await clip_ready_callback({"path": str(result), "index": self._completed - 1})
                return result

        results = await asyncio.gather(*[_render_one(s) for s in segments], return_exceptions=True)

        paths = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"Clip {i} render failed: {r}")
            else:
                paths.append(r)
        return paths
