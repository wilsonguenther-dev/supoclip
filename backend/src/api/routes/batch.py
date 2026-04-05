"""Batch processing API routes for SupoClip.

Supports playlist/channel ingestion, priority lanes, deduplication,
and resume-on-failure for industrial-scale video processing.
"""
import hashlib
import json
import logging
import subprocess
from typing import Optional, List
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...admin_auth import get_optional_auth_user_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tasks/batch", tags=["batch"])


class BatchSubmission(BaseModel):
    youtube_playlist_url: Optional[str] = None
    youtube_channel_url: Optional[str] = None
    urls: Optional[List[str]] = None
    processing_mode: str = Field(default="balanced", pattern="^(fast|balanced|quality)$")
    priority: str = Field(default="normal", pattern="^(urgent|normal|background)$")
    auto_schedule: bool = False


class BatchStatus(BaseModel):
    batch_id: str
    status: str
    total_videos: int
    completed: int
    failed: int
    progress_pct: float
    videos: list


@router.post("")
async def create_batch(
    submission: BatchSubmission,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    user_id = get_optional_auth_user_id(request)

    urls: List[str] = []
    source_type = "url_list"
    source_url: Optional[str] = None

    if submission.urls:
        urls = submission.urls[:100]
        source_type = "url_list"
    elif submission.youtube_playlist_url:
        source_type = "playlist"
        source_url = submission.youtube_playlist_url
        urls = await _extract_playlist_urls(submission.youtube_playlist_url)
    elif submission.youtube_channel_url:
        source_type = "channel"
        source_url = submission.youtube_channel_url
        urls = await _extract_channel_urls(submission.youtube_channel_url)
    else:
        raise HTTPException(400, "Provide urls, youtube_playlist_url, or youtube_channel_url")

    if not urls:
        raise HTTPException(400, "No videos found")

    deduped = await _deduplicate(db, urls, submission.processing_mode)

    batch_result = await db.execute(text("""
        INSERT INTO batch_jobs (source_type, source_url, total_videos, priority, status, user_id)
        VALUES (:source_type, :source_url, :total, :priority, 'pending', :user_id)
        RETURNING id
    """), {
        "source_type": source_type,
        "source_url": source_url,
        "total": len(deduped),
        "priority": submission.priority,
        "user_id": user_id,
    })
    batch_id = str(batch_result.scalar())

    for url in deduped:
        await db.execute(text("""
            INSERT INTO batch_tasks (url, processing_mode, batch_id, status, priority)
            VALUES (:url, :mode, :batch_id, 'pending', :priority)
        """), {
            "url": url,
            "mode": submission.processing_mode,
            "batch_id": batch_id,
            "priority": _priority_to_int(submission.priority),
        })

    await db.commit()

    logger.info(f"Batch {batch_id} created: {len(deduped)} videos (deduped from {len(urls)})")

    return {
        "batch_id": batch_id,
        "total_videos": len(deduped),
        "skipped_duplicates": len(urls) - len(deduped),
        "priority": submission.priority,
        "status": "pending",
    }


@router.get("/{batch_id}")
async def get_batch_status(
    batch_id: str,
    db: AsyncSession = Depends(get_db),
):
    batch = await db.execute(text("""
        SELECT id, source_type, source_url, total_videos, priority, status,
               created_at, completed_at
        FROM batch_jobs WHERE id = :id
    """), {"id": batch_id})
    row = batch.mappings().first()
    if not row:
        raise HTTPException(404, "Batch not found")

    videos = await db.execute(text("""
        SELECT id, url, status, error, progress
        FROM batch_tasks WHERE batch_id = :batch_id
        ORDER BY created_at
    """), {"batch_id": batch_id})

    video_list = []
    completed = 0
    failed = 0
    for v in videos.mappings().all():
        video_list.append({
            "task_id": str(v["id"]),
            "url": v["url"],
            "status": v["status"],
            "error": v.get("error"),
        })
        if v["status"] == "completed":
            completed += 1
        elif v["status"] in ("failed", "error"):
            failed += 1

    total = row["total_videos"] or len(video_list)
    progress = (completed / max(total, 1)) * 100

    return {
        "batch_id": str(row["id"]),
        "status": row["status"],
        "total_videos": total,
        "completed": completed,
        "failed": failed,
        "progress_pct": round(progress, 1),
        "priority": row["priority"],
        "created_at": str(row["created_at"]),
        "videos": video_list,
    }


@router.get("/{batch_id}/manifest")
async def get_batch_manifest(
    batch_id: str,
    db: AsyncSession = Depends(get_db),
):
    videos = await db.execute(text("""
        SELECT bt.id as task_id, bt.url, bt.status,
               gc.id as clip_id, gc.file_path as clip_url,
               gc.virality_score, gc.duration, gc.filename as title
        FROM batch_tasks bt
        LEFT JOIN tasks t ON t.source_id IN (
            SELECT s.id FROM sources s WHERE s.url = bt.url
        )
        LEFT JOIN generated_clips gc ON gc.task_id = t.id
        WHERE bt.batch_id = :batch_id AND bt.status = 'completed'
        ORDER BY gc.virality_score DESC NULLS LAST
    """), {"batch_id": batch_id})

    manifest = {
        "batch_id": batch_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clips": [],
    }
    for row in videos.mappings().all():
        if row.get("clip_id"):
            manifest["clips"].append({
                "task_id": str(row["task_id"]),
                "source_url": row["url"],
                "clip_id": str(row["clip_id"]),
                "clip_url": row.get("clip_url"),
                "virality_score": row.get("virality_score"),
                "duration": row.get("duration"),
                "title": row.get("title"),
            })

    return manifest


@router.post("/{batch_id}/cancel")
async def cancel_batch(
    batch_id: str,
    db: AsyncSession = Depends(get_db),
):
    await db.execute(text("""
        UPDATE batch_tasks SET status = 'cancelled'
        WHERE batch_id = :batch_id AND status IN ('pending', 'queued')
    """), {"batch_id": batch_id})

    await db.execute(text("""
        UPDATE batch_jobs SET status = 'cancelled' WHERE id = :id
    """), {"id": batch_id})

    await db.commit()
    return {"batch_id": batch_id, "status": "cancelled"}


async def _extract_playlist_urls(playlist_url: str) -> List[str]:
    """Extract video URLs from a YouTube playlist using yt-dlp."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--flat-playlist", "-J", playlist_url],
            capture_output=True, text=True, timeout=120,
        )
        data = json.loads(result.stdout)
        entries = data.get("entries", [])
        return [f"https://www.youtube.com/watch?v={e['id']}" for e in entries if e.get("id")]
    except Exception as e:
        logger.error(f"Playlist extraction failed: {e}")
        return []


async def _extract_channel_urls(channel_url: str, limit: int = 50) -> List[str]:
    """Extract recent video URLs from a YouTube channel."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--flat-playlist", "-J", "--playlist-end", str(limit), channel_url],
            capture_output=True, text=True, timeout=120,
        )
        data = json.loads(result.stdout)
        entries = data.get("entries", [])
        return [f"https://www.youtube.com/watch?v={e['id']}" for e in entries if e.get("id")]
    except Exception as e:
        logger.error(f"Channel extraction failed: {e}")
        return []


async def _deduplicate(db: AsyncSession, urls: List[str], mode: str) -> List[str]:
    """Remove URLs already processed in the last 30 days with same mode."""
    deduped = []
    for url in urls:
        existing = await db.execute(text("""
            SELECT 1 FROM batch_tasks
            WHERE url = :url AND processing_mode = :mode
            AND status = 'completed'
            AND created_at > NOW() - INTERVAL '30 days'
            LIMIT 1
        """), {"url": url, "mode": mode})
        if not existing.scalar():
            deduped.append(url)
    return deduped


def _priority_to_int(priority: str) -> int:
    return {"urgent": 1, "normal": 5, "background": 10}.get(priority, 5)
