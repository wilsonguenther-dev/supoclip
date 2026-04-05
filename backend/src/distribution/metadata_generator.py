"""AI-generated distribution metadata for each clip.

Per-clip: platform-specific captions, hashtags, best post times,
A/B hook variants, and virality score breakdowns.
"""
import asyncio
import json
import logging
import os
from typing import List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PlatformMeta:
    caption: str = ""
    hashtags: List[str] = field(default_factory=list)
    best_post_time: str = ""
    hook_score: int = 0


@dataclass
class HookVariant:
    text: str = ""
    style: str = "bold"
    expected_ctr: float = 0.0


@dataclass
class ViralityBreakdown:
    hook_quality: int = 0
    engagement: int = 0
    shareability: int = 0
    value: int = 0
    visual_energy_bonus: int = 0
    total: int = 0


@dataclass
class ClipDistributionMeta:
    clip_id: str = ""
    tiktok: PlatformMeta = field(default_factory=PlatformMeta)
    instagram: PlatformMeta = field(default_factory=PlatformMeta)
    youtube_shorts: PlatformMeta = field(default_factory=PlatformMeta)
    linkedin: PlatformMeta = field(default_factory=PlatformMeta)
    x_twitter: PlatformMeta = field(default_factory=PlatformMeta)
    hook_variants: List[HookVariant] = field(default_factory=list)
    virality_breakdown: ViralityBreakdown = field(default_factory=ViralityBreakdown)


DISTRIBUTION_PROMPT = """You are a social media strategist for viral short-form video.

For this clip, generate platform-specific distribution metadata.

Clip info:
- Title: {title}
- Duration: {duration}s
- Transcript excerpt: {transcript_excerpt}
- Virality score: {virality_score}
- Content type: {content_type}

Return JSON:
{{
  "tiktok": {{
    "caption": "...",
    "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"],
    "best_post_time": "7PM EST Tuesday",
    "hook_score": 85
  }},
  "instagram": {{
    "caption": "...",
    "hashtags": ["#tag1", "#tag2"],
    "best_post_time": "12PM EST Wednesday",
    "reel_cover_frame_hint": "Use the frame where speaker is most animated"
  }},
  "youtube_shorts": {{
    "caption": "...",
    "hashtags": ["#shorts", "#tag1"],
    "best_post_time": "3PM EST Saturday"
  }},
  "linkedin": {{
    "caption": "...",
    "best_post_time": "8AM EST Tuesday"
  }},
  "x_twitter": {{
    "caption": "...",
    "hashtags": ["#tag1"],
    "best_post_time": "9AM EST Wednesday"
  }},
  "hook_variants": [
    {{"text": "Nobody talks about this...", "style": "bold_white", "expected_ctr": 0.12}},
    {{"text": "Stop doing this RIGHT NOW", "style": "red_urgent", "expected_ctr": 0.10}},
    {{"text": "This changed everything for me", "style": "handwritten", "expected_ctr": 0.09}}
  ],
  "virality_breakdown": {{
    "hook_quality": 22,
    "engagement": 20,
    "shareability": 23,
    "value": 22,
    "visual_energy_bonus": 5
  }}
}}"""


async def generate_clip_distribution_meta(
    clip_data: Dict[str, Any],
    content_type: str = "entertainment",
    llm_provider: str = "anthropic",
) -> ClipDistributionMeta:
    """Generate distribution metadata for a single clip using AI."""
    try:
        prompt = DISTRIBUTION_PROMPT.format(
            title=clip_data.get("title", "Untitled"),
            duration=clip_data.get("duration", 30),
            transcript_excerpt=clip_data.get("transcript", "")[:500],
            virality_score=clip_data.get("virality_score", 0),
            content_type=content_type,
        )

        response_text = await _call_llm(prompt, llm_provider)
        return _parse_distribution_response(response_text, clip_data.get("id", ""))

    except Exception as e:
        logger.error(f"Distribution metadata generation failed: {e}")
        return ClipDistributionMeta(clip_id=clip_data.get("id", ""))


async def generate_batch_distribution_meta(
    clips: List[Dict[str, Any]],
    content_type: str = "entertainment",
) -> List[ClipDistributionMeta]:
    """Generate distribution metadata for multiple clips."""
    tasks = [generate_clip_distribution_meta(c, content_type) for c in clips]
    return await asyncio.gather(*tasks, return_exceptions=False)


async def _call_llm(prompt: str, provider: str) -> str:
    import httpx

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": prompt}],
                },
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                },
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    raise ValueError(f"Unknown provider: {provider}")


def _parse_distribution_response(text: str, clip_id: str) -> ClipDistributionMeta:
    meta = ClipDistributionMeta(clip_id=clip_id)
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return meta
        data = json.loads(text[start:end])

        for platform in ["tiktok", "instagram", "youtube_shorts", "linkedin", "x_twitter"]:
            if platform in data:
                p = data[platform]
                setattr(meta, platform, PlatformMeta(
                    caption=p.get("caption", ""),
                    hashtags=p.get("hashtags", []),
                    best_post_time=p.get("best_post_time", ""),
                    hook_score=p.get("hook_score", 0),
                ))

        for hv in data.get("hook_variants", []):
            meta.hook_variants.append(HookVariant(
                text=hv.get("text", ""),
                style=hv.get("style", "bold"),
                expected_ctr=hv.get("expected_ctr", 0.0),
            ))

        vb = data.get("virality_breakdown", {})
        meta.virality_breakdown = ViralityBreakdown(
            hook_quality=vb.get("hook_quality", 0),
            engagement=vb.get("engagement", 0),
            shareability=vb.get("shareability", 0),
            value=vb.get("value", 0),
            visual_energy_bonus=vb.get("visual_energy_bonus", 0),
            total=sum(vb.get(k, 0) for k in [
                "hook_quality", "engagement", "shareability", "value", "visual_energy_bonus"
            ]),
        )

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse distribution metadata: {e}")

    return meta
