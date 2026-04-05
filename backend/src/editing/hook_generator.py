"""AI-generated text hook overlays for the first 3 seconds of clips.

Generates attention-grabbing text based on content type (IGZ-aware).
Returns 3 A/B variants per clip for testing.
"""
import logging
import json
import os
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

HOOK_STYLES = {
    "bold_white": {"font_color": "#FFFFFF", "bg_color": "#000000CC", "font_size": 42, "animation": "fade_in"},
    "red_urgent": {"font_color": "#FF0000", "bg_color": "#000000DD", "font_size": 48, "animation": "shake"},
    "handwritten": {"font_color": "#FFFFFF", "bg_color": None, "font_size": 36, "animation": "write_on"},
    "gradient": {"font_color": "#FFD700", "bg_color": "#000000AA", "font_size": 44, "animation": "slide_up"},
    "minimal": {"font_color": "#FFFFFF", "bg_color": None, "font_size": 32, "animation": "fade_in"},
}

INTENT_HOOK_TEMPLATES = {
    "educational": [
        "Did you know {topic}?",
        "Most people get this wrong about {topic}",
        "{topic} explained in {duration} seconds",
    ],
    "entertainment": [
        "Wait for it...",
        "Nobody expected this",
        "This is actually insane",
    ],
    "sales": [
        "Stop doing this right now",
        "This saved me ${amount}",
        "Why nobody is talking about this",
    ],
    "storytelling": [
        "This changed everything",
        "Here's what happened next...",
        "I never told anyone this before",
    ],
    "interview": [
        "Their answer shocked everyone",
        "The question nobody asks",
        "Listen to what they said about {topic}",
    ],
    "motivational": [
        "If I can do it, so can you",
        "They said it was impossible",
        "The {number} second rule that changed my life",
    ],
    "tutorial": [
        "Here's the trick nobody shows you",
        "Do THIS instead",
        "Stop wasting time doing it the wrong way",
    ],
    "news": [
        "Breaking: {topic}",
        "This just happened",
        "Everyone needs to see this",
    ],
}


@dataclass
class HookOverlay:
    text: str
    style: str
    style_config: Dict[str, Any]
    duration: float = 3.0
    position: str = "center"
    expected_ctr: float = 0.0


def generate_hooks(
    transcript_excerpt: str,
    intent: str = "entertainment",
    clip_title: str = "",
    num_variants: int = 3,
) -> List[HookOverlay]:
    """Generate hook text variants based on content type."""
    templates = INTENT_HOOK_TEMPLATES.get(intent, INTENT_HOOK_TEMPLATES["entertainment"])
    
    # Extract a topic keyword from transcript
    words = transcript_excerpt.split()[:20]
    topic = clip_title if clip_title else " ".join(words[:5]) if words else "this"
    
    hooks = []
    styles = list(HOOK_STYLES.keys())
    
    for i, template in enumerate(templates[:num_variants]):
        text = template.format(
            topic=topic[:40],
            duration="30",
            amount="1000",
            number="5",
        )
        style_name = styles[i % len(styles)]
        style_config = HOOK_STYLES[style_name]
        
        hooks.append(HookOverlay(
            text=text,
            style=style_name,
            style_config=style_config,
            expected_ctr=round(0.12 - (i * 0.02), 3),
        ))
    
    return hooks


async def generate_hooks_with_ai(
    transcript_excerpt: str,
    intent: str = "entertainment",
    clip_title: str = "",
) -> List[HookOverlay]:
    """Generate hooks using AI for more creative results. Falls back to templates."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return generate_hooks(transcript_excerpt, intent, clip_title)
    
    try:
        import httpx
        prompt = f"""Generate 3 attention-grabbing hook texts for a {intent} video clip.
Clip title: {clip_title}
Transcript excerpt: {transcript_excerpt[:300]}

Return JSON array:
[
  {{"text": "...", "style": "bold_white", "expected_ctr": 0.12}},
  {{"text": "...", "style": "red_urgent", "expected_ctr": 0.10}},
  {{"text": "...", "style": "handwritten", "expected_ctr": 0.09}}
]

Rules:
- Under 8 words each
- No clickbait that doesn't deliver
- Match the energy of the content type: {intent}"""

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 512,
                    "messages": [{"role": "user", "content": prompt}],
                },
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            resp.raise_for_status()
            text = resp.json()["content"][0]["text"]
            
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                hooks = []
                for item in data[:3]:
                    style_name = item.get("style", "bold_white")
                    hooks.append(HookOverlay(
                        text=item["text"],
                        style=style_name,
                        style_config=HOOK_STYLES.get(style_name, HOOK_STYLES["bold_white"]),
                        expected_ctr=item.get("expected_ctr", 0.08),
                    ))
                return hooks
    except Exception as e:
        logger.warning(f"AI hook generation failed, using templates: {e}")
    
    return generate_hooks(transcript_excerpt, intent, clip_title)
