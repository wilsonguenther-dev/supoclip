"""H2E-inspired content intelligence for SupoClip clip selection.

Maps video content types to virality scoring weights so the AI analysis
prompt emphasizes the right signals per content type.
"""
import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

INTENT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "educational": {"hook_score": 0.20, "engagement_score": 0.20, "value_score": 0.40, "shareability_score": 0.20},
    "entertainment": {"hook_score": 0.40, "engagement_score": 0.40, "value_score": 0.10, "shareability_score": 0.10},
    "sales": {"hook_score": 0.40, "engagement_score": 0.10, "value_score": 0.10, "shareability_score": 0.40},
    "storytelling": {"hook_score": 0.30, "engagement_score": 0.50, "value_score": 0.10, "shareability_score": 0.10},
    "interview": {"hook_score": 0.15, "engagement_score": 0.40, "value_score": 0.30, "shareability_score": 0.15},
    "tutorial": {"hook_score": 0.20, "engagement_score": 0.15, "value_score": 0.50, "shareability_score": 0.15},
    "motivational": {"hook_score": 0.35, "engagement_score": 0.30, "value_score": 0.10, "shareability_score": 0.25},
    "news": {"hook_score": 0.30, "engagement_score": 0.20, "value_score": 0.25, "shareability_score": 0.25},
}

INTENT_SIGNALS: Dict[str, list[str]] = {
    "educational": ["learn", "understand", "how to", "step by step", "lesson", "explain", "teach", "know"],
    "entertainment": ["funny", "crazy", "watch this", "you won't believe", "hilarious", "prank", "challenge"],
    "sales": ["buy", "offer", "limited", "discount", "deal", "price", "subscribe", "sign up", "call now"],
    "storytelling": ["story", "journey", "happened", "experience", "remember", "once upon", "back when", "grew up"],
    "interview": ["question", "tell us", "your thoughts", "perspective", "opinion", "guest", "host"],
    "tutorial": ["tutorial", "guide", "follow along", "step one", "click here", "open", "install", "setup"],
    "motivational": ["dream", "believe", "possible", "grind", "hustle", "success", "never give up", "mindset"],
    "news": ["breaking", "report", "update", "according to", "sources", "announced", "latest"],
}

ENGAGEMENT_DENSITY_THRESHOLDS = {
    "boredom": 0.3,
    "flow": 0.6,
    "stretch": 0.8,
}


def classify_video_intent(transcript: str) -> Dict[str, Any]:
    lower = transcript.lower()
    scores: Dict[str, int] = {}
    for intent, signals in INTENT_SIGNALS.items():
        score = sum(1 for sig in signals if sig in lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        intent = "entertainment"
        confidence = 0.3
    else:
        intent = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = round(scores[intent] / max(total, 1), 2)

    word_count = len(transcript.split())
    sentences = len(re.findall(r'[.!?]+', transcript))
    words_per_sentence = word_count / max(sentences, 1)
    engagement_density = min(1.0, sentences / max(word_count / 50, 1))

    if engagement_density < ENGAGEMENT_DENSITY_THRESHOLDS["boredom"]:
        flow_state = "boredom"
    elif engagement_density < ENGAGEMENT_DENSITY_THRESHOLDS["flow"]:
        flow_state = "flow"
    elif engagement_density < ENGAGEMENT_DENSITY_THRESHOLDS["stretch"]:
        flow_state = "stretch"
    else:
        flow_state = "anxiety"

    logger.info(f"H2E classification: intent={intent} ({confidence}), flow={flow_state}, density={engagement_density:.2f}")

    return {
        "intent": intent,
        "confidence": confidence,
        "flow_state": flow_state,
        "engagement_density": round(engagement_density, 3),
        "words_per_sentence": round(words_per_sentence, 1),
        "all_scores": scores,
    }


def get_scoring_weights(intent: str) -> Dict[str, float]:
    return INTENT_WEIGHTS.get(intent, INTENT_WEIGHTS["entertainment"])


def get_clip_count_adjustment(flow_state: str, base_count: int) -> int:
    adjustments = {
        "boredom": min(base_count + 3, 12),
        "anxiety": max(base_count - 2, 3),
        "flow": base_count,
        "stretch": base_count,
    }
    return adjustments.get(flow_state, base_count)


def get_clip_padding(flow_state: str) -> float:
    if flow_state == "stretch":
        return 5.0
    if flow_state == "anxiety":
        return 3.0
    return 0.0


def adjust_prompt_for_intent(base_prompt: str, weights: Dict[str, float], intent_data: Dict[str, Any]) -> str:
    intent = intent_data.get("intent", "entertainment")
    flow = intent_data.get("flow_state", "flow")

    weight_instructions = []
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for metric, weight in sorted_weights:
        pct = int(weight * 100)
        name = metric.replace("_score", "").replace("_", " ").title()
        weight_instructions.append(f"- {name}: {pct}% weight")

    addendum = f"""
CONTENT INTELLIGENCE (H2E Analysis):
Content type: {intent} (confidence: {intent_data.get('confidence', 0)})
Engagement flow: {flow}

SCORING WEIGHTS (adjust your virality scoring accordingly):
{chr(10).join(weight_instructions)}

"""
    if flow == "boredom":
        addendum += "NOTE: Content has low engagement density. Favor shorter, punchier clips with strong hooks.\n"
    elif flow == "anxiety":
        addendum += "NOTE: Content is very dense. Favor longer clips that give context. Don't cut mid-explanation.\n"
    elif flow == "stretch":
        addendum += "NOTE: Complex topic. Add extra padding to clip boundaries so viewers get full context.\n"

    return base_prompt + "\n" + addendum
