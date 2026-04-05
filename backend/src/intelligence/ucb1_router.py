"""UCB1 Multi-Arm Bandit model router for AI analysis.

Selects the best LLM for transcript analysis based on historical performance
(virality scores of selected clips). Circuit breaker removes failing models.
"""
import math
import time
import logging
import json
from typing import Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = [
    "anthropic:claude-sonnet-4-6",
    "google-gla:gemini-2.0-flash",
    "openai:gpt-4o-mini",
]

UCB1_EXPLORATION = 1.5
CIRCUIT_BREAKER_ERRORS = 3
CIRCUIT_BREAKER_WINDOW = 300  # 5 min
CIRCUIT_BREAKER_COOLDOWN = 1800  # 30 min


@dataclass
class ModelStats:
    model: str
    total_plays: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    avg_latency_ms: float = 0.0
    errors: List[float] = field(default_factory=list)
    circuit_open_until: float = 0.0

    @property
    def is_circuit_open(self) -> bool:
        return time.time() < self.circuit_open_until


class UCB1ModelRouter:
    def __init__(self, redis_pool=None, models: list[str] | None = None):
        self._redis = redis_pool
        self._models = models or AVAILABLE_MODELS
        self._local_stats: Dict[str, ModelStats] = {
            m: ModelStats(model=m) for m in self._models
        }
        self._total_plays = 0

    async def _load_stats(self):
        if not self._redis:
            return
        try:
            for model in self._models:
                key = f"supoclip:ucb1:{model}"
                data = await self._redis.get(key)
                if data:
                    parsed = json.loads(data.decode() if isinstance(data, bytes) else data)
                    stats = self._local_stats[model]
                    stats.total_plays = parsed.get("plays", 0)
                    stats.total_reward = parsed.get("reward", 0.0)
                    stats.avg_reward = stats.total_reward / max(stats.total_plays, 1)
                    stats.avg_latency_ms = parsed.get("avg_latency", 0.0)
                    stats.errors = parsed.get("errors", [])
                    stats.circuit_open_until = parsed.get("circuit_until", 0.0)
            self._total_plays = sum(s.total_plays for s in self._local_stats.values())
        except Exception as e:
            logger.warning(f"UCB1 stats load failed: {e}")

    async def _save_stats(self, model: str):
        if not self._redis:
            return
        try:
            stats = self._local_stats[model]
            data = {
                "plays": stats.total_plays,
                "reward": stats.total_reward,
                "avg_latency": stats.avg_latency_ms,
                "errors": stats.errors[-20:],
                "circuit_until": stats.circuit_open_until,
            }
            await self._redis.set(f"supoclip:ucb1:{model}", json.dumps(data), ex=604800)
        except Exception as e:
            logger.warning(f"UCB1 stats save failed: {e}")

    async def select_model(self) -> str:
        await self._load_stats()

        available = [
            m for m in self._models
            if not self._local_stats[m].is_circuit_open
        ]

        if not available:
            logger.warning("All models circuit-broken, resetting cooldowns")
            for stats in self._local_stats.values():
                stats.circuit_open_until = 0
            available = self._models.copy()

        for model in available:
            if self._local_stats[model].total_plays == 0:
                logger.info(f"UCB1 exploring: {model} (first play)")
                return model

        best_model = available[0]
        best_score = -1.0
        total = max(self._total_plays, 1)

        for model in available:
            stats = self._local_stats[model]
            exploitation = stats.avg_reward
            exploration = UCB1_EXPLORATION * math.sqrt(math.log(total) / max(stats.total_plays, 1))
            ucb_score = exploitation + exploration
            if ucb_score > best_score:
                best_score = ucb_score
                best_model = model

        logger.info(f"UCB1 selected: {best_model} (score={best_score:.3f})")
        return best_model

    async def record_reward(self, model: str, reward: float, latency_ms: float = 0):
        if model not in self._local_stats:
            return

        stats = self._local_stats[model]
        stats.total_plays += 1
        stats.total_reward += reward
        stats.avg_reward = stats.total_reward / stats.total_plays
        if latency_ms > 0:
            stats.avg_latency_ms = (stats.avg_latency_ms * 0.8) + (latency_ms * 0.2)

        self._total_plays += 1
        await self._save_stats(model)
        logger.info(f"UCB1 reward: {model} +{reward:.2f} (avg={stats.avg_reward:.3f}, plays={stats.total_plays})")

    async def record_error(self, model: str):
        if model not in self._local_stats:
            return

        stats = self._local_stats[model]
        now = time.time()
        stats.errors.append(now)

        recent = [e for e in stats.errors if e > now - CIRCUIT_BREAKER_WINDOW]
        stats.errors = recent

        if len(recent) >= CIRCUIT_BREAKER_ERRORS:
            stats.circuit_open_until = now + CIRCUIT_BREAKER_COOLDOWN
            logger.warning(f"UCB1 circuit OPEN for {model}: {len(recent)} errors in {CIRCUIT_BREAKER_WINDOW}s")

        await self._save_stats(model)
