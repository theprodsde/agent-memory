from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from agent_memory.models import (
    VERIFY_TYPES,
    MemoryAction,
    MemoryEntry,
    RetrievalResult,
)


class DecisionPolicy(ABC):
    """Scoring and action-selection policy for memory resolution."""

    @abstractmethod
    def score(self, entry: MemoryEntry, semantic: float, keyword: float) -> float:
        ...

    @abstractmethod
    def select_action(
        self,
        results: list[RetrievalResult],
        *,
        replay_threshold: float,
        restore_threshold: float,
        verify_threshold: float,
    ) -> tuple[MemoryAction, float, str]:
        ...


class DefaultPolicy(DecisionPolicy):
    """
    Default policy combining:
      semantic score + recency + confidence + usage
    """

    def __init__(
        self,
        semantic_weight: float = 0.55,
        recency_weight: float = 0.15,
        confidence_weight: float = 0.20,
        usage_weight: float = 0.10,
        recency_half_life_days: float = 30.0,
        usage_cap: int = 20,
    ) -> None:
        self.semantic_weight = semantic_weight
        self.recency_weight = recency_weight
        self.confidence_weight = confidence_weight
        self.usage_weight = usage_weight
        self.recency_half_life_days = recency_half_life_days
        self.usage_cap = usage_cap

    def score(self, entry: MemoryEntry, semantic: float, keyword: float) -> float:
        return self.score_breakdown(entry, semantic, keyword)["policy_score"]

    def score_breakdown(self, entry: MemoryEntry, semantic: float, keyword: float) -> dict[str, float]:
        hybrid_semantic = 0.7 * semantic + 0.3 * keyword
        recency = self._recency_score(entry.updated_at)
        usage = min(entry.access_count, self.usage_cap) / self.usage_cap
        confidence = entry.confidence
        policy_score = (
            self.semantic_weight * hybrid_semantic
            + self.recency_weight * recency
            + self.confidence_weight * confidence
            + self.usage_weight * usage
        )
        decision_score = max(policy_score, max(semantic, 0.7 * semantic + 0.3 * keyword))
        return {
            "semantic_score": hybrid_semantic,
            "keyword_score": keyword,
            "recency_score": recency,
            "confidence_score": confidence,
            "usage_score": usage,
            "policy_score": policy_score,
            "final_score": decision_score,
        }

    def select_action(
        self,
        results: list[RetrievalResult],
        *,
        replay_threshold: float,
        restore_threshold: float,
        verify_threshold: float,
    ) -> tuple[MemoryAction, float, str]:
        best = results[0]
        score = best.decision_score
        entry = best.entry

        if score >= replay_threshold:
            return (
                MemoryAction.REPLAY,
                score,
                "High composite score — replaying stored response.",
            )

        if score >= restore_threshold:
            if self._should_verify(entry, score, verify_threshold):
                return (
                    MemoryAction.VERIFY,
                    score,
                    "Moderate score on freshness-sensitive memory — verify before reuse.",
                )
            return (
                MemoryAction.RESTORE,
                score,
                "Moderate score — restoring memory as context for synthesis.",
            )

        return (
            MemoryAction.NONE,
            score,
            "Low composite score — no memory applied.",
        )

    def _should_verify(self, entry: MemoryEntry, score: float, verify_threshold: float) -> bool:
        if entry.requires_verification:
            return True
        if entry.type.value in VERIFY_TYPES and score < verify_threshold:
            return True
        age_days = (datetime.now(timezone.utc) - entry.updated_at).total_seconds() / 86400
        if entry.type.value in VERIFY_TYPES and age_days > self.recency_half_life_days:
            return True
        return False

    def _recency_score(self, updated_at: datetime) -> float:
        age_days = max(0.0, (datetime.now(timezone.utc) - updated_at).total_seconds() / 86400)
        return math.exp(-0.693 * age_days / self.recency_half_life_days)
