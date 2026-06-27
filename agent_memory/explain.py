from __future__ import annotations

from agent_memory.models import MemoryAction, MemoryDecision, RetrievalResult
from agent_memory.policy import DefaultPolicy, DecisionPolicy


def build_score_breakdown(
    result: RetrievalResult,
    policy: DecisionPolicy,
) -> dict[str, float]:
    if isinstance(policy, DefaultPolicy):
        return policy.score_breakdown(result.entry, result.semantic_score, result.keyword_score)

    return {
        "semantic_score": result.semantic_score,
        "keyword_score": result.keyword_score,
        "recency_score": 0.0,
        "confidence_score": result.entry.confidence,
        "usage_score": 0.0,
        "policy_score": result.final_score,
        "final_score": result.decision_score,
    }


def build_reasons(result: RetrievalResult, policy: DecisionPolicy) -> list[str]:
    reasons: list[str] = []
    breakdown = build_score_breakdown(result, policy)

    if result.semantic_score >= 0.85:
        reasons.append("high semantic match")
    elif result.semantic_score >= 0.70:
        reasons.append("moderate semantic match")

    if result.keyword_score >= 0.50:
        reasons.append("keyword match")

    if breakdown.get("recency_score", 0) >= 0.80:
        reasons.append("recent memory")
    elif result.entry.access_count > 0:
        reasons.append("prior usage")

    if result.entry.access_count >= 3:
        reasons.append("frequently used")

    if result.entry.confidence >= 0.90:
        reasons.append("high confidence memory")

    if result.entry.requires_verification:
        reasons.append("requires verification")

    if result.entry.type.value in {"fact", "tool_output", "workflow"}:
        reasons.append("freshness-sensitive type")

    return reasons


def enrich_decision(
    decision: MemoryDecision,
    policy: DecisionPolicy,
    *,
    replay_threshold: float,
    restore_threshold: float,
) -> MemoryDecision:
    """Attach matched IDs, reason tags, and score breakdown to a decision."""
    if not decision.context:
        decision.matched = []
        decision.reasons = ["no matching memories"]
        return decision

    best = decision.context[0]
    decision.matched = [r.entry.id for r in decision.context]
    decision.reasons = build_reasons(best, policy)
    decision.scores = build_score_breakdown(best, policy)

    if decision.action == MemoryAction.REPLAY and best.decision_score >= replay_threshold:
        if "high semantic match" not in decision.reasons and best.semantic_score >= 0.85:
            decision.reasons.insert(0, "exact or near-exact query match")
    elif decision.action == MemoryAction.NONE:
        if best.decision_score < restore_threshold:
            decision.reasons.append("below restore threshold")

    return decision


def format_explanation(decision: MemoryDecision) -> str:
    if not decision.scores:
        return "No score breakdown available (no memory matches)."

    lines = [
        f"action: {decision.action.value}",
        f"confidence: {decision.confidence:.2f}",
        f"matched: {', '.join(decision.matched) or 'none'}",
        "",
    ]

    if decision.reasons:
        lines.append("reasons:")
        for reason in decision.reasons:
            lines.append(f"  - {reason}")
        lines.append("")

    lines.append("scores:")
    label_map = {
        "semantic_score": "semantic_score",
        "keyword_score": "keyword_score",
        "recency_score": "recency_score",
        "confidence_score": "confidence_score",
        "usage_score": "usage_score",
        "policy_score": "policy_score",
        "final_score": "final_score",
    }
    for key, label in label_map.items():
        if key in decision.scores:
            lines.append(f"  {label}: {decision.scores[key]:.2f}")

    return "\n".join(lines)
