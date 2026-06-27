from __future__ import annotations

from agent_memory.models import MemoryEntry, MemoryScope, RetrievalResult
from agent_memory.policy import DecisionPolicy, DefaultPolicy
from agent_memory.store import MemoryStore


class MemoryRetriever:
    """Hybrid retrieval: BM25 keyword search + vector search + policy rerank."""

    def __init__(self, store: MemoryStore, policy: DecisionPolicy | None = None) -> None:
        self._store = store
        self._policy = policy or DefaultPolicy()

    @property
    def policy(self) -> DecisionPolicy:
        return self._policy

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
    ) -> list[RetrievalResult]:
        vector_hits = self._store.search(query, top_k=top_k * 2, scopes=scopes)
        keyword_hits = self._store.keyword_search(query, top_k=top_k * 2, scopes=scopes)

        fused = self._reciprocal_rank_fusion(vector_hits, keyword_hits, k=60)

        results: list[RetrievalResult] = []
        for rank, (entry, semantic, keyword) in enumerate(fused[:top_k], start=1):
            final_score = self._policy.score(entry, semantic, keyword)
            results.append(
                RetrievalResult(
                    entry=entry,
                    semantic_score=semantic,
                    keyword_score=keyword,
                    final_score=final_score,
                    rank=rank,
                )
            )

        results.sort(key=lambda r: r.final_score, reverse=True)
        for index, result in enumerate(results, start=1):
            result.rank = index
        return results

    def retrieve_best(
        self,
        query: str,
        *,
        scopes: list[MemoryScope] | None = None,
    ) -> RetrievalResult | None:
        results = self.retrieve(query, top_k=1, scopes=scopes)
        return results[0] if results else None

    def record_access(self, entry: MemoryEntry) -> MemoryEntry:
        entry.touch()
        return self._store.update(entry)

    @staticmethod
    def _reciprocal_rank_fusion(
        vector_hits: list[tuple[MemoryEntry, float]],
        keyword_hits: list[tuple[MemoryEntry, float]],
        *,
        k: int = 60,
    ) -> list[tuple[MemoryEntry, float, float]]:
        scores: dict[str, dict[str, float | MemoryEntry]] = {}

        for rank, (entry, score) in enumerate(vector_hits, start=1):
            bucket = scores.setdefault(entry.id, {"entry": entry, "semantic": 0.0, "keyword": 0.0})
            bucket["semantic"] = max(float(bucket["semantic"]), score)
            bucket["rrf"] = float(bucket.get("rrf", 0.0)) + 1.0 / (k + rank)

        for rank, (entry, score) in enumerate(keyword_hits, start=1):
            bucket = scores.setdefault(entry.id, {"entry": entry, "semantic": 0.0, "keyword": 0.0})
            bucket["keyword"] = max(float(bucket["keyword"]), score)
            bucket["rrf"] = float(bucket.get("rrf", 0.0)) + 1.0 / (k + rank)

        fused = [
            (
                bucket["entry"],  # type: ignore[arg-type]
                float(bucket.get("semantic", 0.0)),
                float(bucket.get("keyword", 0.0)),
                float(bucket.get("rrf", 0.0)),
            )
            for bucket in scores.values()
        ]
        fused.sort(key=lambda item: item[3], reverse=True)
        return [(entry, semantic, keyword) for entry, semantic, keyword, _ in fused]
