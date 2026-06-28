from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from agent_memory.models import MemoryEntry, MemoryScope, MemoryState, MemoryType


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class MemoryStore(ABC):
    """Abstract base class for memory storage backends.
    This defines the common interface that all memory store implementations
    must follow, enabling interchangeability between backends (ChromaDB, SQLite, etc.)
    while maintaining SOLID principles - specifically the Liskov Substitution Principle
    and Dependency Inversion Principle.
    """

    @abstractmethod
    def store(self, entry: MemoryEntry) -> MemoryEntry:
        """Store a memory entry."""
        ...

    @abstractmethod
    def get(self, memory_id: str) -> MemoryEntry | None:
        """Retrieve a memory entry by ID."""
        ...

    @abstractmethod
    def update(self, entry: MemoryEntry) -> MemoryEntry:
        """Update an existing memory entry."""
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry by ID. Returns True if deleted."""
        ...

    @abstractmethod
    def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryEntry]:
        """List memory entries with optional filtering."""
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        """Search memories using semantic/vector similarity."""
        ...

    @abstractmethod
    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        """Search memories using keyword/BM25 matching."""
        ...

    @property
    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored memories."""
        ...


class ChromaDBStore(MemoryStore):
    """ChromaDB-backed persistent memory storage with scope filtering."""

    def __init__(
        self,
        persist_dir: str | Path = ".agent_memory",
        collection_name: str = "agent_memories",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def store(self, entry: MemoryEntry) -> MemoryEntry:
        entry.updated_at = datetime.now(timezone.utc)
        entry.refresh_state()
        self._collection.upsert(
            ids=[entry.id],
            documents=[self._search_document(entry)],
            metadatas=[self._entry_to_metadata(entry)],
        )
        return entry

    def get(self, memory_id: str) -> MemoryEntry | None:
        result = self._collection.get(ids=[memory_id], include=["metadatas", "documents"])
        if not result["ids"]:
            return None
        return self._metadata_to_entry(
            memory_id=result["ids"][0],
            document=result["documents"][0] or "",
            metadata=result["metadatas"][0] or {},
        )

    def update(self, entry: MemoryEntry) -> MemoryEntry:
        return self.store(entry)

    def delete(self, memory_id: str) -> bool:
        if not self.get(memory_id):
            return False
        self._collection.delete(ids=[memory_id])
        return True

    def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryEntry]:
        where = self._build_where(scopes=scopes, include_archived=include_archived, memory_type=memory_type)
        kwargs: dict[str, Any] = {
            "include": ["metadatas", "documents"],
            "limit": limit,
            "offset": offset,
        }
        if where:
            kwargs["where"] = where

        result = self._collection.get(**kwargs)
        entries: list[MemoryEntry] = []
        for memory_id, document, metadata in zip(
            result["ids"],
            result["documents"] or [],
            result["metadatas"] or [],
        ):
            entries.append(
                self._metadata_to_entry(
                    memory_id=memory_id,
                    document=document or "",
                    metadata=metadata or {},
                )
            )
        return self._filter_entries(entries, include_archived=include_archived, include_expired=include_expired)

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        if self.count == 0:
            return []

        where = self._build_where(scopes=scopes, include_archived=include_archived)
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(top_k, self.count),
            "include": ["metadatas", "documents", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)

        matches: list[tuple[MemoryEntry, float]] = []
        ids = result["ids"][0] if result["ids"] else []
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []

        for memory_id, doc, metadata, distance in zip(ids, documents, metadatas, distances):
            similarity = max(0.0, 1.0 - float(distance))
            entry = self._metadata_to_entry(
                memory_id=memory_id,
                document=doc or "",
                metadata=metadata or {},
            )
            matches.append((entry, similarity))

        return self._filter_matches(matches, include_archived=include_archived, include_expired=include_expired)

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        from rank_bm25 import BM25Okapi

        entries = self.list_all(
            limit=10_000,
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
        )
        if not entries:
            return []

        corpus = [_tokenize(self._search_document(e)) for e in entries]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(_tokenize(query))
        max_score = max(scores) if len(scores) else 1.0
        if max_score <= 0:
            return []

        ranked = sorted(
            zip(entries, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]

        return [(entry, float(score / max_score)) for entry, score in ranked if score > 0]

    @staticmethod
    def _search_document(entry: MemoryEntry) -> str:
        return f"{entry.query}\n{entry.content}\n{' '.join(entry.tags)}"

    def _build_where(
        self,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        memory_type: MemoryType | None = None,
    ) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []

        if not include_archived:
            clauses.append({"archived": False})

        if scopes:
            scope_values = [s.value for s in scopes]
            if len(scope_values) == 1:
                clauses.append({"scope": scope_values[0]})
            else:
                clauses.append({"scope": {"$in": scope_values}})

        if memory_type:
            clauses.append({"type": memory_type.value})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _entry_to_metadata(self, entry: MemoryEntry) -> dict[str, Any]:
        return {
            "response": entry.response,
            "content": entry.content,
            "type": entry.type.value,
            "scope": entry.scope.value,
            "metadata_json": json.dumps(entry.metadata),
            "tags": json.dumps(entry.tags),
            "confidence": entry.confidence,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "access_count": entry.access_count,
            "last_accessed_at": entry.last_accessed_at.isoformat() if entry.last_accessed_at else "",
            "archived": entry.archived,
            "requires_verification": entry.requires_verification,
            "expires_at": entry.expires_at.isoformat() if entry.expires_at else "",
            "state": entry.state.value,
        }

    def _metadata_to_entry(self, memory_id: str, document: str, metadata: dict[str, Any]) -> MemoryEntry:
        last_accessed = metadata.get("last_accessed_at") or None
        expires_raw = metadata.get("expires_at") or None
        query = document.split("\n", 1)[0] if document else ""
        entry = MemoryEntry(
            id=memory_id,
            query=query,
            response=metadata.get("response", ""),
            content=metadata.get("content", metadata.get("response", "")),
            type=MemoryType(metadata.get("type", MemoryType.CONVERSATION.value)),
            scope=MemoryScope(metadata.get("scope", MemoryScope.USER.value)),
            metadata=json.loads(metadata.get("metadata_json", "{}")),
            tags=json.loads(metadata.get("tags", "[]")),
            confidence=float(metadata.get("confidence", 1.0)),
            created_at=datetime.fromisoformat(metadata["created_at"])
            if metadata.get("created_at")
            else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(metadata["updated_at"])
            if metadata.get("updated_at")
            else datetime.now(timezone.utc),
            access_count=int(metadata.get("access_count", 0)),
            last_accessed_at=datetime.fromisoformat(last_accessed) if last_accessed else None,
            archived=bool(metadata.get("archived", False)),
            requires_verification=bool(metadata.get("requires_verification", False)),
            expires_at=datetime.fromisoformat(expires_raw) if expires_raw else None,
            state=MemoryState(metadata.get("state", MemoryState.ACTIVE.value)),
        )
        entry.refresh_state()
        return entry

    @staticmethod
    def _filter_entries(
        entries: list[MemoryEntry],
        *,
        include_archived: bool,
        include_expired: bool,
    ) -> list[MemoryEntry]:
        filtered: list[MemoryEntry] = []
        for entry in entries:
            if entry.archived and not include_archived:
                continue
            if entry.is_expired and not include_expired:
                continue
            filtered.append(entry)
        return filtered

    @staticmethod
    def _filter_matches(
        matches: list[tuple[MemoryEntry, float]],
        *,
        include_archived: bool,
        include_expired: bool,
    ) -> list[tuple[MemoryEntry, float]]:
        return [
            (entry, score)
            for entry, score in matches
            if (include_archived or not entry.archived) and (include_expired or not entry.is_expired)
        ]
