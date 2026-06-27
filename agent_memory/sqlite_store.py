from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_memory.models import MemoryEntry, MemoryScope, MemoryState, MemoryType


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class SqliteMemoryStore:
    """SQLite-backed persistent memory storage with scope filtering and BM25 keyword search."""

    def __init__(
        self,
        persist_dir: str | Path = ".agent_memory",
        collection_name: str = "agent_memories",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.persist_dir / f"{collection_name}.db"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    tags TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    requires_verification INTEGER NOT NULL DEFAULT 0,
                    archived INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL DEFAULT 'active',
                    access_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_state ON memories(state)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_expires_at ON memories(expires_at)
            """)
            conn.commit()

    @property
    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            return cursor.fetchone()[0]

    def store(self, entry: MemoryEntry) -> MemoryEntry:
        entry.updated_at = datetime.now(timezone.utc)
        entry.refresh_state()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories (
                    id, query, response, content, type, scope, metadata, tags,
                    confidence, requires_verification, archived, state, access_count,
                    created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.query,
                    entry.response,
                    entry.content,
                    entry.type.value,
                    entry.scope.value,
                    json.dumps(entry.metadata),
                    json.dumps(entry.tags),
                    entry.confidence,
                    int(entry.requires_verification),
                    int(entry.archived),
                    entry.state.value,
                    entry.access_count,
                    entry.created_at.isoformat(),
                    entry.updated_at.isoformat(),
                    entry.expires_at.isoformat() if entry.expires_at else None,
                ),
            )
            conn.commit()
        return entry

    def get(self, memory_id: str) -> MemoryEntry | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_entry(row)

    def update(self, entry: MemoryEntry) -> MemoryEntry:
        return self.store(entry)

    def delete(self, memory_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0

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
        where_clauses = []
        params: list[Any] = []

        if not include_archived:
            where_clauses.append("archived = 0")

        if scopes:
            scope_values = [s.value for s in scopes]
            placeholders = ",".join("?" * len(scope_values))
            where_clauses.append(f"scope IN ({placeholders})")
            params.extend(scope_values)

        if memory_type:
            where_clauses.append("type = ?")
            params.append(memory_type.value)

        if not include_expired:
            where_clauses.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(datetime.now(timezone.utc).isoformat())

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        params.extend([limit, offset])

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                f"SELECT * FROM memories {where_sql} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                params,
            )
            return [self._row_to_entry(row) for row in cursor.fetchall()]

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        # For SQLite, we'll do a simple LIKE-based search with BM25 ranking
        # In a production system, you'd want to use SQLite's FTS5 extension
        entries = self.list_all(
            limit=10_000,
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
        )
        if not entries:
            return []

        # First try BM25 keyword matching
        from rank_bm25 import BM25Okapi

        corpus = [_tokenize(self._search_document(e)) for e in entries]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(_tokenize(query))
        max_score = max(scores) if len(scores) else 1.0
        
        # If BM25 finds matches, use those
        if max_score > 0:
            ranked = sorted(
                zip(entries, scores),
                key=lambda pair: pair[1],
                reverse=True,
            )[:top_k]
            return [(entry, float(score / max_score)) for entry, score in ranked if score > 0]
        
        # Fallback: simple token overlap search for partial matches
        # Filter out common stop words
        stop_words = {'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'}
        
        query_tokens = set(_tokenize(query)) - stop_words
        if not query_tokens:
            return []
        
        matches = []
        for entry in entries:
            doc_text = self._search_document(entry).lower()
            doc_tokens = set(_tokenize(doc_text)) - stop_words
            
            # Check for significant token overlap (at least 1 meaningful token)
            overlap = len(query_tokens & doc_tokens)
            if overlap > 0:
                # Score based on token overlap ratio
                score = overlap / max(len(query_tokens), 1)
                matches.append((entry, score))
        
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            max_match_score = matches[0][1]
            return [(entry, float(score / max_match_score)) for entry, score in matches[:top_k]]
        
        return []

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        # Same as search for SQLite backend
        return self.search(
            query,
            top_k=top_k,
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
        )

    @staticmethod
    def _search_document(entry: MemoryEntry) -> str:
        return f"{entry.query}\n{entry.content}\n{' '.join(entry.tags)}"

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            query=row["query"],
            response=row["response"],
            content=row["content"],
            type=MemoryType(row["type"]),
            scope=MemoryScope(row["scope"]),
            metadata=json.loads(row["metadata"]),
            tags=json.loads(row["tags"]),
            confidence=row["confidence"],
            requires_verification=bool(row["requires_verification"]),
            archived=bool(row["archived"]),
            state=MemoryState(row["state"]),
            access_count=row["access_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
        )

    def _filter_entries(
        self,
        entries: list[MemoryEntry],
        *,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[MemoryEntry]:
        filtered = []
        now = datetime.now(timezone.utc)
        for entry in entries:
            entry.refresh_state()
            if not include_archived and entry.archived:
                continue
            if not include_expired and entry.is_expired:
                continue
            filtered.append(entry)
        return filtered

    def _filter_matches(
        self,
        matches: list[tuple[MemoryEntry, float]],
        *,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        filtered = []
        for entry, score in matches:
            entry.refresh_state()
            if not include_archived and entry.archived:
                continue
            if not include_expired and entry.is_expired:
                continue
            filtered.append((entry, score))
        return filtered