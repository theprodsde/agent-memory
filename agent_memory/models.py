from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

# Memory types that default to verify-before-reuse when similarity is moderate.
VERIFY_TYPES: frozenset[str] = frozenset({"fact", "tool_output", "workflow"})


class MemoryAction(str, Enum):
    """How the agent should use a retrieved memory."""

    REPLAY = "replay"
    RESTORE = "restore"
    VERIFY = "verify"
    NONE = "none"


class MemoryType(str, Enum):
    CONVERSATION = "conversation"
    FACT = "fact"
    WORKFLOW = "workflow"
    TOOL_OUTPUT = "tool_output"
    DOCUMENT = "document"
    SUMMARY = "summary"
    CODE = "code"
    PREFERENCE = "preference"


class MemoryScope(str, Enum):
    SESSION = "session"
    USER = "user"
    PROJECT = "project"
    WORKSPACE = "workspace"
    TEAM = "team"
    GLOBAL = "global"


class MemoryState(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    EXPIRED = "expired"
    DELETED = "deleted"


class MemoryEntry(BaseModel):
    """A stored memory with structured metadata."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    response: str
    content: str = ""
    type: MemoryType = MemoryType.CONVERSATION
    scope: MemoryScope = MemoryScope.USER
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    last_accessed_at: datetime | None = None
    archived: bool = False
    requires_verification: bool = False
    expires_at: datetime | None = None
    state: MemoryState = MemoryState.ACTIVE

    @property
    def is_expired(self) -> bool:
        if self.state == MemoryState.EXPIRED:
            return True
        if self.expires_at is None:
            return False
        now = datetime.now(timezone.utc)
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return now >= expires

    def refresh_state(self) -> None:
        if self.archived:
            self.state = MemoryState.ARCHIVED
        elif self.is_expired:
            self.state = MemoryState.EXPIRED
        else:
            self.state = MemoryState.ACTIVE

    @model_validator(mode="after")
    def default_content(self) -> MemoryEntry:
        if not self.content:
            self.content = self.response
        return self

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "query": self.query,
            "response": self.response,
            "type": self.type.value,
            "scope": self.scope.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            "access_count": self.access_count,
            "confidence": self.confidence,
            "tags": self.tags,
            "archived": self.archived,
            "state": self.state.value,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


class RetrievalResult(BaseModel):
    """A memory match from hybrid retrieval."""

    entry: MemoryEntry
    semantic_score: float
    keyword_score: float = 0.0
    final_score: float = 0.0
    rank: int = 0

    @property
    def similarity(self) -> float:
        """Backward-compatible alias for decision score."""
        return self.decision_score

    @property
    def decision_score(self) -> float:
        """Score used for action thresholds — never below raw retrieval signals."""
        retrieval = max(self.semantic_score, 0.7 * self.semantic_score + 0.3 * self.keyword_score)
        return max(self.final_score, retrieval)


class MemoryDecision(BaseModel):
    """Agent-facing decision on how to answer a query."""

    action: MemoryAction
    query: str
    confidence: float
    response: str | None = None
    memory: MemoryEntry | None = None
    context: list[RetrievalResult] = Field(default_factory=list)
    reason: str = ""
    matched: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)

    def explain(self) -> str:
        from agent_memory.explain import format_explanation

        return format_explanation(self)

    def __repr__(self) -> str:
        matched = ", ".join(self.matched[:3])
        if len(self.matched) > 3:
            matched += ", ..."
        return (
            f"Decision(action={self.action.value!r}, confidence={self.confidence:.2f}, "
            f"matched=[{matched}], reasons={self.reasons!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()
