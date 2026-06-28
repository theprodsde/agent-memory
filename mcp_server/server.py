"""MCP server exposing agent memory tools to Cursor and other MCP clients."""

from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from agent_memory import Memory, MemoryAction

PERSIST_DIR = os.environ.get("AGENT_MEMORY_DIR", str(Path.home() / ".agent_memory"))
COLLECTION = os.environ.get("AGENT_MEMORY_COLLECTION", "agent_memories")

mcp = FastMCP("agent-memory")
_memory: Memory | None = None


def get_memory() -> Memory:
    global _memory
    if _memory is None:
        _memory = Memory(persist_dir=PERSIST_DIR, collection_name=COLLECTION)
    return _memory


@mcp.tool()
def remember_memory(
    query: str,
    response: str,
    tags: list[str] | None = None,
    metadata_json: str = "{}",
    memory_type: str = "conversation",
    scope: str = "user",
    confidence: float = 1.0,
    requires_verification: bool = False,
) -> str:
    """Store a query/response pair in persistent agent memory."""
    memory = get_memory()
    metadata = json.loads(metadata_json) if metadata_json else {}
    entry = memory.remember(
        query,
        response,
        metadata=metadata,
        tags=tags or [],
        type=memory_type,
        scope=scope,
        confidence=confidence,
        requires_verification=requires_verification,
    )
    return json.dumps(entry.to_dict(), indent=2)


@mcp.tool()
def resolve_memory(
    query: str,
    mode: str = "auto",
    top_k: int = 3,
    scope: list[str] | None = None,
    enable_verify: bool = True,
) -> str:
    """
    Retrieve memory and decide how the agent should respond.

    Actions: replay, restore, verify, none
    Modes: auto (default), replay, restore, verify
    """
    memory = get_memory()
    decision = memory.resolve(
        query,
        mode=mode,
        top_k=top_k,
        scope=scope,
        enable_verify=enable_verify,
    )

    payload: dict = {
        "action": decision.action.value,
        "confidence": round(decision.confidence, 4),
        "reason": decision.reason,
        "query": decision.query,
    }

    if decision.action == MemoryAction.REPLAY:
        payload["response"] = decision.response
        payload["memory_id"] = decision.memory.id if decision.memory else None
        payload["instruction"] = "Return this exact response to the user."
    elif decision.action == MemoryAction.RESTORE:
        payload["context"] = [
            {
                "memory_id": r.entry.id,
                "score": round(r.final_score, 4),
                "type": r.entry.type.value,
                "prior_query": r.entry.query,
                "prior_response": r.entry.response,
                "tags": r.entry.tags,
            }
            for r in decision.context
        ]
        payload["prompt_context"] = memory.format_restore_context(decision)
        payload["instruction"] = "Use retrieved memory as context. Adapt the prior answer."
    elif decision.action == MemoryAction.VERIFY:
        payload["memory_id"] = decision.memory.id if decision.memory else None
        payload["memory"] = decision.memory.to_dict() if decision.memory else None
        payload["prompt_context"] = memory.format_verify_context(decision)
        payload["instruction"] = "Validate memory freshness before reuse or regeneration."
    else:
        payload["instruction"] = "No relevant memory found. Answer from scratch."
        if decision.context:
            payload["near_misses"] = [
                {"score": round(r.final_score, 4), "prior_query": r.entry.query}
                for r in decision.context
            ]

    return json.dumps(payload, indent=2)


@mcp.tool()
def list_memories(
    limit: int = 20,
    offset: int = 0,
    scope: list[str] | None = None,
    include_archived: bool = False,
) -> str:
    """List stored memories with pagination and optional scope filter."""
    memory = get_memory()
    entries = memory.list(limit=limit, offset=offset, scope=scope, include_archived=include_archived)
    return json.dumps(
        {
            "total_in_page": len(entries),
            "memories": [e.to_dict() | {"response_preview": e.response[:200]} for e in entries],
        },
        indent=2,
    )


@mcp.tool()
def get_memory_by_id(memory_id: str) -> str:
    """Fetch a single memory by ID."""
    memory = get_memory()
    entry = memory.get(memory_id)
    if not entry:
        return json.dumps({"error": f"Memory {memory_id} not found."})
    return json.dumps(entry.to_dict() | {"response": entry.response, "metadata": entry.metadata})


@mcp.tool()
def forget_memory(memory_id: str) -> str:
    """Delete a memory by ID."""
    memory = get_memory()
    deleted = memory.forget(memory_id)
    return json.dumps({"deleted": deleted, "memory_id": memory_id}, indent=2)


@mcp.tool()
def archive_memory(memory_id: str) -> str:
    """Archive a memory so it is excluded from retrieval."""
    memory = get_memory()
    entry = memory.archive(memory_id)
    if not entry:
        return json.dumps({"error": f"Memory {memory_id} not found."})
    return json.dumps({"archived": True, "memory": entry.to_dict()}, indent=2)


@mcp.tool()
def consolidate_memories() -> str:
    """Merge near-duplicate memories into summary entries."""
    memory = get_memory()
    created = memory.consolidate()
    return json.dumps(
        {
            "consolidated_count": len(created),
            "summaries": [e.to_dict() for e in created],
        },
        indent=2,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
