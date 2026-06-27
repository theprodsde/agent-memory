from pathlib import Path

import pytest

from agent_memory import Memory, MemoryAction, MemoryScope, MemoryType


@pytest.fixture
def memory(tmp_path: Path) -> Memory:
    return Memory(persist_dir=tmp_path / "mem")


def test_remember_and_get(memory: Memory) -> None:
    entry = memory.remember("What is Python?", "Python is a programming language.")
    fetched = memory.get(entry.id)
    assert fetched is not None
    assert fetched.query == "What is Python?"
    assert fetched.response == "Python is a programming language."


def test_replay_on_near_identical_query(memory: Memory) -> None:
    memory.remember("What is Python?", "Python is a programming language.")
    decision = memory.resolve("What is Python?", mode="replay")
    assert decision.action == MemoryAction.REPLAY
    assert decision.response == "Python is a programming language."


def test_restore_on_similar_query(memory: Memory) -> None:
    memory.remember(
        "Explain Python in one sentence",
        "Python is a high-level, readable programming language.",
    )
    decision = memory.resolve("Give me a one-sentence explanation of Python", mode="restore")
    assert decision.action in (MemoryAction.RESTORE, MemoryAction.REPLAY)
    assert decision.context


def test_none_on_unrelated_query(memory: Memory) -> None:
    memory.remember("What is Python?", "Python is a programming language.")
    decision = memory.resolve("What is the capital of France?", mode="auto")
    assert decision.action == MemoryAction.NONE


def test_forget(memory: Memory) -> None:
    entry = memory.remember("temp", "temp response")
    assert memory.forget(entry.id) is True
    assert memory.get(entry.id) is None


def test_persistence_across_instances(tmp_path: Path) -> None:
    persist = tmp_path / "persist"
    m1 = Memory(persist_dir=persist)
    entry = m1.remember("persist test", "still here")
    m2 = Memory(persist_dir=persist)
    fetched = m2.get(entry.id)
    assert fetched is not None
    assert fetched.response == "still here"


def test_format_restore_context(memory: Memory) -> None:
    memory.remember("q1", "r1", tags=["demo"])
    decision = memory.resolve("q1", mode="restore")
    context = memory.format_restore_context(decision)
    assert "Retrieved Memory Context" in context
    assert "r1" in context


def test_memory_types_and_scopes(memory: Memory) -> None:
    entry = memory.remember(
        "deploy steps",
        "run make deploy",
        type=MemoryType.WORKFLOW,
        scope=MemoryScope.PROJECT,
    )
    assert entry.type == MemoryType.WORKFLOW
    assert entry.scope == MemoryScope.PROJECT


def test_verify_on_fact_type(memory: Memory) -> None:
    memory.remember(
        "API rate limit",
        "1000 req/min",
        type=MemoryType.FACT,
    )
    decision = memory.resolve("What is the API rate limit?", mode="verify")
    assert decision.action in (MemoryAction.VERIFY, MemoryAction.REPLAY, MemoryAction.RESTORE)
    if decision.action == MemoryAction.VERIFY:
        assert decision.memory is not None


def test_archive_excludes_from_search(memory: Memory) -> None:
    entry = memory.remember("secret", "hidden answer")
    memory.archive(entry.id)
    decision = memory.resolve("secret", mode="replay")
    assert decision.action == MemoryAction.NONE


def test_list_with_scope_filter(memory: Memory) -> None:
    memory.remember("a", "1", scope=MemoryScope.USER)
    memory.remember("b", "2", scope=MemoryScope.PROJECT)
    user_only = memory.list(scope=[MemoryScope.USER])
    assert len(user_only) == 1
    assert user_only[0].scope == MemoryScope.USER


def test_memory_manager_alias(memory: Memory) -> None:
    from agent_memory import MemoryManager

    assert MemoryManager is Memory
