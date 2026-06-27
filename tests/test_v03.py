"""Tests for observability, TTL, benchmark, and eval."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_memory import Memory, MemoryAction, MemoryState
from agent_memory.eval import EvalCase, EvalDataset, run_eval, seed_dataset
from agent_memory.ttl import parse_ttl


@pytest.fixture
def memory(tmp_path: Path) -> Memory:
    return Memory(persist_dir=tmp_path / "mem")


def test_decision_repr_and_explain(memory: Memory) -> None:
    memory.remember("What is Python?", "Python is a programming language.")
    decision = memory.resolve("What is Python?")
    text = repr(decision)
    assert "action=" in text
    assert "confidence=" in text
    assert decision.matched
    assert decision.reasons
    explanation = decision.explain()
    assert "semantic_score:" in explanation
    assert "final_score:" in explanation


def test_parse_ttl() -> None:
    expires = parse_ttl("30d")
    assert expires is not None
    assert expires > datetime.now(timezone.utc)


def test_ttl_expires_memory(memory: Memory) -> None:
    entry = memory.remember("temp", "gone soon", ttl=1)
    entry.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    entry.refresh_state()
    memory.store.update(entry)
    decision = memory.resolve("temp")
    assert decision.action == MemoryAction.NONE


def test_cleanup_marks_expired(memory: Memory) -> None:
    entry = memory.remember("old", "stale", ttl=1)
    entry.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    entry.refresh_state()
    memory.store.update(entry)
    result = memory.cleanup()
    assert result["expired"] == 1
    fetched = memory.get(entry.id)
    assert fetched is not None
    assert fetched.state == MemoryState.EXPIRED


def test_stats(memory: Memory) -> None:
    memory.remember("a", "1")
    memory.remember("b", "2", type="fact")
    stats = memory.stats()
    assert stats["total"] == 2
    assert stats["by_type"]["conversation"] == 1
    assert stats["by_type"]["fact"] == 1


def test_eval_dataset(tmp_path: Path, memory: Memory) -> None:
    dataset = EvalDataset(
        name="test",
        memories=[{"query": "hello", "response": "world"}],
        cases=[EvalCase(query="hello", expected_action="replay")],
    )
    seed_dataset(memory, dataset)
    result = run_eval(memory, dataset)
    assert result.total == 1
    assert result.correct == 1


def test_eval_loads_json_file() -> None:
    path = Path(__file__).resolve().parent.parent / "benchmarks" / "datasets" / "customer_support.json"
    dataset = EvalDataset.load(path)
    assert dataset.name == "customer_support"
    assert len(dataset.cases) >= 3
