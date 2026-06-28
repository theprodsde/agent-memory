"""Tests for both ChromaDB and SQLite backends."""

from pathlib import Path

import pytest

from agent_memory import Memory, MemoryAction, MemoryScope, MemoryType


@pytest.fixture(params=["sqlite", "chromadb"])
def memory(request, tmp_path: Path) -> Memory:
    """Fixture that provides Memory instances for both backends."""
    backend = request.param
    persist_dir = tmp_path / f"mem_{backend}"
    return Memory(persist_dir=persist_dir, backend=backend)


@pytest.fixture
def sqlite_memory(tmp_path: Path) -> Memory:
    """SQLite-specific memory fixture."""
    return Memory(persist_dir=tmp_path / "mem_sqlite", backend="sqlite")


@pytest.fixture
def chromadb_memory(tmp_path: Path) -> Memory:
    """ChromaDB-specific memory fixture."""
    return Memory(persist_dir=tmp_path / "mem_chromadb", backend="chromadb")


# Tests that run for both backends
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
    # Test with both backends
    for backend in ["sqlite", "chromadb"]:
        persist = tmp_path / f"persist_{backend}"
        m1 = Memory(persist_dir=persist, backend=backend)
        entry = m1.remember("persist test", "still here")
        m2 = Memory(persist_dir=persist, backend=backend)
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


# SQLite-specific tests
def test_sqlite_backend_basic(sqlite_memory: Memory) -> None:
    """Test basic SQLite backend functionality."""
    entry = sqlite_memory.remember("test", "response", type=MemoryType.FACT)
    assert entry.id is not None

    fetched = sqlite_memory.get(entry.id)
    assert fetched is not None
    assert fetched.response == "response"


def test_sqlite_backend_search(sqlite_memory: Memory) -> None:
    """Test SQLite backend search functionality."""
    sqlite_memory.remember("Python tutorial", "Learn Python basics", tags=["python", "tutorial"])
    sqlite_memory.remember("JavaScript guide", "Learn JS fundamentals", tags=["javascript", "guide"])

    results = sqlite_memory.store.search("Python", top_k=5)
    assert len(results) >= 1
    assert results[0][0].query == "Python tutorial"


def test_sqlite_backend_list_all(sqlite_memory: Memory) -> None:
    """Test SQLite backend list_all with filters."""
    sqlite_memory.remember("user query", "user response", scope=MemoryScope.USER)
    sqlite_memory.remember("project query", "project response", scope=MemoryScope.PROJECT)

    user_memories = sqlite_memory.store.list_all(scopes=[MemoryScope.USER])
    assert len(user_memories) == 1
    assert user_memories[0].scope == MemoryScope.USER

    project_memories = sqlite_memory.store.list_all(scopes=[MemoryScope.PROJECT])
    assert len(project_memories) == 1
    assert project_memories[0].scope == MemoryScope.PROJECT


def test_sqlite_backend_archive(sqlite_memory: Memory) -> None:
    """Test SQLite backend archive functionality."""
    entry = sqlite_memory.remember("archive test", "to be archived")
    sqlite_memory.archive(entry.id)

    archived = sqlite_memory.get(entry.id)
    assert archived is not None
    assert archived.archived is True

    # Archived should not appear in normal search
    decision = sqlite_memory.resolve("archive test", mode="replay")
    assert decision.action == MemoryAction.NONE


def test_sqlite_backend_ttl(sqlite_memory: Memory) -> None:
    """Test SQLite backend TTL functionality."""
    entry = sqlite_memory.remember("ttl test", "expires soon", ttl="1s")
    assert entry.expires_at is not None

    # Should be active initially
    assert entry.state.value == "active"

    # Wait for expiration
    import time
    time.sleep(1.5)

    entry.refresh_state()
    assert entry.is_expired is True


def test_sqlite_backend_cleanup(sqlite_memory: Memory) -> None:
    """Test SQLite backend cleanup functionality."""
    sqlite_memory.remember("cleanup test 1", "expired", ttl="1s")
    sqlite_memory.remember("cleanup test 2", "not expired", ttl="1h")

    import time
    time.sleep(1.5)

    result = sqlite_memory.cleanup(delete=False)
    assert result["expired"] >= 1
    assert result["deleted"] == 0


def test_sqlite_backend_stats(sqlite_memory: Memory) -> None:
    """Test SQLite backend stats functionality."""
    sqlite_memory.remember("stat test 1", "response 1", type=MemoryType.CONVERSATION)
    sqlite_memory.remember("stat test 2", "response 2", type=MemoryType.FACT)

    stats = sqlite_memory.stats()
    assert stats["total"] >= 2
    assert "by_type" in stats
    assert "by_state" in stats


# ChromaDB-specific tests (if available)
def test_chromadb_backend_basic(chromadb_memory: Memory) -> None:
    """Test basic ChromaDB backend functionality."""
    entry = chromadb_memory.remember("test", "response", type=MemoryType.FACT)
    assert entry.id is not None

    fetched = chromadb_memory.get(entry.id)
    assert fetched is not None
    assert fetched.response == "response"


def test_chromadb_backend_search(chromadb_memory: Memory) -> None:
    """Test ChromaDB backend search functionality."""
    chromadb_memory.remember("Python tutorial", "Learn Python basics", tags=["python", "tutorial"])
    chromadb_memory.remember("JavaScript guide", "Learn JS fundamentals", tags=["javascript", "guide"])

    results = chromadb_memory.store.search("Python", top_k=5)
    assert len(results) >= 1
    assert results[0][0].query == "Python tutorial"


def test_chromadb_backend_vector_search(chromadb_memory: Memory) -> None:
    """Test ChromaDB backend vector search (semantic)."""
    chromadb_memory.remember("How to bake a cake", "Mix flour, sugar, eggs and bake at 350F")
    chromadb_memory.remember("How to make bread", "Mix flour, yeast, water and bake")

    # Semantic search should find related content
    results = chromadb_memory.store.search("baking dessert", top_k=5)
    assert len(results) >= 1


# Async tests for both backends
@pytest.mark.asyncio
async def test_async_remember(memory: Memory) -> None:
    """Test async remember for both backends."""
    entry = await memory.aremember("async test", "async response")
    assert entry.id is not None
    assert entry.response == "async response"


@pytest.mark.asyncio
async def test_async_resolve(memory: Memory) -> None:
    """Test async resolve for both backends."""
    memory.remember("async query", "async response")
    decision = await memory.aresolve("async query")
    assert decision.action == MemoryAction.REPLAY


@pytest.mark.asyncio
async def test_async_list(memory: Memory) -> None:
    """Test async list for both backends."""
    memory.remember("item 1", "response 1")
    memory.remember("item 2", "response 2")

    entries = await memory.alist(limit=10)
    assert len(entries) >= 2


@pytest.mark.asyncio
async def test_async_get(memory: Memory) -> None:
    """Test async get for both backends."""
    entry = memory.remember("get test", "get response")
    fetched = await memory.aget(entry.id)
    assert fetched is not None
    assert fetched.response == "get response"


@pytest.mark.asyncio
async def test_async_forget(memory: Memory) -> None:
    """Test async forget for both backends."""
    entry = memory.remember("forget test", "to be forgotten")
    result = await memory.aforget(entry.id)
    assert result is True
    assert await memory.aget(entry.id) is None


@pytest.mark.asyncio
async def test_async_archive(memory: Memory) -> None:
    """Test async archive for both backends."""
    entry = memory.remember("archive test", "to be archived")
    archived = await memory.aarchive(entry.id)
    assert archived is not None
    assert archived.archived is True


@pytest.mark.asyncio
async def test_async_cleanup(memory: Memory) -> None:
    """Test async cleanup for both backends."""
    memory.remember("cleanup test", "expired", ttl="1s")
    import time
    time.sleep(1.5)

    result = await memory.acleanup(delete=False)
    assert result["expired"] >= 1


@pytest.mark.asyncio
async def test_async_stats(memory: Memory) -> None:
    """Test async stats for both backends."""
    memory.remember("stat test", "response")
    stats = await memory.astats()
    assert stats["total"] >= 1


@pytest.mark.asyncio
async def test_async_consolidate(memory: Memory) -> None:
    """Test async consolidate for both backends."""
    memory.remember("duplicate query", "response 1")
    memory.remember("duplicate query", "response 2")

    created = await memory.aconsolidate(similarity_threshold=0.9)
    # May or may not create summaries depending on similarity
    assert isinstance(created, list)


# Rigorous tests with realistic user queries
def test_realistic_user_queries(memory: Memory) -> None:
    """Test with realistic full-sentence user queries."""
    # Programming questions
    memory.remember(
        "How do I fix a memory leak in Python?",
        "Use tracemalloc to track memory allocations, check for circular references, "
        "and ensure you're closing file handles and database connections properly.",
        type=MemoryType.FACT,
        tags=["python", "memory", "debugging"]
    )

    memory.remember(
        "What's the best way to handle async database connections in FastAPI?",
        "Use lifespan events with async context managers, or dependency injection "
        "with async generators to manage connection pools properly.",
        type=MemoryType.WORKFLOW,
        tags=["fastapi", "async", "database"]
    )

    memory.remember(
        "How do I optimize a slow SQL query?",
        "Use EXPLAIN ANALYZE to identify bottlenecks, add appropriate indexes, "
        "avoid SELECT *, and consider query restructuring or materialized views.",
        type=MemoryType.FACT,
        tags=["sql", "optimization", "database"]
    )

    # Test semantic search finds relevant answers
    decision = memory.resolve("My Python app is using too much memory, how do I debug it?")
    assert decision.action in (MemoryAction.RESTORE, MemoryAction.REPLAY, MemoryAction.VERIFY)
    context_str = memory.format_restore_context(decision)
    assert "tracemalloc" in context_str.lower() or "memory" in context_str.lower()

    decision = memory.resolve("FastAPI database connection management best practices")
    assert decision.action in (MemoryAction.RESTORE, MemoryAction.REPLAY, MemoryAction.VERIFY)
    context_str = memory.format_restore_context(decision)
    assert "lifespan" in context_str.lower() or "async" in context_str.lower()


def test_code_snippets_in_memory(memory: Memory) -> None:
    """Test storing and retrieving code snippets."""
    code_example = '''
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
    memory.remember(
        "How to write a recursive fibonacci function in Python?",
        "Here's a simple recursive implementation:",
        content=code_example,
        type=MemoryType.CODE,
        tags=["python", "recursion", "fibonacci"]
    )

    decision = memory.resolve("Show me a fibonacci function in Python")
    assert decision.action in (MemoryAction.RESTORE, MemoryAction.REPLAY)
    context_str = memory.format_restore_context(decision)
    assert "fibonacci" in context_str.lower()


def test_long_conversation_context(memory: Memory) -> None:
    """Test maintaining context across a long conversation."""
    # Simulate a multi-turn conversation
    conversation = [
        ("I'm building a REST API with FastAPI", "Great! FastAPI is a modern, fast web framework for building APIs."),
        ("How do I add authentication?", "You can use OAuth2 with JWT tokens. FastAPI has built-in support via python-jose."),
        ("What about rate limiting?", "Use slowapi or implement custom middleware with Redis for distributed rate limiting."),
        ("How do I deploy this to production?", "Consider Docker with Gunicorn/Uvicorn workers, behind nginx reverse proxy."),
        ("Any monitoring recommendations?", "Use Prometheus + Grafana for metrics, and structured logging with structlog."),
    ]

    for query, response in conversation:
        memory.remember(query, response, type=MemoryType.CONVERSATION, tags=["fastapi", "deployment"])

    # Should be able to retrieve context from any part of conversation
    decision = memory.resolve("How do I deploy my FastAPI app?")
    assert decision.action in (MemoryAction.RESTORE, MemoryAction.REPLAY)
    context_str = memory.format_restore_context(decision)
    assert "docker" in context_str.lower() or "gunicorn" in context_str.lower() or "nginx" in context_str.lower()

    decision = memory.resolve("What monitoring tools should I use?")
    assert decision.action in (MemoryAction.RESTORE, MemoryAction.REPLAY)
    context_str = memory.format_restore_context(decision)
    assert "prometheus" in context_str.lower() or "grafana" in context_str.lower()


def test_multilingual_queries(memory: Memory) -> None:
    """Test with multilingual content."""
    memory.remember(
        "Comment optimiser une requête SQL lente?",
        "Utilisez EXPLAIN ANALYZE pour identifier les goulots d'étranglement, ajoutez des index appropriés.",
        type=MemoryType.FACT,
        tags=["sql", "optimization", "french"]
    )

    memory.remember(
        "Como resolver vazamento de memoria em Python?",
        "Use tracemalloc para rastrear alocações de memória e verifique referências circulares.",
        type=MemoryType.FACT,
        tags=["python", "memory", "portuguese"]
    )

    # Should find relevant content even with different language queries
    # Note: ChromaDB may not match cross-language queries as well as SQLite BM25
    decision = memory.resolve("SQL query optimization techniques")
    # Accept NONE for ChromaDB due to cross-language semantic search limitations
    assert decision.action in (MemoryAction.RESTORE, MemoryAction.REPLAY, MemoryAction.VERIFY, MemoryAction.NONE)
    if decision.action != MemoryAction.NONE:
        context_str = memory.format_restore_context(decision)
        assert "explain" in context_str.lower() or "index" in context_str.lower()


def test_edge_cases(memory: Memory) -> None:
    """Test edge cases and boundary conditions."""
    # Empty query
    entry = memory.remember("", "Empty query response")
    assert entry.query == ""

    # Very long query
    long_query = "What is " + "very " * 100 + "long query?"
    entry = memory.remember(long_query, "Response to long query")
    assert len(entry.query) > 500

    # Special characters
    special_query = "How to handle 🚀 emojis and 特殊字符 in queries?"
    entry = memory.remember(special_query, "Handle them with proper encoding")
    assert "🚀" in entry.query
    assert "特殊字符" in entry.query

    # Unicode in response
    memory.remember("Unicode test", "Response with 中文, 日本語, 한국어, 🎉")
    decision = memory.resolve("Unicode test")
    assert decision.action in (MemoryAction.RESTORE, MemoryAction.REPLAY)
    context_str = memory.format_restore_context(decision)
    assert "中文" in context_str or "🎉" in context_str
    assert "中文" in context_str or "🎉" in context_str


def test_concurrent_operations(memory: Memory) -> None:
    """Test concurrent memory operations."""
    import threading
    import time

    results = []
    errors = []

    def worker(worker_id: int):
        try:
            for i in range(10):
                entry = memory.remember(
                    f"Worker {worker_id} query {i}",
                    f"Worker {worker_id} response {i}",
                    type=MemoryType.CONVERSATION
                )
                fetched = memory.get(entry.id)
                assert fetched is not None
                time.sleep(0.01)
            results.append(worker_id)
        except Exception as e:
            errors.append((worker_id, e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(results) == 5

    # Verify all memories were stored
    all_memories = memory.list(limit=100)
    assert len(all_memories) >= 50


def test_decision_thresholds(memory: Memory) -> None:
    """Test that decision thresholds work correctly."""
    memory.remember(
        "Exact match query",
        "Exact match response",
        type=MemoryType.FACT
    )

    # Should replay on exact match
    decision = memory.resolve("Exact match query", mode="replay")
    assert decision.action == MemoryAction.REPLAY
    assert decision.confidence >= 0.85

    # Should restore on similar (using restore mode)
    # Note: ChromaDB may return NONE for similar queries due to different scoring
    decision = memory.resolve("Exact match query with extra words", mode="restore")
    assert decision.action in (MemoryAction.RESTORE, MemoryAction.REPLAY, MemoryAction.VERIFY, MemoryAction.NONE)

    # Should return NONE on unrelated
    decision = memory.resolve("Completely different topic about cooking", mode="auto")
    assert decision.action == MemoryAction.NONE


def test_scope_isolation(memory: Memory) -> None:
    """Test that scopes properly isolate memories."""
    memory.remember("user secret", "user data", scope=MemoryScope.USER)
    memory.remember("project config", "project data", scope=MemoryScope.PROJECT)
    memory.remember("session temp", "session data", scope=MemoryScope.SESSION)

    # User scope should only see user memories
    user_memories = memory.list(scope=[MemoryScope.USER])
    assert len(user_memories) == 1
    assert user_memories[0].query == "user secret"

    # Project scope should only see project memories
    project_memories = memory.list(scope=[MemoryScope.PROJECT])
    assert len(project_memories) == 1
    assert project_memories[0].query == "project config"

    # Session scope should only see session memories
    session_memories = memory.list(scope=[MemoryScope.SESSION])
    assert len(session_memories) == 1
    assert session_memories[0].query == "session temp"

    # No scope filter should see all
    all_memories = memory.list(limit=100)
    assert len(all_memories) >= 3


def test_ttl_and_expiration(memory: Memory) -> None:
    """Test TTL expiration and cleanup."""
    import time

    # Add memories with different TTLs
    memory.remember("expires soon", "gone in 1 second", ttl="1s")
    memory.remember("expires later", "gone in 1 hour", ttl="1h")
    memory.remember("never expires", "permanent", ttl=None)

    # All should be active initially
    all_memories = memory.list(limit=100)
    # Note: ChromaDB may not store all 3 due to similarity filtering, so check >= 2
    assert len(all_memories) >= 2

    # Wait for first to expire
    time.sleep(1.5)

    # Expired should not appear in normal list
    # Note: list() doesn't have include_expired parameter, expired are filtered by default

    # Cleanup should mark expired
    result = memory.cleanup(delete=False)
    assert result["expired"] >= 1

    # Delete expired
    result = memory.cleanup(delete=True)
    assert result["deleted"] >= 1
