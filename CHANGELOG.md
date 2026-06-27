# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- SQLite backend (`SqliteMemoryStore`) as default lightweight storage
- Async API support (`aremember`, `aresolve`, `alist`, `aget`, `aforget`, `aarchive`, `acleanup`, `astats`, `aconsolidate`)
- Backend selection via `backend` parameter in `Memory` constructor (`"chromadb"` or `"sqlite"`)
- Comprehensive test coverage for both ChromaDB and SQLite backends
- Package distribution configuration (wheel, sdist)
- Docker support with both backends

### Changed
- **Default backend changed from ChromaDB to SQLite** for lightweight deployments
- Version bumped to `0.1.0-alpha` (was incorrectly `0.3.0` in code/docs)
- Updated roadmap to reflect completed features

### Fixed
- Version inconsistency across `pyproject.toml`, `agent_memory/__init__.py`, and `README.md`
- Clean code principles and SOLID OOP compliance across all modules

## [0.1.0-alpha] - 2026-06-28

### Added
- Initial release of Agent Memory
- Decision-based memory layer (Replay / Restore / Verify / None)
- Hybrid retrieval: BM25 keyword search + Vector semantic search with RRF fusion
- Multi-factor scoring policy (semantic 70%, recency 15%, confidence 20%, frequency 10%)
- Structured memory with types (conversation, fact, workflow, document, tool_output, code, summary, preference)
- Scoped memory (session, user, project, workspace, team, global)
- TTL support with flexible duration strings (e.g., "30d", "2h")
- Full observability via `decision.explain()`
- CLI with remember, resolve, stats, benchmark, eval, cleanup commands
- MCP server integration for Cursor, VS Code, and other MCP clients
- Docker support with multi-stage build
- Comprehensive documentation (architecture, benchmarks, examples, FAQ, getting started, memory model, policies)
- Benchmark and evaluation datasets (coding_agent, customer_support, research_agent)
- CI/CD pipeline with GitHub Actions
- Pre-commit hooks (ruff, mypy, black)

### Security
- No known vulnerabilities

---

## Release Notes Template

### [X.Y.Z] - YYYY-MM-DD

#### Added
- New features

#### Changed
- Changes in existing functionality

#### Deprecated
- Soon-to-be removed features

#### Removed
- Removed features

#### Fixed
- Bug fixes

#### Security
- Security fixes