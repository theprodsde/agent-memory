# Development Guide

> **Note:** For quick installation and basic usage, see the [README.md](README.md).
> This guide is for **developers** who want to contribute, extend, or debug Agent Memory.

---

## Development Setup

### Prerequisites
- Python 3.10+
- Git
- Docker (optional, for containerized development)

### Clone and Setup

#### Windows (PowerShell)
```powershell
git clone https://github.com/TheProdSDE/agent-memory.git
cd agent-memory
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

#### Windows (Command Prompt)
```cmd
git clone https://github.com/TheProdSDE/agent-memory.git
cd agent-memory
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e ".[dev]"
```

#### Linux/macOS
```bash
git clone https://github.com/TheProdSDE/agent-memory.git
cd agent-memory
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Docker Development

### Build the Image
```bash
docker build -t agent-memory:latest .
```

### Run CLI Commands (with persistent data volume)
```bash
# Show help
docker run --rm agent-memory:latest agent-memory --help

# Store a memory
docker run --rm -v agent_memory_data:/home/appuser/.agent_memory \
  agent-memory:latest agent-memory remember "How do I reset my password?" "Go to Settings → Security → Reset Password"

# Resolve a query
docker run --rm -v agent_memory_data:/home/appuser/.agent_memory \
  agent-memory:latest agent-memory resolve "How do I reset my password?"

# Show stats
docker run --rm -v agent_memory_data:/home/appuser/.agent_memory \
  agent-memory:latest agent-memory stats

# Run benchmark
docker run --rm -v agent_memory_data:/home/appuser/.agent_memory \
  agent-memory:latest agent-memory benchmark --seed

# Run evaluation
docker run --rm -v agent_memory_data:/home/appuser/.agent_memory \
  agent-memory:latest agent-memory eval
```

### Run MCP Server (for Cursor/VS Code integration)
```bash
# Start MCP server in background
docker run -d --name agent-memory-mcp \
  -v agent_memory_data:/home/appuser/.agent_memory \
  -p 8000:8000 \
  agent-memory:latest agent-memory-mcp

# View logs
docker logs -f agent-memory-mcp

# Stop
docker stop agent-memory-mcp
```

### Using Docker Compose (Recommended for Docker)
```bash
# Start CLI environment
docker-compose up agent-memory

# Start MCP server
docker-compose up agent-memory-mcp

# Run one-off commands
docker-compose run --rm agent-memory agent-memory remember "test" "response"
docker-compose run --rm agent-memory agent-memory resolve "test"
docker-compose run --rm agent-memory agent-memory stats
```

---

## Installing from PyPI (Production)

```bash
# Install from PyPI
pip install agent-memory

# Or install from GitHub
pip install git+https://github.com/TheProdSDE/agent-memory.git

# Verify
agent-memory --help
```

---

## Python API Usage (Library Mode)

```python
from agent_memory import Memory, MemoryAction, MemoryType

# Initialize
memory = Memory(persist_dir=".agent_memory")

# Store memories
memory.remember(
    query="How do I reset my password?",
    response="Go to Settings → Security → Reset Password and follow the email link.",
    type=MemoryType.CONVERSATION,
    tags=["auth", "faq"]
)

memory.remember(
    query="Current API rate limit",
    response="1000 requests/minute per API key.",
    type=MemoryType.FACT,
    requires_verification=True
)

# Resolve queries
decision = memory.resolve("How do I reset my password?")

match decision.action:
    case MemoryAction.REPLAY:
        print(f"Replay: {decision.response}")
    case MemoryAction.RESTORE:
        print("Context for synthesis:")
        print(memory.format_restore_context(decision))
    case MemoryAction.VERIFY:
        print("Verify before reuse:")
        print(memory.format_verify_context(decision))
    case MemoryAction.NONE:
        print("No relevant memory - answer from scratch")
```

---

## CLI Commands Reference

| Command | Description | Example |
|---------|-------------|---------|
| `remember` | Store a query/response pair | `agent-memory remember "query" "response" --type fact --scope user --ttl 30d` |
| `resolve` | Query memory and get decision | `agent-memory resolve "query" --explain` |
| `stats` | Show memory statistics | `agent-memory stats` |
| `cleanup` | Expire/delete stale memories | `agent-memory cleanup --delete` |
| `benchmark` | Run latency/hit-rate benchmark | `agent-memory benchmark --seed --repeat 3` |
| `eval` | Run evaluation datasets | `agent-memory eval --datasets ./benchmarks/datasets` |

### Common Options
- `--data-dir PATH` - Persistence directory (default: `.agent_memory`)

---

## MCP Server for Cursor/VS Code

### Option 1: Local Installation
Add to `~/.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "agent-memory-mcp",
      "env": {
        "AGENT_MEMORY_DIR": "C:\\Users\\YourName\\.agent_memory"
      }
    }
  }
}
```

### Option 2: Docker-based (Recommended for isolation)
```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "agent_memory_data:/home/appuser/.agent_memory",
        "agent-memory:latest",
        "agent-memory-mcp"
      ]
    }
  }
}
```

### Available MCP Tools
- `remember_memory` - Store a query/response pair
- `resolve_memory` - Retrieve and decide action
- `list_memories` - List with pagination
- `get_memory` - Fetch single memory
- `forget_memory` - Delete memory
- `archive_memory` - Archive memory
- `consolidate_memories` - Merge duplicates

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MEMORY_DIR` | `~/.agent_memory` | Persistence directory |
| `AGENT_MEMORY_COLLECTION` | `agent_memories` | ChromaDB collection name |

---

## Data Persistence

### Virtual Environment
Data stored in `.agent_memory/` (project dir) or `~/.agent_memory/` (global)

### Docker
Data persisted in Docker volume `agent_memory_data`

### Custom Location
```bash
# CLI
agent-memory --data-dir /custom/path remember "q" "r"

# Python
memory = Memory(persist_dir="/custom/path")

# Docker
docker run -v /host/path:/home/appuser/.agent_memory ...
```

---

## Makefile Commands

We provide a `Makefile` for common development tasks:

```bash
# Install dependencies
make install

# Run tests
make test

# Run tests with coverage
make test-cov

# Run linting
make lint

# Run type checking
make typecheck

# Run all checks
make check

# Run benchmarks
make benchmark
make benchmark-full

# Run evaluation
make eval

# Build Docker image
make docker-build

# Clean build artifacts
make clean

# Reset database
make reset-db
```

---

## Development Workflow

### Run Tests
```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_memory.py -v

# With coverage
pip install pytest-cov
python -m pytest tests/ --cov=agent_memory --cov-report=html
```

### Run Benchmarks
```bash
# Quick benchmark
agent-memory benchmark

# With seeded data from eval datasets
agent-memory benchmark --seed --repeat 3

# Custom baseline
agent-memory benchmark --baseline-ms 500
```

### Run Evaluation
```bash
# All datasets
agent-memory eval

# Specific directory
agent-memory eval --datasets ./benchmarks/datasets
```

### Code Quality
```bash
# Linting
ruff check agent_memory/ tests/

# Type checking
mypy agent_memory/ --ignore-missing-imports

# Formatting
ruff format agent_memory/ tests/
```

---

## Troubleshooting

### ChromaDB Issues
```bash
# Reset database
rm -rf .agent_memory
# or in Docker
docker volume rm agent_memory_data
```

### Import Errors
```bash
# Reinstall in development mode
pip install -e ".[dev]" --force-reinstall --no-deps
```

### MCP Server Not Found
```bash
# Ensure package is installed
pip install -e "."

# Or use full module path
python -m mcp_server.server
```

### Port Conflicts (MCP Server)
```bash
# Change port in docker-compose.yml or use different port
docker run -p 8001:8000 ...
```

---

## Project Structure
```
agent-memory/
├── agent_memory/          # Core package
│   ├── __init__.py        # Exports
│   ├── manager.py         # Memory SDK (main API)
│   ├── store.py           # ChromaDB storage
│   ├── retriever.py       # Hybrid retrieval (BM25 + vector)
│   ├── decision.py        # Decision engine
│   ├── policy.py          # Scoring policy
│   ├── models.py          # Data models
│   ├── cli.py             # CLI commands
│   └── ...                # Other modules
├── mcp_server/            # MCP server
│   └── server.py          # FastMCP server
├── benchmarks/            # Evaluation datasets
├── tests/                 # Unit tests
├── examples/              # Usage examples
├── docs/                  # Documentation
├── Dockerfile             # Multi-stage Docker build
├── docker-compose.yml     # Docker Compose config
├── pyproject.toml         # Package config
└── README.md              # Main documentation
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Contribution Steps
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `python -m pytest tests/`
5. Run linting: `ruff check agent_memory/ tests/`
6. Commit and push
7. Open a Pull Request

---

## Next Steps

1. **Try the example**: `python examples/basic_usage.py`
2. **Run benchmarks**: `agent-memory benchmark --seed`
3. **Integrate with Cursor**: Configure MCP server
4. **Read the docs**: Check `docs/` folder for architecture, policies, and FAQ