# Getting Started

```bash
pip install -e ".[dev]"
```

```python
from agent_memory import Memory

memory = Memory()
memory.remember("Reset password", "Settings → Security → Reset Password")
decision = memory.resolve("I forgot my password")

print(decision)
print(decision.explain())
```

## CLI

```bash
agent-memory remember "query" "response"
agent-memory resolve "query" --explain
agent-memory stats
agent-memory benchmark --seed
agent-memory eval
agent-memory cleanup
```
