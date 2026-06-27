# Memory Model

States: `active`, `archived`, `expired`, `deleted`

Types: `conversation`, `fact`, `workflow`, `document`, `tool_output`, `code`, `summary`, `preference`

Scopes: `session`, `user`, `project`, `workspace`, `team`, `global`

```python
memory.remember(
    query="...",
    response="...",
    type="fact",
    scope="project",
    ttl="30d",
)
```
