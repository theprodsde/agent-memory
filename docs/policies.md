# Policies

Default policy weights:

- semantic (55%)
- recency (15%)
- confidence (20%)
- usage (10%)

```python
from agent_memory import Memory, DefaultPolicy

memory = Memory(policy=DefaultPolicy(semantic_weight=0.6))
```

Custom policies implement `DecisionPolicy` with `score()` and `select_action()`.
