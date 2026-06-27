# Architecture

```text
Query → BM25 + Vector Search → RRF Fusion → Policy Rerank → Decision Engine → Action
```

Components:

| Module | Role |
|--------|------|
| `store.py` | ChromaDB persistence |
| `retriever.py` | Hybrid retrieval |
| `policy.py` | Score composition |
| `decision.py` | Action selection |
| `explain.py` | Observability / debugging |
