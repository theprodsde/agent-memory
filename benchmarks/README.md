# Evaluation datasets for `agent-memory eval`

Each JSON file defines:

- `memories` — seeded before evaluation
- `cases` — query + expected action (`replay`, `restore`, `verify`, `none`)

Run:

```bash
agent-memory eval
agent-memory benchmark --seed
```
