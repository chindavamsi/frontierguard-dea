# Executable examples

Run any example from an installed development environment:

```bash
uv run python examples/classical.py
uv run python examples/multistage.py
uv run python examples/economic.py
uv run python examples/productivity.py
```

Each example uses a small auditable fixture and prints the result table plus a
model-specific diagnostic. Together they cover classical, multi-stage,
economic, and productivity workflows. The equations and score conventions are
documented in [`docs/METHODS.md`](../docs/METHODS.md).
