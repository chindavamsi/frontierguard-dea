## Summary

Describe the problem and the smallest coherent change that solves it.

## Validation

List the exact commands you ran and their results.

```text
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest --cov=frontierguard
uv build --no-sources --clear
```

## Mathematical contract

For a new or changed DEA method, include:

- [ ] objective, constraints, conventions, and edge cases;
- [ ] hand-calculated or published golden cases;
- [ ] feasibility and identity tests;
- [ ] scale, permutation, and unit-invariance tests where valid;
- [ ] an independent cross-language or cross-package oracle; and
- [ ] documented numerical and infeasibility behavior.

Write “Not applicable” for a change that cannot alter analytical results.

## Checklist

- [ ] The change is focused and linked to an issue when appropriate.
- [ ] Public behavior and typing are tested.
- [ ] Documentation and examples reflect the final API.
- [ ] `docs/FEATURE_MATRIX.md` reports the method's true validation status.
- [ ] `CHANGELOG.md` includes a concise user-facing entry.
- [ ] No generated files, credentials, or confidential research data are included.
- [ ] I agree that my contribution is licensed under the repository's MIT License.
