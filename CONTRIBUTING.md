# Contributing to FrontierGuard DEA

Thank you for helping build a trustworthy open-source DEA toolkit. Contributions
are welcome from researchers, practitioners, educators, and software engineers.

FrontierGuard is an alpha research preview. Correctness and transparent model
conventions take priority over expanding the feature count.

## Before you start

- Read the [method contracts](docs/METHODS.md), [feature matrix](docs/FEATURE_MATRIX.md),
  and [validation policy](docs/VALIDATION.md).
- Search the issue tracker before opening a new issue.
- Open a method proposal before implementing a new DEA formulation or changing
  a mathematical convention.
- Report vulnerabilities privately according to [SECURITY.md](SECURITY.md).
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md).

Small bug fixes and documentation improvements can go directly to a pull
request. Discuss large API, solver, data-model, or architecture changes first.

## Development setup

FrontierGuard supports Python 3.10 through 3.13 and uses
[uv](https://docs.astral.sh/uv/) for reproducible environments.

```bash
git clone https://github.com/chindavamsi/frontierguard-dea.git
cd frontierguard-dea
uv sync --all-groups
uv run pytest
```

The committed `uv.lock` is the dependency source of truth for development and
CI. Do not hand-edit it. After intentionally changing dependencies, regenerate
it with `uv lock` and include the resulting diff.

## Required local checks

Run the full gate before requesting review:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy
uv run pytest --cov=frontierguard --cov-report=term-missing
uv build --no-sources --clear
uvx --from twine==6.2.0 twine check --strict dist/*
```

CI repeats these checks and tests the supported Python matrix. The configured
branch-coverage floor is 90%; method implementations should normally have
substantially stronger behavioral coverage than that minimum.

## Adding or changing a DEA method

Every analytical method is an auditable contract. A method is not considered
supported until the pull request provides:

1. the objective, constraints, orientation, returns-to-scale assumptions,
   disposability conventions, normalization, and zero/negative-data behavior;
2. one or more hand-calculated or published golden cases;
3. primal feasibility and model-identity assertions, not only score snapshots;
4. permutation, scale, and unit-invariance tests wherever theory requires them;
5. comparison with an independent established implementation or a documented
   reason why no oracle exists;
6. explicit solver-status, infeasibility, degeneracy, and tolerance behavior;
7. immutable result metadata sufficient to identify the method conventions,
   solver, data fingerprint, and audit evidence (full reproduction still
   requires the original input data and environment); and
8. updates to `docs/METHODS.md`, `docs/FEATURE_MATRIX.md`,
   `docs/VALIDATION.md`, and `CHANGELOG.md`.

Do not silently resolve a literature ambiguity. Expose materially different
conventions through separate parameters or method descriptors and cite the
source defining each convention.

If test data comes from a paper or another package, document its provenance and
license. Prefer tiny synthetic fixtures that can be independently verified.
Never commit confidential, proprietary, or personally identifying research
data.

## Code and API expectations

- Keep public APIs typed and documented.
- Use explicit model classes rather than adding unrelated switches to one
  universal function.
- Route optimization through the shared solver and audit layers.
- Preserve input data and return immutable results.
- Raise FrontierGuard's public exception types for expected user-facing errors.
- Include solver status and numerical audit evidence in results.
- Avoid adding runtime dependencies unless they materially improve the package.
- Keep methods deterministic unless randomness is part of the stated contract;
  then expose and test a seed.

Tests belong under `tests/` and should describe user-visible behavior. Use
property-based tests for invariants that span many datasets.

## Documentation changes

Write for a researcher who knows DEA but may be new to Python. Examples should
be executable, use small data, name all important model assumptions, and show
how to inspect diagnostics. Do not describe a planned method as available.

## Pull requests

Create a focused branch, make logically grouped commits, and open a pull request
against `main`. Complete the pull-request template and link the relevant issue.
Draft pull requests are welcome for early design feedback.

A maintainer may request changes when:

- analytical behavior is under-specified;
- the evidence does not distinguish competing conventions;
- numerical assertions depend only on a single solver's printed status;
- the change broadens the API without a stable use case; or
- documentation overstates validation or support.

Approval does not transfer copyright. By submitting a contribution, you agree
that it is your original work (or that you have permission to submit it) and
license it under the project's [MIT License](LICENSE). FrontierGuard does not
currently require a contributor license agreement.

## Recognition

Substantial contributors may be added to [AUTHORS.md](AUTHORS.md) with their
consent. Every merged contribution remains permanently attributed in Git
history and the GitHub contributor graph.
