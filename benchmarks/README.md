# FrontierGuard DEA conformance benchmarks

This directory is the executable seed of a cross-language DEA conformance
suite. It currently proves only the local FrontierGuard adapter against a
small radial DEA oracle set. It does **not** claim that R, Python, Julia, or
MATLAB packages have already been run or found conformant.

The suite separates four things that are easy to conflate:

1. a mathematical case and its declared conventions;
2. an expected oracle with documented authority;
3. an implementation-specific adapter result; and
4. a validator that compares normalized outputs and mathematical relations.

See [PROTOCOL.md](PROTOCOL.md) for the research protocol and
[adapters/SPECIFICATION.md](adapters/SPECIFICATION.md) for the contract that
future competitor adapters must follow.

## Current scope

The initial manifest covers:

- a three-DMU, hand-calculated CRS input-oriented envelopment case;
- the repository's five-DMU rational CRS input-oriented case;
- the same five-DMU case under CRS output orientation; and
- the same five-DMU case under VRS input orientation.

Every case uses the phase-one radial result (`slacks = "none"`). Slack
projections, peer weights, non-radial methods, economic models, productivity
indices, and network models require separate oracle contracts and are not
silently treated as covered.

The five-DMU values are rational and independently checkable, but the
repository has not yet recovered a primary bibliographic source for the
fixture. Its case files therefore label it an analytical repository oracle,
not a completed external cross-language validation.

## Requirements

- Python 3.10 or newer;
- the FrontierGuard source checkout containing this directory; and
- the project's development dependencies, including the Draft 2020-12 JSON
  Schema validator.

From the repository root, either use the project environment:

```bash
uv sync --all-groups
uv run python benchmarks/scripts/validate_assets.py
```

or an already prepared virtual environment:

```bash
.venv/bin/python benchmarks/scripts/validate_assets.py
```

The scripts add `src/` to `sys.path`, so an editable installation is not
required.

## Run the local adapter

Choose a disposable output directory. Generated results contain timestamps and
environment details and should not be mistaken for source oracles.

```bash
benchmark_results="$(mktemp -d)"
uv run python benchmarks/scripts/run_frontierguard.py \
  --output-dir "$benchmark_results"
uv run python benchmarks/scripts/validate_results.py \
  --results-dir "$benchmark_results"
```

The same flow with the repository virtual environment is:

```bash
benchmark_results="$(mktemp -d)"
.venv/bin/python benchmarks/scripts/run_frontierguard.py \
  --output-dir "$benchmark_results"
.venv/bin/python benchmarks/scripts/validate_results.py \
  --results-dir "$benchmark_results"
```

Run one case by manifest ID:

```bash
uv run python benchmarks/scripts/run_frontierguard.py \
  --case radial-crs-input-analytical-3dmu \
  --output-dir /tmp/frontierguard-one-case
uv run python benchmarks/scripts/validate_results.py \
  --case radial-crs-input-analytical-3dmu \
  --results-dir /tmp/frontierguard-one-case
```

All commands return a non-zero exit code for a schema-invalid asset or result
(including schema-forbidden extra fields), failed solve, non-optimal baseline
or metamorphic execution status, missing result, oracle mismatch, failed
metamorphic relation, or failed cross-case relation.

## Files

```text
benchmarks/
├── PROTOCOL.md
├── README.md
├── adapters/
│   └── SPECIFICATION.md
├── cases/
│   ├── manifest.json
│   ├── radial_crs_input_analytical_3dmu.json
│   ├── radial_crs_input_five_dmu.json
│   ├── radial_crs_output_five_dmu.json
│   └── radial_vrs_input_five_dmu.json
├── citations/
│   └── REFERENCES.md
├── schemas/
│   ├── case.schema.json
│   ├── manifest.schema.json
│   └── result.schema.json
└── scripts/
    ├── _common.py
    ├── run_frontierguard.py
    ├── validate_assets.py
    └── validate_results.py
```

## Adding an external adapter

An adapter for another package must:

1. consume a case file without changing its mathematical assumptions;
2. record the package, runtime, version, solver, and every normalization;
3. emit a document satisfying `schemas/result.schema.json`;
4. retain native output values when it derives a normalized comparison field;
5. use the failure taxonomy in `PROTOCOL.md`; and
6. never copy implementation code or restricted fixtures into this
   MIT-licensed repository.

Adapters are research instruments. A package is not called conformant until
its output was generated in a pinned, reproducible environment and the result
passes the oracle, metamorphic, and applicable cross-case checks.

## Licensing and citation

Files created in this directory are covered by the repository's MIT license.
The initial numerical cases are small analytical facts, but their provenance is
still recorded in each case. Third-party datasets must carry a redistribution
review and explicit source citation before being added.

Use the project-level `CITATION.cff` to cite FrontierGuard. Method and solver
references used by this benchmark are listed in
[citations/REFERENCES.md](citations/REFERENCES.md).
