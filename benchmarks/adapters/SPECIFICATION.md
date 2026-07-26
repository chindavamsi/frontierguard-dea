# Competitor adapter specification

Version: **0.1.0**

This specification defines how R, Python, Julia, and MATLAB implementations
can join the benchmark without becoming dependencies of FrontierGuard. It is a
language-neutral contract, not evidence that those runtimes have already been
executed.

## Command contract

Each adapter should expose a non-interactive command with this shape:

```text
<runtime> <adapter-entrypoint> --case <absolute-case-json> --output <absolute-result-json>
```

The process must:

- read exactly one case satisfying `../schemas/case.schema.json`;
- write exactly one strict JSON result satisfying
  `../schemas/result.schema.json`;
- write diagnostics to standard error, not standard output;
- exit `0` only after a schema-valid result was written;
- use a non-zero exit for an adapter crash or inability to write its result;
- still write a normalized non-`success` result when the runtime remains able
  to classify a solver or support failure.

An orchestration layer may execute metamorphic variants itself or let the
adapter apply the transforms declared in the case. Either way, the emitted
`executions` list must contain `baseline` plus every declared variant.

## Required translation behavior

Adapters must set every model option explicitly. They may not rely on a
package's default orientation, returns to scale, second-stage optimization,
disposability, super-efficiency, or score direction.

The adapter must preserve:

- DMU, input, and output names;
- row/column alignment;
- the full- or external-reference declaration;
- orientation and returns-to-scale restrictions;
- self-exclusion policy;
- slack-phase policy;
- expected normalized status; and
- the case tolerance only as a comparison tolerance, not as permission to
  alter the solver model.

If the package cannot represent the exact model, report
`unsupported_model`. Do not approximate one model with another and call it a
conformance run.

## Score normalization

The common fields are:

- `radial_factor`: \(\theta\) for input orientation and \(\phi\) for output
  orientation;
- `technical_efficiency`: \(\theta\) for input orientation and \(1/\phi\) for
  output orientation.

If a package reports only one convention, the adapter may derive the other
only through an explicit, mathematically valid operation. Record it in
`implementation.normalizations`, for example:

```json
{
  "target_field": "technical_efficiency",
  "source_field": "native_output_efficiency_score",
  "operation": "reciprocal",
  "reason": "Package reports the output expansion factor phi."
}
```

Keep native values in `executions[].native_metrics` where practical. Never
clip an efficiency score into `[0, 1]`, because external-reference and
super-efficiency models may legitimately lie outside that interval.

## Result provenance

Populate:

- `implementation.name` and `implementation.version`;
- `implementation.adapter_name` and `adapter_version`;
- `implementation.language` and `runtime_version`;
- `implementation.package_source`;
- `implementation.package_license`;
- `implementation.solver_name` and `solver_version`, if exposed;
- `implementation.source_revision` for adapter code; and
- all conversions in `implementation.normalizations`.

Record the actual runtime version, not a compatibility target. If the package
does not expose its solver, write `unknown` and explain it in `notes`; do not
guess.

## Language-specific wrappers

The following are implementation sketches. Package calls and field names must
be checked against the exact pinned version before use.

### R

```text
Rscript benchmarks/adapters/r/<package>/run.R \
  --case /absolute/case.json \
  --output /absolute/result.json
```

Use a project-local `renv.lock`. Record `R.version.string`,
`packageVersion()`, BLAS details, and the solver reported by the package.
Candidate packages include Benchmarking and deaR, but their exact model
mapping and data licenses require adapter-specific review.

### Python

```text
python benchmarks/adapters/python/<package>/run.py \
  --case /absolute/case.json \
  --output /absolute/result.json
```

Use a dedicated virtual environment and lockfile rather than FrontierGuard's
environment. Candidate packages include pyDEA, Pyfrontier, and pyStoNED.
Where a project is unmaintained or constrains dependency versions, preserve a
container digest and do not loosen dependencies silently.

### Julia

```text
julia --project=benchmarks/adapters/julia/<package> \
  benchmarks/adapters/julia/<package>/run.jl \
  --case /absolute/case.json \
  --output /absolute/result.json
```

Commit `Project.toml` and `Manifest.toml` for a publication run. Record
`VERSION`, the package version, optimizer package/version, and solver
attributes. DataEnvelopmentAnalysis.jl is a candidate subject to exact
contract review.

### MATLAB

```text
matlab -batch "run_adapter( ...
  '/absolute/case.json', '/absolute/result.json')"
```

Record the MATLAB release, operating system, Optimization Toolbox version,
solver, and third-party toolbox version/commit. MATLAB itself and many DEA
toolboxes are not open source; never redistribute proprietary code, and
publish only outputs whose terms permit it. GNU Octave must be treated as a
separate runtime rather than being labelled MATLAB.

## Independent review checklist

Before an adapter's results enter a paper:

- [ ] Package documentation for the pinned version supports the exact model.
- [ ] Defaults are overridden explicitly.
- [ ] Input/output and DMU orientation have been inspected.
- [ ] Output score direction and reciprocal rules have been verified.
- [ ] Self-reference and reference-set handling are explicit.
- [ ] Native statuses map faithfully to the protocol taxonomy.
- [ ] Strict JSON contains no `NaN` or infinity tokens.
- [ ] Baseline oracle comparison passes or has an adjudication note.
- [ ] Metamorphic transformations were executed in the same environment.
- [ ] Adapter source was reviewed by someone other than its author.
- [ ] Package, solver, data, and adapter citations are present.
- [ ] Redistribution rights for source data and result artifacts are recorded.

## What conformance means

Passing a case means only that the pinned implementation agreed with that case
under the declared options and tolerances. It is not certification of the
whole package. A publication should report the exact case coverage and make
all failing or unsupported results visible.
