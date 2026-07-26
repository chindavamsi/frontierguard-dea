# Validation strategy

## Test pyramid

### Unit and contract tests

- matrix shape, finiteness, non-negativity, labels, and immutability;
- enum and method configuration;
- LP status mapping and failure messages;
- score convention, target, slack, and peer serialization.

### Algebra and feasibility properties

- primal inequalities and returns-to-scale constraints;
- target reconstruction from intensities;
- phase-two radial-factor preservation;
- serial network overall/product closure;
- independent scale-adjusted post-solve residual auditing;
- cost/revenue accounting and Farrell decomposition identities.

### Metamorphic tests

- positive unit changes leave dimensionless scores unchanged;
- row and variable permutations preserve aligned results;
- duplicating an efficient reference cannot worsen efficiency;
- adding a dominated reference cannot improve the frontier;
- CRS replication and scaling properties;
- serial and parallel execution agree.

### Independent oracles

Every model requires at least two:

1. a hand-calculated or rational fixture;
2. a published package dataset and documented expected output;
3. an independent multiplier/envelopment dual formulation; or
4. a cross-language comparison with R or Julia.

Using the same model builder with a different solver is not an independent
method oracle.

## Coverage targets

These are improvement targets. The enforced pre-alpha release gate is 90%
overall branch coverage; not every model module has reached the 95% target yet.

- at least 95% branch coverage for model construction, validation, and solver
  status handling;
- at least 90% for the package overall;
- 100% pass for feasibility, duality, and model-specific identities.

## Numerical policy

Tests use scale-aware feasibility tolerances. Result objects preserve normalized
solver status and solver identity and never silently replace infeasible or
unbounded values. Solver adapters retain raw backend diagnostics. Any clipping
is limited to residual values inside the declared numerical tolerance and
cannot alter an objective score.

Every returned optimal result records its maximum independently recomputed
residual and the tolerance used by the audit. A solution above tolerance fails
instead of being relabeled as valid.

## Release ladder

- **Pre-alpha:** core code and local oracles.
- **Alpha:** cross-language golden suite and full docs for each advertised
  method.
- **Beta:** real user studies, performance benchmarks, stable evidence schema,
  and external methodology review.
- **1.0:** every checked feature in the public matrix meets its oracle gate and
  the API has a published compatibility policy.
