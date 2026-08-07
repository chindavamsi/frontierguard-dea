# DEA conformance benchmark protocol

Protocol version: **0.1.0**

Status: **initial executable protocol; independent cross-language runs
pending**

## 1. Research question

For a fully specified DEA mathematical program, do independent software
implementations return numerically equivalent primary scores and satisfy the
same feasibility, identity, and invariance properties?

The benchmark evaluates conformance to a declared model contract. It does not
rank packages by convenience, feature count, speed, or popularity. A mismatch
is an observation to investigate, not proof that one implementation is wrong:
different defaults, degeneracy, alternate optima, or undocumented conventions
may explain it.

## 2. Unit of analysis

One case is one dataset plus one complete model specification. Orientation,
returns to scale, slack phase, exclusion policy, score convention, tolerances,
and expected fields are part of the case identity. Changing any of them creates
a new case.

The first protocol slice covers radial envelopment DEA only:

### Input-oriented model

For evaluated DMU \(o\):

\[
\min_{\theta,\lambda}\theta
\]

subject to:

\[
X^\top\lambda\leq\theta x_o,\quad
Y^\top\lambda\geq y_o,\quad
\lambda\geq0.
\]

The reported radial factor and technical efficiency both equal \(\theta\).

### Output-oriented model

\[
\max_{\phi,\lambda}\phi
\]

subject to:

\[
X^\top\lambda\leq x_o,\quad
Y^\top\lambda\geq\phi y_o,\quad
\lambda\geq0.
\]

The reported radial factor is \(\phi\), while normalized technical efficiency
is \(1/\phi\). Adapters must record any reciprocal conversion explicitly.

### Returns to scale

- CRS: no intensity-sum restriction.
- VRS: \(\mathbf{1}^\top\lambda=1\).
- NIRS: \(\mathbf{1}^\top\lambda\leq1\).
- NDRS: \(\mathbf{1}^\top\lambda\geq1\).

All initial cases evaluate each DMU against the full in-sample reference
technology and retain self in the reference set. Inputs and outputs are
non-negative, and larger output is preferred.

## 3. Oracle authority

Oracle authority is ordered but not collapsed into one label:

1. `analytical`: derived by hand, exact rational arithmetic, dominance, or a
   closed-form construction;
2. `primary_publication`: values or data recoverable from a cited primary
   paper;
3. `independent_formulation`: independently coded multiplier/envelopment or
   combinatorial formulation;
4. `external_implementation`: reproduced from a pinned version of another
   package in a documented environment.

An implementation cannot validate itself merely by regenerating its expected
file. Initial fixture values are committed source oracles; generated adapter
results go elsewhere.

Alternate DEA optima make peer intensities and multipliers unsuitable as
universal equality oracles unless uniqueness is proved. The first suite
therefore compares radial factors, normalized technical-efficiency scores,
statuses, and independent audit bounds. Later schemas may add set-valued or
feasibility-based comparisons for targets and weights.

## 4. Numerical comparison

For finite expected value \(e\) and observed value \(o\), equality passes when:

\[
|o-e|\leq a+r|e|,
\]

where \(a\) is the absolute tolerance and \(r\) is the relative tolerance
declared in the case. Initial defaults are \(a=10^{-9}\) and \(r=10^{-8}\).

Additional rules:

- array order is aligned by unique DMU name, never assumed from row position;
- non-finite JSON values are forbidden;
- an `optimal` status is required where the oracle declares it;
- the implementation's maximum audit residual must not exceed the case audit
  tolerance;
- rounded display tables are not oracle sources; and
- tolerance changes require a protocol revision or a case provenance note.

## 5. Metamorphic relations

The suite executes transformations from each case file and checks the declared
relation against the baseline:

### Positive unit scaling

Multiplying any input or output column by a strictly positive constant must
leave radial efficiency and radial factors invariant when both evaluated and
reference observations use the same units.

### DMU permutation

Permuting DMU rows must leave name-aligned scores invariant.

### Variable permutation

Permuting input columns among themselves or output columns among themselves,
with their names permuted identically, must leave scores invariant.

### Dominated-reference augmentation

Adding an observation that is strictly dominated by an existing reference must
not alter scores of the original DMUs. This transformation is specified for
future fixtures but is not in the first manifest because constructing a
universally dominated point depends on model orientation and zero components.

### Reference duplication

Duplicating a reference observation must not alter primary scores under radial
DEA. It may change a solver-selected intensity vector, so weights are not
compared.

Every metamorphic result is aligned back to the baseline DMU names. A relation
is a separate result from oracle agreement; both must pass.

## 6. Cross-case relations

The manifest can declare relations across cases sharing data. The initial
five-DMU CRS input/output pair asserts:

\[
\theta_o = 1/\phi_o
\]

for this single-output CRS fixture, expressed as equality between input
technical efficiency and output technical efficiency and as reciprocal radial
factors. This is an explicit fixture relation, not a blanket claim that input-
and output-oriented scores coincide under every technology.

## 7. Failure taxonomy

Adapters must report exactly one top-level outcome category:

| Category | Meaning |
|---|---|
| `success` | Adapter emitted complete finite outputs for every execution. |
| `invalid_input` | Case violates the implementation's documented input contract. |
| `unsupported_model` | Package cannot express the declared model without changing it. |
| `infeasible` | Solver established primal infeasibility for the declared program. |
| `unbounded` | Solver established unboundedness. |
| `iteration_limit` | Iteration or node limit ended the solve. |
| `time_limit` | Time limit ended the solve. |
| `solution_limit` | Configured solution limit ended the solve. |
| `numerical_failure` | Non-finite, unstable, or audit-invalid numerical result. |
| `solver_error` | Solver failed without a more specific normalized status. |
| `adapter_error` | Translation, process, parsing, or output-mapping failure. |
| `schema_error` | Case or produced result violates its interchange schema. |
| `crash` | Unhandled runtime failure. |

The validator may additionally report `oracle_mismatch`,
`metamorphic_mismatch`, or `cross_case_mismatch`; these are validation
findings, not solver statuses. Missing functionality is `unsupported_model`,
not a failed numerical result.

No adapter may convert an infeasible, unbounded, limited, or failed solve into
`optimal`, substitute a missing score with zero or one, or silently drop a DMU.

## 8. Reproducibility metadata

Every result records:

- implementation and adapter names and versions;
- language and runtime version;
- operating system and machine architecture;
- solver name and, when available, version;
- source revision;
- UTC start and finish timestamps;
- case and protocol versions;
- every score normalization performed by the adapter; and
- one execution record per baseline or metamorphic variant.

For publication-grade runs, pin the package environment, store a lockfile or
container digest, record CPU/solver configuration, and preserve raw logs.
Timing comparisons require a separate warm-up and repetition protocol; this
conformance suite intentionally makes no performance claim.

## 9. Adapter execution protocol

For each package and pinned environment:

1. validate benchmark assets;
2. translate the declared case without altering defaults implicitly;
3. run baseline and declared transformations;
4. emit strict JSON satisfying `schemas/result.schema.json`;
5. validate the result against oracle and relations;
6. retain logs and environment manifest;
7. have a second researcher review the mapping for publication-grade evidence.

Asset and result validation uses the declared JSON Schema Draft 2020-12
documents, including `additionalProperties` constraints. A top-level
`success` outcome is necessary but not sufficient: every required baseline and
metamorphic execution must also report `optimal` for every DMU.

Implementations should be run in a randomized or rotated order when runtime or
resource comparisons are later added. Conformance itself is insensitive to
execution order.

## 10. Interpretation and reporting

Report counts separately for:

- supported and successfully executed cases;
- oracle agreements;
- metamorphic agreements;
- explicit unsupported models;
- solver/model failures; and
- unresolved convention mismatches.

Do not compute a single “quality score” that conflates unsupported features,
incorrect translations, numerical mismatches, and solver failure. Publish the
case files, adapter source, exact versions, raw result JSON, and adjudication
notes.

## 11. Threats to validity

- Small analytical cases do not establish numerical reliability at scale.
- Agreement among packages can reflect a shared convention or shared error.
- Solver tolerances and presolve can affect degenerate solutions.
- Package defaults may not map cleanly to the protocol.
- The first suite does not test slacks, targets, economic models, FDH,
  productivity, network DEA, statistical inference, or performance.
- The five-DMU fixture's original external bibliographic provenance is not yet
  established.

These limitations must remain visible in any white paper, dashboard, or
publication derived from the suite.

## 12. Extension gates

A new model family enters the benchmark only after it has:

1. explicit mathematical and reporting conventions;
2. redistributable input data;
3. at least one analytical or primary-publication oracle;
4. a package-neutral result-field definition;
5. applicable feasibility and identity checks;
6. declared metamorphic relations and tolerance policy; and
7. at least one independently reviewed adapter mapping.

## 13. Licensing, clean-room boundaries, and citation

Benchmark source files are MIT-licensed with the parent repository. Do not copy
GPL or proprietary implementation source into adapters. Public output may be
used as independently generated evidence only when package terms and dataset
licenses permit redistribution. Restricted packages or datasets require legal
and research-governance review.

Method and solver citations are listed in
`citations/REFERENCES.md`. Each future external adapter must add its package
citation and license URL. Publication artifacts should cite both FrontierGuard
and the compared packages rather than presenting their outputs without credit.
