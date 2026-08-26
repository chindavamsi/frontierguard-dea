# FrontierGuard DEA

**One rigorous Python toolkit for efficiency, productivity, and network
analysis.**

FrontierGuard DEA is a new open-source package being built to unify the
fragmented DEA workflow behind one typed, inspectable API. Multi-stage and
network DEA are architectural primitives, not afterthoughts.

The project is an alpha research preview. The implemented foundation currently
includes:

- radial input- and output-oriented envelopment models;
- CRS, VRS, NIRS, and NDRS technologies;
- an optional second phase for slacks, targets, and peers;
- weighted additive, non-oriented SBM, and explicit directional-distance models;
- separate evaluation and reference technologies;
- radial super-efficiency with explicit infeasibility handling;
- scale efficiency and IRS/CRS/DRS diagnosis;
- exact input- and output-oriented Free Disposal Hull models;
- CRS multiplier and solver-selected cross-efficiency models;
- type-I assurance-region restrictions in original data units;
- cost, revenue, technical, and allocative efficiency with common or
  DMU-specific prices;
- adjacent-period Malmquist productivity panels with audited efficiency and
  technical-change decomposition;
- a general CRS relational model for any number of serial stages; and
- an experimental graph-defined relational model for acyclic stage networks;
- immutable results with solver status, model conventions, data fingerprints,
  and independent post-solve audit residuals.

SciPy-HiGHS is the zero-configuration solver. Install the `highs` extra to use
the direct native HiGHS adapter with duals, reduced costs, sparse matrices, and
deterministic solver controls.

The larger roadmap includes RAM/BAM and Russell families, environmental
models, aggressive/benevolent cross-efficiency, super-SBM, robust partial
frontiers, window analysis, global and Malmquist–Luenberger productivity,
bootstrap inference, second-stage analysis, dynamic/network SBM, richer
reporting, and cross-language validation.

## Installation

After version `0.1.0a1` is published to PyPI, install it with:

```bash
python -m pip install --pre "frontierguard-dea==0.1.0a1"
```

The optional native HiGHS adapter is available through the `highs` extra:

```bash
python -m pip install --pre "frontierguard-dea[highs]==0.1.0a1"
```

Contributors can clone the repository and run `uv sync --all-groups`.
Until the PyPI publication is confirmed, install the reviewed source branch or
tag directly from GitHub rather than assuming the index entry exists.

## Quick start

```python
import pandas as pd
import frontierguard as fg

frame = pd.DataFrame(
    {
        "dmu": ["A", "B", "C"],
        "staff": [1.0, 2.0, 3.0],
        "cases": [1.0, 1.0, 2.0],
    }
)

data = fg.DEAData.from_frame(
    frame,
    inputs=["staff"],
    outputs=["cases"],
    dmu="dmu",
)

result = fg.RadialDEA(orientation="input", rts="vrs").fit(data)
print(result.to_frame())
print(result.peers("B"))
```

## Multi-stage DEA

```python
network = fg.SerialNetworkData(
    inputs=[[1.0], [2.0], [2.0]],
    links=([[1.0], [1.0], [2.0]],),
    outputs=[[1.0], [1.0], [1.5]],
    dmu_names=("A", "B", "C"),
    stage_names=("operations", "delivery"),
)

result = fg.SerialNetworkDEA().fit(network)
print(result.to_frame())
```

The serial relational model reports overall and stage efficiencies while
verifying that their product closes to the overall score.

For branched or general acyclic production systems, construct
`NetworkStage`, `NetworkLink`, and `NetworkData` objects and fit
`RelationalNetworkDEA`. The general engine reports stage-gap contributions and
verifies that they sum to overall inefficiency.

## Trust policy

FrontierGuard will not label a method “supported” until it has:

1. an explicit mathematical contract and conventions;
2. hand-calculated or published golden cases;
3. primal feasibility and identity tests;
4. scale, permutation, and unit-invariance tests where theoretically valid;
5. a cross-language oracle against an established implementation; and
6. documented failure modes.

See [the feature matrix](docs/FEATURE_MATRIX.md),
[ecosystem audit](docs/ECOSYSTEM.md),
[architecture](docs/ARCHITECTURE.md), [methods](docs/METHODS.md), and
[validation strategy](docs/VALIDATION.md). The
[executable examples](examples/README.md) cover classical, multi-stage, and
economic workflows.

## Community and project status

FrontierGuard DEA is licensed under the [MIT License](LICENSE) and maintained
by [Vamsi Chinda](https://github.com/chindavamsi). Before contributing, read
the [contribution guide](CONTRIBUTING.md), [governance](GOVERNANCE.md), and
[Code of Conduct](CODE_OF_CONDUCT.md). Report security concerns through the
private process in [SECURITY.md](SECURITY.md).

Research users can cite the project using [CITATION.cff](CITATION.cff). Release
history and compatibility-impacting changes are recorded in
[CHANGELOG.md](CHANGELOG.md).

The publication programme includes a versioned
[technical white paper](paper/whitepaper/README.md)
([PDF](output/pdf/frontierguard-dea-white-paper-v0.1.pdf)) and an executable
[cross-language conformance benchmark](benchmarks/README.md). These materials
describe current evidence and limitations; they do not imply peer review or
complete cross-language validation.
