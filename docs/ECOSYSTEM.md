# DEA ecosystem and compatibility target

Snapshot date: 2026-07-26

FrontierGuard's goal is not API mimicry. It is a clean-room, permissively
licensed implementation of published DEA methods with a consistent data,
solver, audit, and result contract. “Parity” means the same mathematical
capability has been implemented and independently validated, not that code,
fixtures, or undocumented conventions have been copied.

No maintained package currently combines broad deterministic DEA,
network/dynamic models, productivity and economic models, statistical
inference, modern solver diagnostics, and one stable result schema.

## Capability landscape

Legend: **yes** = documented first-class support; **partial** = a narrower
variant or user formulation is required; **no** = not documented in the public
API.

| Capability | Benchmarking 0.33 | deaR 1.5.4 | DJL 3.9 | DataEnvelopmentAnalysis.jl 0.9.2 | Pyfrontier 1.1.1 | pyDEA 1.6 | pyStoNED 0.7.5 |
|---|---|---|---|---|---|---|---|
| Radial input/output; CRS/VRS | yes | yes | yes | yes | yes | yes | yes |
| NIRS/NDRS and generalized RTS | yes | yes | partial | partial | yes | partial | partial |
| Slacks, peers, targets, multipliers | yes | yes | yes | yes | partial | yes | partial |
| Additive/non-radial/SBM families | partial | yes | yes | yes | partial | no | partial |
| Directional/environmental models | yes | yes | yes | yes | no | partial | yes |
| Super-efficiency | yes | yes | yes | no | yes | yes | no |
| Cross-efficiency | no | yes | no | no | yes | no | no |
| Assurance regions/weight restrictions | yes | partial | no | partial | yes | yes | no |
| Non-discretionary/categorical factors | partial | yes | partial | partial | partial | yes | no |
| FDH/partial frontiers | yes | partial | partial | yes | no | no | yes |
| Cost/revenue/profit | yes | yes | no | yes | no | no | no |
| Malmquist/productivity | yes | yes | yes | yes | no | no | no |
| Bootstrap/RTS inference | yes | partial | no | yes | no | no | no |
| Network production stages | no | no | two-stage | no | no | no | no |
| Dynamic carry-overs | no | no | yes | no | no | no | no |
| Stochastic frontier/CNLS | no | no | no | no | no | no | yes |

Important terminology:

1. **Lexicographic/two-phase solving** means optimizing a radial score and then
   its slacks.
2. **Network/multi-stage DEA** means connected production stages with explicit
   intermediate flows.
3. **Statistical two-stage analysis** means DEA followed by environmental
   regression or a method such as Simar–Wilson.

FrontierGuard uses separate APIs for all three.

## Primary comparison sources

- [R Benchmarking](https://cran.r-project.org/package=Benchmarking)
- [R deaR](https://cran.r-project.org/package=deaR)
- [R DJL](https://cran.r-project.org/package=DJL)
- [DataEnvelopmentAnalysis.jl](https://javierbarbero.github.io/DataEnvelopmentAnalysis.jl/stable/)
- [Pyfrontier](https://pypi.org/project/Pyfrontier/)
- [pyDEA](https://pypi.org/project/pyDEA/)
- [pyStoNED](https://github.com/ds2010/pyStoNED)
- [HiGHS](https://github.com/ERGO-Code/HiGHS)

Specialist comparison packages include
[rDEA](https://cran.r-project.org/package=rDEA) for Simar–Wilson inference,
[rcDEA](https://cran.r-project.org/package=rcDEA) for conditional and order-m
frontiers, [SdeaR](https://cran.r-project.org/package=SdeaR) for stochastic
DEA, [metafrontier](https://cran.r-project.org/package=metafrontier),
[productivity](https://cran.r-project.org/package=productivity), and
[robustDEA](https://github.com/alabijak/robustDEA).

## Clean-room and licensing boundary

- Implement equations from primary scholarly publications and independently
  derived optimization programs.
- Use package documentation to identify capability and convention differences.
- Recreate numerical expectations from citable public data or primary papers.
- Do not copy GPL implementation code into FrontierGuard's MIT-licensed code.
- Do not vendor opaque outputs merely because another package generated them.
- FEAR is explicitly excluded as an implementation reference or oracle. Its
  [academic-use license](https://pwwilson.com/Software/FEAR/license/LICENSE.html)
  restricts redistribution, reverse engineering, and development of
  functionally equivalent technology. Any future use requires legal review.

## Oracle register

Every implemented model must have at least one analytical or published oracle
and one independent formulation or cross-language check.

| Model family | Initial oracle |
|---|---|
| CCR/BCC radial | hand-worked rational fixtures; Charnes–Cooper–Rhodes school data |
| Slacks and projections | five-DMU Julia/pyDEA parity fixture |
| SBM | Tone (2001) published table and `deaR::Tone2001` source data |
| Undesirable outputs | published Hua–Bian and Tone examples |
| Cross-efficiency | Doyle–Green (1994) example |
| Cost/revenue/profit | Coelli et al. published example and accounting identity |
| Malmquist | published panel example; verify `M = EC × TC` |
| Serial network DEA | analytical internal-mismatch fixture; published Kao–Hwang data |
| Dynamic/network SBM | Tone–Tsutsui primary-paper examples before stabilization |
| Bootstrap/inference | deterministic simulation with fixed RNG plus published algorithm |
| Metafrontier | truth-bearing simulated group technologies |

Alternate DEA optima mean intensities and multipliers need not be identical
across solvers. Golden tests compare scores, targets, feasibility, duality, and
model identities unless uniqueness has been established.

## FrontierGuard's defensible advantage

1. A graph-defined, first-class network technology, with dynamic production
   explicitly gated on future published oracles.
2. One immutable, typed result schema with explicit score conventions.
3. Independent post-solve mathematical auditing and solver provenance.
4. Direct HiGHS-oriented sparse backend architecture.
5. A published path from analytical fixtures to independent cross-language
   golden validation before methods can become stable.
6. Common compatibility gates for future environmental, robust, dynamic, and
   inferential modules rather than premature API aliases.
