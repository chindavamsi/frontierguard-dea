# Feature and parity matrix

This is the release contract, not a wish list. “Provisional” means implemented,
audited, and covered by local analytical or published numerical fixtures, but
the full cross-language release gate is not yet complete. “Experimental” means
the API or formulation still needs broader review. “Gate” means theory or
oracle work must precede implementation.

| Capability | Status | Primary comparison/oracle |
|---|---|---|
| Radial input/output DEA | provisional | analytical, five-DMU, and eleven-DMU fixtures |
| CRS, VRS, NIRS, NDRS | provisional | analytical RTS fixtures |
| Slacks, targets, peers | provisional | primal reconstruction tests |
| Separate reference technology | provisional | external-reference fixtures |
| Serial multi-stage relational DEA | provisional | internal-mismatch analytical oracle |
| General DAG / parallel network DEA | experimental | virtual-gap identity and branched fixtures |
| Network SBM / dynamic network SBM | gate | Tone–Tsutsui models |
| Weighted additive DEA | provisional | analytical slack/projection oracle |
| RAM / BAM | planned | DataEnvelopmentAnalysis.jl, deaR |
| Non-oriented SBM | provisional | analytical slack-ratio and solver-duality fixtures |
| Super-SBM | gate | Tone super-SBM examples |
| Russell / ERG / Hölder families | planned | DataEnvelopmentAnalysis.jl |
| Directional distance functions | provisional | analytical directional oracle |
| Undesirable outputs / environmental DEA | planned | DataEnvelopmentAnalysis.jl |
| Radial super-efficiency | provisional | analytical and infeasible-VRS fixtures |
| CRS multiplier and selected cross-efficiency | provisional | radial duality and five-DMU fixtures |
| Assurance-region weight ratios | provisional | constrained multiplier fixtures |
| Aggressive/benevolent cross-efficiency variants | planned | deaR Doyle–Green data |
| Non-discretionary variables | planned | pyDEA |
| Categorical and weakly disposable variables | gate | primary methodology |
| FDH | provisional | published eleven-DMU fixture |
| FDH+, order-m, order-alpha | planned | Benchmarking, rcDEA, frontiles |
| Robust perturbation and rank/peer stability | planned | robustDEA |
| Cost and revenue efficiency | provisional | published cost and analytical revenue fixtures |
| Profit and profitability efficiency | gate | DataEnvelopmentAnalysis.jl |
| Scale efficiency and RTS classification | provisional | IRS/CRS/DRS analytical fixture |
| Adjacent-period Malmquist (CRS/VRS) | provisional | published four-distance fixture |
| Global Malmquist and Malmquist–Luenberger | planned | deaR, Julia |
| Aligned adjacent-period panel analysis | provisional | transition/identity fixtures |
| Window DEA | planned | productivity packages |
| Bootstrap bias correction and intervals | planned | deaR, Julia |
| Simar–Wilson second-stage double bootstrap | gate | published algorithms |
| Fuzzy DEA | gate | deaR |
| Inverse DEA / allocation / closest targets | gate | primary methods |
| Sparse and parallel large-n execution | planned | Julia big-data radial |
| Markdown/CSV/JSON evidence bundles | provisional | checksum and mismatch tests |
| Plots and HTML reports | planned | deaR |
| SciPy-HiGHS and native HiGHS solver backends | provisional | backend conformance suite |

The comparison inventory is being updated against 2026 package documentation.
No future method will reuse a familiar name for a materially different
mathematical program.

See [ECOSYSTEM.md](ECOSYSTEM.md) for the dated R/Python/Julia inventory,
licensing boundary, and numerical-oracle register.
