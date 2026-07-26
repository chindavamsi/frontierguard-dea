# Benchmark references

These references establish model definitions and solver context. They do not
by themselves prove the provenance of every numerical fixture.

## Classical radial DEA

Charnes, A., Cooper, W. W., & Rhodes, E. (1978). Measuring the efficiency of
decision making units. *European Journal of Operational Research, 2*(6),
429–444. <https://doi.org/10.1016/0377-2217(78)90138-8>

Banker, R. D., Charnes, A., & Cooper, W. W. (1984). Some models for estimating
technical and scale inefficiencies in data envelopment analysis. *Management
Science, 30*(9), 1078–1092. <https://doi.org/10.1287/mnsc.30.9.1078>

## Solver

Huangfu, Q., & Hall, J. A. J. (2018). Parallelizing the dual revised simplex
method. *Mathematical Programming Computation, 10*, 119–142.
<https://doi.org/10.1007/s12532-017-0130-5>

## Fixture provenance note

The three-DMU case is a hand-derived construction documented directly in its
case file.

The five-DMU data are already present in FrontierGuard's `load_five_dmu()`
dataset and radial regression tests. Its expected scores are exact rational
values. Until an original primary source is recovered and verified, the
benchmark labels it an analytical repository fixture and does not describe it
as a primary-publication oracle.

Future cross-language evidence must cite the exact version and canonical
citation of every compared package, as well as its solver where known.
