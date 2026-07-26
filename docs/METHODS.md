# Implemented method contracts

## Radial envelopment DEA

Let \(X\in\mathbb{R}^{n\times m}_+\) and
\(Y\in\mathbb{R}^{n\times s}_+\) define the reference technology. For evaluated
DMU \(o\), the input-oriented model is:

\[
\min_{\theta,\lambda}\ \theta
\]

subject to:

\[
X^\top\lambda\leq\theta x_o,\qquad
Y^\top\lambda\geq y_o,\qquad
\lambda\geq0.
\]

The output-oriented model maximizes \(\phi\):

\[
X^\top\lambda\leq x_o,\qquad
Y^\top\lambda\geq\phi y_o,\qquad
\lambda\geq0.
\]

FrontierGuard reports:

- input technical efficiency \(=\theta\);
- output radial factor \(=\phi\); and
- output technical efficiency \(=1/\phi\).

Returns-to-scale constraints are:

| Technology | Constraint |
|---|---|
| CRS | none |
| VRS | \(\mathbf{1}^\top\lambda=1\) |
| NIRS | \(\mathbf{1}^\top\lambda\leq1\) |
| NDRS | \(\mathbf{1}^\top\lambda\geq1\) |

### Slack phase

After fixing the optimal radial factor, the optional second phase maximizes
input and output slacks subject to the projection equalities. The default
objective normalizes each slack by the largest observed magnitude for that
variable; `max_sum` uses raw units, and `none` retains the phase-one projection.
The radial efficiency never changes in phase two.

Targets equal \(X^\top\lambda\) and \(Y^\top\lambda\). Peer weights are the
positive components of \(\lambda\).

## Weighted additive DEA

For declared non-negative slack weights \(a\) and \(b\), the weighted additive
model maximizes:

\[
a^\top s^- + b^\top s^+
\]

subject to:

\[
X^\top\lambda+s^-=x_o,\qquad
Y^\top\lambda-s^+=y_o,
\]

the selected returns-to-scale constraint, and non-negative
\(\lambda,s^-,s^+\). The result is an inefficiency measure with efficient value
zero; it is not silently converted into a radial efficiency.

`unit_normalized` weights each slack by the inverse joint column maximum and is
invariant to a positive change of units. `raw` requires explicit weights and
retains their units.

## Directional-distance DEA

Given non-negative input-contraction direction \(g_x\) and output-expansion
direction \(g_y\), FrontierGuard maximizes \(\beta\) subject to:

\[
X^\top\lambda\leq x_o-\beta g_x,\qquad
Y^\top\lambda\geq y_o+\beta g_y.
\]

At least one directional component must be positive for each DMU. Directions
can be shared vectors or DMU-specific matrices. FrontierGuard reports
\(\beta\), the declared directions, reference intensities, residual slacks, and
targets. Efficient value is zero. The directional factor is a free variable:
it may be negative when an observation lies beyond or outside a separately
declared reference technology along the chosen direction. FrontierGuard
preserves that signed value rather than clipping it into an improvement-only
measure.

## Non-oriented slack-based measure

For strictly positive evaluated inputs and outputs, Tone's non-oriented SBM is:

\[
\rho =
\frac{1-\frac{1}{m}\sum_i s_i^-/x_{io}}
     {1+\frac{1}{s}\sum_r s_r^+/y_{ro}},
\]

subject to:

\[
X^\top\lambda+s^-=x_o,\qquad
Y^\top\lambda-s^+=y_o,
\]

the selected returns-to-scale constraint, and non-negative variables.
FrontierGuard solves the exact Charnes–Cooper linear transformation and
recovers \(\lambda,s^-,s^+\) in the original scale. It reports both \(\rho\)
and the transformation variable \(t\), then independently reconstructs the
fractional score and projection identities.

Canonical SBM denominators cannot be zero. FrontierGuard therefore requires
strictly positive evaluated input and output components and never inserts an
epsilon or translates data. Reference coefficients may be zero because they
are not denominators. `exclude_self=True` changes only the reference set and
must not be interpreted as super-SBM.

## Serial relational network DEA

For a \(K\)-stage serial CRS process with external inputs \(x\), links
\(z^{(1)},\ldots,z^{(K-1)}\), and final outputs \(y\), FrontierGuard uses one
shared multiplier system.

For evaluated DMU \(o\), normalize:

\[
v^\top x_o=1.
\]

Maximize:

\[
u^\top y_o
\]

subject to every reference DMU \(j\):

\[
w_1^\top z^{(1)}_j-v^\top x_j\leq0,
\]

\[
w_k^\top z^{(k)}_j-w_{k-1}^\top z^{(k-1)}_j\leq0
\quad(k=2,\ldots,K-1),
\]

\[
u^\top y_j-w_{K-1}^\top z^{(K-1)}_j\leq0,
\]

with non-negative weights.

Stage efficiencies are adjacent weighted-flow ratios. Their telescoping product
equals overall efficiency whenever every selected stage denominator is
positive:

\[
E_{\text{overall}}=\prod_{k=1}^{K}E_k.
\]

Columns are internally normalized by their positive means for numerical
stability; returned multiplier weights are transformed back to original units.
DEA efficiency scores are invariant to this positive unit change.

## General acyclic relational network DEA

`NetworkData` represents an arbitrary directed acyclic graph. Each stage can
have external inputs and outputs; each `NetworkLink` has one producer, one
consumer, and a shared multiplier.

For stage \(k\) and reference DMU \(j\), define virtual input and output:

\[
D_{kj}=v_k^\top x^k_j+
\sum_{\ell\in In(k)}w_\ell^\top z_{\ell j},
\]

\[
N_{kj}=u_k^\top y^k_j+
\sum_{\ell\in Out(k)}w_\ell^\top z_{\ell j}.
\]

The centralized CRS program normalizes total external virtual input for the
evaluated DMU to one, maximizes total external virtual output, and enforces:

\[
N_{kj}\leq D_{kj}\quad\text{for every stage and reference DMU}.
\]

Internal link values cancel when stage constraints are summed. Consequently,
the general-DAG identity is:

\[
1-E_o=\sum_k(D_{ko}-N_{ko}).
\]

FrontierGuard reports both selected stage ratios \(N_{ko}/D_{ko}\) and the
stage gaps \(D_{ko}-N_{ko}\). A stage ratio is undefined when its selected
denominator is zero; the overall efficiency and gap decomposition remain
valid. Stage weights and stage ratios may be non-unique under alternate
optima, so this method is currently experimental.

## Free Disposal Hull

FDH drops the convexity assumption and compares each evaluated DMU with one
observed reference peer plus free disposability. FrontierGuard enumerates
feasible peers exactly rather than approximating FDH with a convex model.

For input orientation, a reference \(j\) is feasible when \(y_j\geq y_o\), and
its radial factor is:

\[
\theta_j=\max_i\frac{x_{ji}}{x_{io}}.
\]

The minimum feasible \(\theta_j\) is selected. Output orientation analogously
selects the maximum:

\[
\phi_j=\min_r\frac{y_{jr}}{y_{or}}
\]

among peers with \(x_j\leq x_o\). Output technical efficiency is \(1/\phi\).
Zero evaluated components receive explicit feasibility handling; division by
zero is never silently performed. FDH+, order-\(m\), and order-\(\alpha\) are
separate future methods.

## Scale and super-efficiency diagnostics

Scale efficiency uses FrontierGuard's common normalized technical-efficiency
convention:

\[
SE_o=\frac{E_o^{CRS}}{E_o^{VRS}}.
\]

If CRS and VRS scores coincide, the DMU is classified as constant returns. If
not, equality of NIRS and VRS scores identifies decreasing returns; otherwise
the classification is increasing returns.

Radial super-efficiency evaluates each DMU against a reference set that
excludes itself. Input scores above one are retained. For output orientation,
the raw factor remains \(\phi\) and the larger-is-better super-efficiency score
is \(1/\phi\). VRS and restricted-RTS super-efficiency can be infeasible;
callers choose fail-fast or explicit recorded missing results.

## CRS multiplier and cross-efficiency DEA

The input-oriented CRS multiplier model normalizes the evaluated DMU's virtual
input and maximizes its virtual output:

\[
\max_{u,v}\ u^\top y_o
\]

subject to:

\[
v^\top x_o=1,\qquad
u^\top y_j-v^\top x_j\leq0\quad\text{for every reference DMU }j,
\qquad u,v\geq0.
\]

The output-oriented dual convention normalizes \(u^\top y_o=1\), minimizes
\(v^\top x_o\), reports that objective as the expansion factor \(\phi\), and
reports technical efficiency as \(1/\phi\).

For each solver-selected optimal multiplier system \((u_o,v_o)\), the
cross-efficiency assigned by appraiser \(o\) to evaluated DMU \(j\) is:

\[
E_{oj}=\frac{u_o^\top y_j}{v_o^\top x_j}.
\]

A zero denominator makes that particular cross score undefined; strict JSON
exports use `null`, never a non-standard `NaN` token. The diagonal is audited
against the model's own technical-efficiency score.

Type-I assurance regions can impose:

\[
L\leq\frac{w_p}{w_q}\leq U
\]

on two input weights or two output weights. FrontierGuard compiles these
restrictions without division and interprets their bounds in the original data
units, even though it scales columns internally for numerical stability.
Assurance regions are model assumptions and can change efficiency scores.

Multiplier solutions need not be unique. The current cross-efficiency matrix
therefore records one solver-selected optimum and must not be interpreted as a
unique ranking. Aggressive, benevolent, and other secondary-objective
cross-efficiency formulations are separate future methods. VRS multiplier
cross-efficiency is also excluded because transporting a free intercept between
DMUs requires an explicit convention rather than an implicit extension of the
CRS ratio model.

## Cost and revenue efficiency

Given observed non-negative input prices \(w_o\), cost efficiency solves:

\[
C_o^*=\min_{\lambda}\ w_o^\top X^\top\lambda
\]

subject to \(Y^\top\lambda\geq y_o\), the selected returns-to-scale
constraint, and \(\lambda\geq0\). FrontierGuard reports:

\[
CE_o=\frac{C_o^*}{w_o^\top x_o},\qquad
AE_o=\frac{CE_o}{TE_o^{input}}.
\]

Given observed non-negative output prices \(p_o\), revenue efficiency solves:

\[
R_o^*=\max_{\lambda}\ p_o^\top Y^\top\lambda
\]

subject to \(X^\top\lambda\leq x_o\), the selected returns-to-scale
constraint, and \(\lambda\geq0\). FrontierGuard reports:

\[
RE_o=\frac{p_o^\top y_o}{R_o^*},\qquad
AE_o=\frac{RE_o}{TE_o^{output}}.
\]

Prices may be one common factor-price vector or a DMU-specific price matrix.
Observed cost or revenue must be positive so each efficiency ratio is defined.
The package records a separate price fingerprint, verifies the economic,
technical, and allocative identity, and returns price-optimal targets and
reference intensities. Profit efficiency is deliberately not inferred from
these ratios because zero or negative observed or optimal profits require a
different normalization and boundedness contract.

## Adjacent-period Malmquist productivity

For an observation from period \(a\) evaluated against period \(b\)'s
technology, FrontierGuard defines \(e_{a|b}\) as normalized radial efficiency:
the input contraction factor \(\theta\), or the reciprocal output expansion
factor \(1/\phi\). Own-period frontier observations equal one; a cross-period
term may exceed one when the observation lies beyond the reference technology.

For adjacent periods \(t\) and \(t+1\), the four-distance index is:

\[
M =
\left[
\frac{e_{t+1|t}}{e_{t|t}}
\times
\frac{e_{t+1|t+1}}{e_{t|t+1}}
\right]^{1/2}.
\]

Its efficiency-change and technical-change components are:

\[
EC=\frac{e_{t+1|t+1}}{e_{t|t}},
\]

\[
TC=
\left[
\frac{e_{t+1|t}}{e_{t+1|t+1}}
\times
\frac{e_{t|t}}{e_{t|t+1}}
\right]^{1/2},
\qquad M=EC\times TC.
\]

The implementation computes ratios in log space, audits all four underlying
radial solves, and independently verifies the decomposition. A value above one
denotes productivity growth under this convention. CRS is the standard
scale-inclusive total-factor productivity specification. The explicit VRS
option performs the same adjacent four-distance calculation against VRS
technologies and must not be interpreted as that CRS scale-inclusive index.
Global Malmquist and environmental Malmquist–Luenberger indexes are different
methods and are not aliases of this estimator.

## Current boundary

- Data must be non-negative and finite.
- Missing data are never imputed.
- The radial models are convex envelopment models.
- Relational network models are CRS. The stable foundation is serial; the
  arbitrary acyclic-graph engine is experimental.
- Canonical non-negative multipliers are the default. If a selected stage has
  zero virtual input, that stage score is reported as undefined while the
  overall score remains valid.
- A positive `weight_floor` is an explicit weight-restriction variant. It can
  resolve zero denominators but changes the mathematical model and is never
  enabled silently.
- Efficiency is descriptive and relative to the declared reference technology.
- No causal or probabilistic interpretation follows from a score alone.
