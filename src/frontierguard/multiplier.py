"""CRS multiplier DEA, assurance regions, and cross-efficiency appraisal.

The module deliberately implements only constant returns to scale.  Variable
returns multiplier models require a free intercept, and using one appraiser's
intercept for another DMU's cross score is not a neutral extension of the CCR
ratio model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from frontierguard._data import DEAData
from frontierguard._enums import (
    Orientation,
    ReturnsToScale,
    SolverStatus,
    coerce_enum,
)
from frontierguard._errors import (
    DataValidationError,
    ModelValidationError,
    NumericalError,
    SolverError,
)
from frontierguard._solver import LinearProgram, LinearSolver, ScipyHighsSolver


class WeightKind(str, Enum):
    """Multiplier family targeted by an assurance-region restriction."""

    INPUT = "input"
    OUTPUT = "output"


WeightIdentifier = str | int


@dataclass(frozen=True, slots=True)
class AssuranceRegion:
    """A homogeneous type-I ratio restriction on two input or output weights.

    ``lower <= numerator_weight / denominator_weight <= upper`` is compiled
    without division, so the restriction remains well-defined when both
    multipliers are zero.  Bounds apply to multipliers in the data's original
    units.  They are therefore unit-dependent economic assumptions, as in the
    classical assurance-region model.
    """

    kind: WeightKind | str
    numerator: WeightIdentifier
    denominator: WeightIdentifier
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        resolved_kind = coerce_enum(self.kind, WeightKind, label="kind")
        object.__setattr__(self, "kind", resolved_kind)
        if self.lower is None and self.upper is None:
            raise ModelValidationError(
                "An assurance region requires a lower bound, an upper bound, or both."
            )
        for label, value in (("lower", self.lower), ("upper", self.upper)):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ModelValidationError(
                    f"Assurance-region {label} must be finite and non-negative."
                )
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower > self.upper
        ):
            raise ModelValidationError("Assurance-region lower cannot exceed upper.")
        if (
            type(self.numerator) is type(self.denominator)
            and self.numerator == self.denominator
        ):
            raise ModelValidationError(
                "Assurance-region numerator and denominator must differ."
            )
        for label, identifier in (
            ("numerator", self.numerator),
            ("denominator", self.denominator),
        ):
            if not isinstance(identifier, (str, int)) or isinstance(identifier, bool):
                raise ModelValidationError(
                    f"Assurance-region {label} must be a name or zero-based index."
                )
            if isinstance(identifier, str) and not identifier.strip():
                raise ModelValidationError(
                    f"Assurance-region {label} name cannot be blank."
                )
            if isinstance(identifier, int) and identifier < 0:
                raise ModelValidationError(
                    f"Assurance-region {label} index cannot be negative."
                )

    def to_dict(self) -> dict[str, str | int | float | None]:
        """Return a strict JSON-compatible restriction."""
        return {
            "kind": cast(WeightKind, self.kind).value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass(frozen=True, slots=True)
class _ResolvedRestriction:
    specification: AssuranceRegion
    numerator: int
    denominator: int


def _readonly(values: ArrayLike) -> NDArray[np.float64]:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _nullable_vector(values: NDArray[np.float64]) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _nullable_matrix(
    values: NDArray[np.float64],
) -> list[list[float | None]]:
    return [_nullable_vector(row) for row in values]


@dataclass(frozen=True, slots=True, init=False)
class MultiplierResult:
    """Immutable selected multipliers and cross-efficiency scores."""

    dmu_names: tuple[str, ...]
    reference_names: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    orientation: Orientation
    returns_to_scale: ReturnsToScale
    radial_factor: NDArray[np.float64]
    technical_efficiency: NDArray[np.float64]
    input_weights: NDArray[np.float64]
    output_weights: NDArray[np.float64]
    cross_efficiency: NDArray[np.float64]
    statuses: tuple[SolverStatus, ...]
    solver: str
    weight_floor: float
    restrictions: tuple[AssuranceRegion, ...]
    data_fingerprint: str
    reference_fingerprint: str
    primal_residual: NDArray[np.float64]
    ratio_residual: NDArray[np.float64]
    restriction_residual: NDArray[np.float64]
    objective_residual: NDArray[np.float64]
    audit_tolerance: float

    def __init__(
        self,
        *,
        dmu_names: Sequence[str],
        reference_names: Sequence[str],
        input_names: Sequence[str],
        output_names: Sequence[str],
        orientation: Orientation,
        radial_factor: ArrayLike,
        technical_efficiency: ArrayLike,
        input_weights: ArrayLike,
        output_weights: ArrayLike,
        cross_efficiency: ArrayLike,
        statuses: Sequence[SolverStatus],
        solver: str,
        weight_floor: float,
        restrictions: Sequence[AssuranceRegion],
        data_fingerprint: str,
        reference_fingerprint: str,
        primal_residual: ArrayLike,
        ratio_residual: ArrayLike,
        restriction_residual: ArrayLike,
        objective_residual: ArrayLike,
        audit_tolerance: float,
    ) -> None:
        object.__setattr__(self, "dmu_names", tuple(dmu_names))
        object.__setattr__(self, "reference_names", tuple(reference_names))
        object.__setattr__(self, "input_names", tuple(input_names))
        object.__setattr__(self, "output_names", tuple(output_names))
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "returns_to_scale", ReturnsToScale.CRS)
        object.__setattr__(self, "radial_factor", _readonly(radial_factor))
        object.__setattr__(
            self, "technical_efficiency", _readonly(technical_efficiency)
        )
        object.__setattr__(self, "input_weights", _readonly(input_weights))
        object.__setattr__(self, "output_weights", _readonly(output_weights))
        object.__setattr__(self, "cross_efficiency", _readonly(cross_efficiency))
        object.__setattr__(self, "statuses", tuple(statuses))
        object.__setattr__(self, "solver", str(solver))
        object.__setattr__(self, "weight_floor", float(weight_floor))
        object.__setattr__(self, "restrictions", tuple(restrictions))
        object.__setattr__(self, "data_fingerprint", data_fingerprint)
        object.__setattr__(self, "reference_fingerprint", reference_fingerprint)
        object.__setattr__(self, "primal_residual", _readonly(primal_residual))
        object.__setattr__(self, "ratio_residual", _readonly(ratio_residual))
        object.__setattr__(
            self, "restriction_residual", _readonly(restriction_residual)
        )
        object.__setattr__(self, "objective_residual", _readonly(objective_residual))
        object.__setattr__(self, "audit_tolerance", float(audit_tolerance))

    @property
    def weight_floor_changes_model(self) -> bool:
        """Whether a noncanonical positive floor changed the feasible weight set."""
        return self.weight_floor > 0.0

    @property
    def max_audit_residual(self) -> float:
        """Largest residual across independent audit categories."""
        return float(
            max(
                np.max(self.primal_residual, initial=0.0),
                np.max(self.ratio_residual, initial=0.0),
                np.max(self.restriction_residual, initial=0.0),
                np.max(self.objective_residual, initial=0.0),
            )
        )

    def to_frame(self) -> pd.DataFrame:
        """Return primary radial and efficiency conventions by evaluated DMU."""
        return pd.DataFrame(
            {
                "dmu": self.dmu_names,
                "technical_efficiency": self.technical_efficiency,
                "radial_factor": self.radial_factor,
                "status": [status.value for status in self.statuses],
            }
        ).set_index("dmu")

    def cross_frame(self) -> pd.DataFrame:
        """Return appraisers by rows and evaluated peers by columns."""
        return pd.DataFrame(
            self.cross_efficiency,
            index=pd.Index(self.dmu_names, name="appraiser"),
            columns=pd.Index(self.dmu_names, name="evaluated"),
        )

    def selected_weights(
        self,
        dmu: str | int,
    ) -> dict[str, dict[str, float]]:
        """Return one DMU's selected input and output multipliers."""
        position = self._position(dmu)
        return {
            "inputs": {
                name: float(value)
                for name, value in zip(
                    self.input_names,
                    self.input_weights[position],
                    strict=True,
                )
            },
            "outputs": {
                name: float(value)
                for name, value in zip(
                    self.output_names,
                    self.output_weights[position],
                    strict=True,
                )
            },
        }

    def peer_average(
        self,
        *,
        exclude_self: bool = True,
        appraiser_weights: (
            Mapping[str, float] | Sequence[float] | NDArray[np.float64] | None
        ) = None,
        skip_undefined: bool = True,
    ) -> NDArray[np.float64]:
        """Average appraisals for each DMU with explicit missing-cell semantics.

        Appraiser weights are normalized independently for every evaluated
        column after optional self-exclusion and undefined-cell filtering.
        When ``skip_undefined`` is false, any positively weighted undefined
        appraisal makes that column's average undefined.
        """
        weights = self._appraiser_weights(appraiser_weights)
        averages = np.full(len(self.dmu_names), np.nan)
        for evaluated in range(len(self.dmu_names)):
            included = weights > 0.0
            if exclude_self:
                included[evaluated] = False
            defined = np.isfinite(self.cross_efficiency[:, evaluated])
            if not skip_undefined and np.any(included & ~defined):
                continue
            included &= defined
            total = float(weights[included].sum())
            if total > 0.0:
                averages[evaluated] = float(
                    weights[included]
                    @ self.cross_efficiency[included, evaluated]
                    / total
                )
        return _readonly(averages)

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible representation; undefined is null."""
        peer_all = self.peer_average(exclude_self=False)
        peer_excluding_self = self.peer_average(exclude_self=True)
        return {
            "schema_version": "0.1.0",
            "model": "crs_multiplier_cross_efficiency",
            "orientation": self.orientation.value,
            "returns_to_scale": self.returns_to_scale.value,
            "solver": self.solver,
            "weight_floor": self.weight_floor,
            "weight_floor_changes_model": self.weight_floor_changes_model,
            "restrictions": [
                restriction.to_dict() for restriction in self.restrictions
            ],
            "data_fingerprint": self.data_fingerprint,
            "reference_fingerprint": self.reference_fingerprint,
            "audit_tolerance": self.audit_tolerance,
            "max_audit_residual": self.max_audit_residual,
            "dmu_names": list(self.dmu_names),
            "reference_names": list(self.reference_names),
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "radial_factor": self.radial_factor.tolist(),
            "technical_efficiency": self.technical_efficiency.tolist(),
            "input_weights": self.input_weights.tolist(),
            "output_weights": self.output_weights.tolist(),
            "cross_efficiency": _nullable_matrix(self.cross_efficiency),
            "peer_average_all": _nullable_vector(peer_all),
            "peer_average_excluding_self": _nullable_vector(peer_excluding_self),
            "statuses": [status.value for status in self.statuses],
            "audit": {
                "primal_residual": self.primal_residual.tolist(),
                "ratio_residual": self.ratio_residual.tolist(),
                "restriction_residual": self.restriction_residual.tolist(),
                "objective_residual": self.objective_residual.tolist(),
            },
        }

    def _position(self, dmu: str | int) -> int:
        if isinstance(dmu, str):
            try:
                return self.dmu_names.index(dmu)
            except ValueError as exc:
                raise DataValidationError(f"Unknown DMU {dmu!r}.") from exc
        if isinstance(dmu, bool) or not isinstance(dmu, int):
            raise DataValidationError("DMU must be a name or zero-based index.")
        if not 0 <= dmu < len(self.dmu_names):
            raise DataValidationError(f"DMU position {dmu} is out of range.")
        return dmu

    def _appraiser_weights(
        self,
        values: Mapping[str, float] | Sequence[float] | NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        if values is None:
            weights = np.ones(len(self.dmu_names), dtype=float)
        elif isinstance(values, Mapping):
            unknown = set(values).difference(self.dmu_names)
            if unknown:
                names = ", ".join(sorted(str(name) for name in unknown))
                raise DataValidationError(f"Unknown appraisers: {names}.")
            try:
                weights = np.asarray(
                    [float(values.get(name, 0.0)) for name in self.dmu_names],
                    dtype=float,
                )
            except (TypeError, ValueError) as exc:
                raise DataValidationError(
                    "appraiser_weights must contain numeric values."
                ) from exc
        else:
            try:
                weights = np.asarray(values, dtype=float)
            except (TypeError, ValueError) as exc:
                raise DataValidationError(
                    "appraiser_weights must contain numeric values."
                ) from exc
        if weights.shape != (len(self.dmu_names),):
            raise DataValidationError(
                f"appraiser_weights must contain {len(self.dmu_names)} values."
            )
        if not np.isfinite(weights).all() or (weights < 0.0).any():
            raise DataValidationError(
                "appraiser_weights must be finite and non-negative."
            )
        if not np.any(weights > 0.0):
            raise DataValidationError(
                "appraiser_weights must contain at least one positive value."
            )
        return weights


@dataclass(frozen=True, slots=True)
class _Audit:
    primal: float
    ratio: float
    restriction: float
    objective: float

    @property
    def maximum(self) -> float:
        return float(
            np.max(
                np.asarray(
                    [
                        self.primal,
                        self.ratio,
                        self.restriction,
                        self.objective,
                    ],
                    dtype=float,
                )
            )
        )


class MultiplierDEA:
    """CCR multiplier DEA with optional assurance regions and cross appraisal.

    The canonical model permits zero input and output weights.  A positive
    ``weight_floor`` is available explicitly, but it changes the mathematical
    model.  The floor is applied to internally scaled weights so its meaning is
    stable under a change of measurement units.
    """

    def __init__(
        self,
        *,
        orientation: Orientation | str = Orientation.INPUT,
        rts: ReturnsToScale | str = ReturnsToScale.CRS,
        assurance_regions: Sequence[AssuranceRegion] = (),
        weight_floor: float = 0.0,
        solver: LinearSolver | None = None,
        tolerance: float = 1e-9,
    ) -> None:
        self.orientation = coerce_enum(orientation, Orientation, label="orientation")
        returns_to_scale = coerce_enum(rts, ReturnsToScale, label="rts")
        if returns_to_scale is not ReturnsToScale.CRS:
            raise ModelValidationError(
                "MultiplierDEA supports CRS only; VRS-family cross scores need "
                "an explicitly specified intercept convention."
            )
        self.returns_to_scale = returns_to_scale
        if not np.isfinite(weight_floor) or weight_floor < 0.0:
            raise ModelValidationError("weight_floor must be finite and non-negative.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ModelValidationError("tolerance must be positive and finite.")
        restrictions = tuple(assurance_regions)
        if not all(
            isinstance(restriction, AssuranceRegion) for restriction in restrictions
        ):
            raise ModelValidationError(
                "assurance_regions must contain AssuranceRegion instances."
            )
        self.assurance_regions = restrictions
        self.weight_floor = float(weight_floor)
        self.tolerance = float(tolerance)
        self.audit_tolerance = max(10.0 * self.tolerance, 1e-7)
        self.solver = ScipyHighsSolver() if solver is None else solver

    def fit(
        self,
        data: DEAData,
        *,
        reference: DEAData | None = None,
    ) -> MultiplierResult:
        """Select weights for every evaluated DMU against one CRS technology."""
        if not isinstance(data, DEAData):
            raise ModelValidationError("data must be a DEAData instance.")
        reference_data = data if reference is None else reference
        if not isinstance(reference_data, DEAData):
            raise ModelValidationError("reference must be a DEAData instance.")
        self._validate_technology(data, reference_data)
        restrictions = self._resolve_restrictions(data)

        input_scale = np.maximum(
            np.maximum(data.inputs.max(axis=0), reference_data.inputs.max(axis=0)),
            self.tolerance,
        )
        output_scale = np.maximum(
            np.maximum(
                data.outputs.max(axis=0),
                reference_data.outputs.max(axis=0),
            ),
            self.tolerance,
        )
        evaluated_inputs = data.inputs / input_scale
        evaluated_outputs = data.outputs / output_scale
        reference_inputs = reference_data.inputs / input_scale
        reference_outputs = reference_data.outputs / output_scale

        n_eval = data.n_dmus
        input_weights = np.empty((n_eval, data.n_inputs))
        output_weights = np.empty((n_eval, data.n_outputs))
        factors = np.empty(n_eval)
        efficiencies = np.empty(n_eval)
        primal_residuals = np.empty(n_eval)
        ratio_residuals = np.empty(n_eval)
        restriction_residuals = np.empty(n_eval)
        objective_residuals = np.empty(n_eval)
        statuses: list[SolverStatus] = []

        restriction_rows = self._restriction_rows(
            restrictions=restrictions,
            input_scale=input_scale,
            output_scale=output_scale,
            n_inputs=data.n_inputs,
            n_outputs=data.n_outputs,
        )
        for position, label in enumerate(data.dmu_names):
            (
                factor,
                efficiency,
                scaled_input_weights,
                scaled_output_weights,
                status,
                solver_objective,
            ) = self._fit_one(
                inputs=reference_inputs,
                outputs=reference_outputs,
                evaluated_inputs=evaluated_inputs[position],
                evaluated_outputs=evaluated_outputs[position],
                restriction_rows=restriction_rows,
                label=label,
            )
            raw_input_weights = scaled_input_weights / input_scale
            raw_output_weights = scaled_output_weights / output_scale
            audit = self._audit(
                reference_inputs=reference_inputs,
                reference_outputs=reference_outputs,
                evaluated_inputs=evaluated_inputs[position],
                evaluated_outputs=evaluated_outputs[position],
                scaled_input_weights=scaled_input_weights,
                scaled_output_weights=scaled_output_weights,
                raw_input_weights=raw_input_weights,
                raw_output_weights=raw_output_weights,
                restrictions=restrictions,
                radial_factor=factor,
                technical_efficiency=efficiency,
                solver_objective=solver_objective,
            )
            if not np.isfinite(audit.maximum) or audit.maximum > self.audit_tolerance:
                raise NumericalError(
                    f"Multiplier audit for {label!r} failed with residual "
                    f"{audit.maximum:.3e} above {self.audit_tolerance:.3e}."
                )
            input_weights[position] = raw_input_weights
            output_weights[position] = raw_output_weights
            factors[position] = factor
            efficiencies[position] = efficiency
            primal_residuals[position] = audit.primal
            ratio_residuals[position] = audit.ratio
            restriction_residuals[position] = audit.restriction
            objective_residuals[position] = audit.objective
            statuses.append(status)

        denominators = input_weights @ data.inputs.T
        numerators = output_weights @ data.outputs.T
        cross_efficiency = np.full((n_eval, n_eval), np.nan)
        defined = denominators > self.tolerance
        np.divide(
            numerators,
            denominators,
            out=cross_efficiency,
            where=defined,
        )
        tiny = np.isfinite(cross_efficiency) & (
            np.abs(cross_efficiency) <= self.tolerance
        )
        cross_efficiency[tiny] = 0.0

        self_scores = cross_efficiency[np.arange(n_eval), np.arange(n_eval)]
        if not np.isfinite(self_scores).all():
            raise NumericalError("A selected DMU has an undefined self-efficiency.")
        cross_residual = np.abs(self_scores - efficiencies) / np.maximum(
            1.0,
            np.maximum(np.abs(self_scores), np.abs(efficiencies)),
        )
        ratio_residuals = np.maximum(ratio_residuals, cross_residual)
        if np.max(ratio_residuals, initial=0.0) > self.audit_tolerance:
            raise NumericalError(
                "Cross-efficiency diagonal does not reproduce selected scores."
            )

        return MultiplierResult(
            dmu_names=data.dmu_names,
            reference_names=reference_data.dmu_names,
            input_names=data.input_names,
            output_names=data.output_names,
            orientation=self.orientation,
            radial_factor=factors,
            technical_efficiency=efficiencies,
            input_weights=input_weights,
            output_weights=output_weights,
            cross_efficiency=cross_efficiency,
            statuses=statuses,
            solver=self.solver.name,
            weight_floor=self.weight_floor,
            restrictions=self.assurance_regions,
            data_fingerprint=data.fingerprint,
            reference_fingerprint=reference_data.fingerprint,
            primal_residual=primal_residuals,
            ratio_residual=ratio_residuals,
            restriction_residual=restriction_residuals,
            objective_residual=objective_residuals,
            audit_tolerance=self.audit_tolerance,
        )

    def _fit_one(
        self,
        *,
        inputs: NDArray[np.float64],
        outputs: NDArray[np.float64],
        evaluated_inputs: NDArray[np.float64],
        evaluated_outputs: NDArray[np.float64],
        restriction_rows: tuple[NDArray[np.float64], ...],
        label: str,
    ) -> tuple[
        float,
        float,
        NDArray[np.float64],
        NDArray[np.float64],
        SolverStatus,
        float,
    ]:
        n_inputs = inputs.shape[1]
        n_outputs = outputs.shape[1]
        n_variables = n_inputs + n_outputs
        technology = np.hstack((-inputs, outputs))
        inequalities = (
            technology
            if not restriction_rows
            else np.vstack((technology, *restriction_rows))
        )
        inequality_bounds = np.zeros(inequalities.shape[0])
        equality = np.zeros((1, n_variables))
        objective = np.zeros(n_variables)
        if self.orientation is Orientation.INPUT:
            equality[0, :n_inputs] = evaluated_inputs
            objective[n_inputs:] = -evaluated_outputs
        else:
            equality[0, n_inputs:] = evaluated_outputs
            objective[:n_inputs] = evaluated_inputs

        solution = self.solver.solve(
            LinearProgram(
                objective=objective,
                inequalities=inequalities,
                inequality_bounds=inequality_bounds,
                equalities=equality,
                equality_bounds=[1.0],
                variable_bounds=[(self.weight_floor, None)] * n_variables,
            )
        )
        if solution.status is not SolverStatus.OPTIMAL:
            raise SolverError(
                f"Multiplier model for {label!r} failed with "
                f"{solution.status.value}: {solution.message}"
            )
        if (
            solution.variables.shape != (n_variables,)
            or not np.isfinite(solution.variables).all()
        ):
            raise NumericalError(
                f"Multiplier solver returned invalid variables for {label!r}."
            )
        if not np.isfinite(solution.objective):
            raise NumericalError(
                f"Multiplier solver returned an invalid objective for {label!r}."
            )
        weights = np.asarray(solution.variables, dtype=float).copy()
        if (weights < -self.audit_tolerance).any():
            raise NumericalError(
                f"Multiplier solver returned negative weights for {label!r}."
            )
        weights = np.maximum(weights, 0.0)
        input_weights = weights[:n_inputs]
        output_weights = weights[n_inputs:]
        if self.orientation is Orientation.INPUT:
            efficiency = float(output_weights @ evaluated_outputs)
            factor = efficiency
        else:
            factor = float(input_weights @ evaluated_inputs)
            if factor <= self.tolerance:
                raise NumericalError(
                    f"Output expansion factor for {label!r} is not positive."
                )
            efficiency = 1.0 / factor
        return (
            factor,
            efficiency,
            input_weights,
            output_weights,
            SolverStatus.OPTIMAL,
            float(solution.objective),
        )

    def _audit(
        self,
        *,
        reference_inputs: NDArray[np.float64],
        reference_outputs: NDArray[np.float64],
        evaluated_inputs: NDArray[np.float64],
        evaluated_outputs: NDArray[np.float64],
        scaled_input_weights: NDArray[np.float64],
        scaled_output_weights: NDArray[np.float64],
        raw_input_weights: NDArray[np.float64],
        raw_output_weights: NDArray[np.float64],
        restrictions: tuple[_ResolvedRestriction, ...],
        radial_factor: float,
        technical_efficiency: float,
        solver_objective: float,
    ) -> _Audit:
        virtual_inputs = reference_inputs @ scaled_input_weights
        virtual_outputs = reference_outputs @ scaled_output_weights
        technology_violation = np.maximum(virtual_outputs - virtual_inputs, 0.0)
        technology_scale = np.maximum(
            1.0, np.maximum(np.abs(virtual_inputs), np.abs(virtual_outputs))
        )
        primal = float(np.max(technology_violation / technology_scale, initial=0.0))
        lower_bound_violation = np.maximum(
            self.weight_floor
            - np.concatenate((scaled_input_weights, scaled_output_weights)),
            0.0,
        )
        primal = max(
            primal,
            float(
                np.max(
                    lower_bound_violation / max(1.0, self.weight_floor),
                    initial=0.0,
                )
            ),
        )
        evaluated_virtual_input = float(scaled_input_weights @ evaluated_inputs)
        evaluated_virtual_output = float(scaled_output_weights @ evaluated_outputs)
        normalization = (
            evaluated_virtual_input
            if self.orientation is Orientation.INPUT
            else evaluated_virtual_output
        )
        primal = max(primal, abs(normalization - 1.0) / max(1.0, abs(normalization)))

        if evaluated_virtual_input <= self.tolerance:
            ratio = float("inf")
        else:
            selected_ratio = evaluated_virtual_output / evaluated_virtual_input
            ratio = abs(selected_ratio - technical_efficiency) / max(
                1.0, abs(selected_ratio), abs(technical_efficiency)
            )

        restriction = 0.0
        for resolved in restrictions:
            weights = (
                raw_input_weights
                if resolved.specification.kind is WeightKind.INPUT
                else raw_output_weights
            )
            numerator = float(weights[resolved.numerator])
            denominator = float(weights[resolved.denominator])
            scale = max(1.0, abs(numerator), abs(denominator))
            if resolved.specification.lower is not None:
                restriction = max(
                    restriction,
                    max(
                        resolved.specification.lower * denominator - numerator,
                        0.0,
                    )
                    / scale,
                )
            if resolved.specification.upper is not None:
                restriction = max(
                    restriction,
                    max(
                        numerator - resolved.specification.upper * denominator,
                        0.0,
                    )
                    / scale,
                )

        if self.orientation is Orientation.INPUT:
            objective_value = evaluated_virtual_output
            solver_value = -solver_objective
            declared_value = technical_efficiency
        else:
            objective_value = evaluated_virtual_input
            solver_value = solver_objective
            declared_value = radial_factor
        objective = max(
            abs(objective_value - solver_value)
            / max(1.0, abs(objective_value), abs(solver_value)),
            abs(objective_value - declared_value)
            / max(1.0, abs(objective_value), abs(declared_value)),
        )
        return _Audit(
            primal=primal,
            ratio=ratio,
            restriction=restriction,
            objective=objective,
        )

    def _validate_technology(
        self,
        data: DEAData,
        reference: DEAData,
    ) -> None:
        if data.n_inputs != reference.n_inputs:
            raise ModelValidationError(
                "Evaluation and reference data must have the same number of inputs."
            )
        if data.n_outputs != reference.n_outputs:
            raise ModelValidationError(
                "Evaluation and reference data must have the same number of outputs."
            )
        if data.input_names != reference.input_names:
            raise ModelValidationError(
                "Evaluation and reference input names must match exactly."
            )
        if data.output_names != reference.output_names:
            raise ModelValidationError(
                "Evaluation and reference output names must match exactly."
            )

    def _resolve_restrictions(
        self,
        data: DEAData,
    ) -> tuple[_ResolvedRestriction, ...]:
        resolved: list[_ResolvedRestriction] = []
        for restriction in self.assurance_regions:
            names = (
                data.input_names
                if restriction.kind is WeightKind.INPUT
                else data.output_names
            )
            numerator = self._resolve_identifier(
                restriction.numerator,
                names=names,
            )
            denominator = self._resolve_identifier(
                restriction.denominator,
                names=names,
            )
            if numerator == denominator:
                raise ModelValidationError(
                    "Assurance-region numerator and denominator resolve to the "
                    "same multiplier."
                )
            resolved.append(
                _ResolvedRestriction(
                    specification=restriction,
                    numerator=numerator,
                    denominator=denominator,
                )
            )
        return tuple(resolved)

    def _resolve_identifier(
        self,
        identifier: WeightIdentifier,
        *,
        names: tuple[str, ...],
    ) -> int:
        if isinstance(identifier, str):
            try:
                return names.index(identifier)
            except ValueError as exc:
                raise ModelValidationError(
                    f"Unknown multiplier name {identifier!r}."
                ) from exc
        if identifier >= len(names):
            raise ModelValidationError(
                f"Multiplier index {identifier} is out of range for "
                f"{len(names)} weights."
            )
        return identifier

    def _restriction_rows(
        self,
        *,
        restrictions: tuple[_ResolvedRestriction, ...],
        input_scale: NDArray[np.float64],
        output_scale: NDArray[np.float64],
        n_inputs: int,
        n_outputs: int,
    ) -> tuple[NDArray[np.float64], ...]:
        rows: list[NDArray[np.float64]] = []
        for resolved in restrictions:
            specification = resolved.specification
            if specification.kind is WeightKind.INPUT:
                offset = 0
                scale = input_scale
            else:
                offset = n_inputs
                scale = output_scale
            if specification.lower is not None:
                row = np.zeros(n_inputs + n_outputs)
                row[offset + resolved.denominator] = (
                    specification.lower / scale[resolved.denominator]
                )
                row[offset + resolved.numerator] = -1.0 / scale[resolved.numerator]
                rows.append(self._normalize_restriction_row(row))
            if specification.upper is not None:
                row = np.zeros(n_inputs + n_outputs)
                row[offset + resolved.numerator] = 1.0 / scale[resolved.numerator]
                row[offset + resolved.denominator] = (
                    -specification.upper / scale[resolved.denominator]
                )
                rows.append(self._normalize_restriction_row(row))
        return tuple(rows)

    @staticmethod
    def _normalize_restriction_row(
        row: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Condition a homogeneous restriction without changing its feasible set."""
        scale = float(np.max(np.abs(row), initial=0.0))
        return row if scale == 0.0 else row / scale


def multiplier_dea(
    data: DEAData,
    *,
    orientation: Orientation | str = Orientation.INPUT,
    rts: ReturnsToScale | str = ReturnsToScale.CRS,
    assurance_regions: Sequence[AssuranceRegion] = (),
    weight_floor: float = 0.0,
    solver: LinearSolver | None = None,
    tolerance: float = 1e-9,
    reference: DEAData | None = None,
) -> MultiplierResult:
    """Convenience function for :class:`MultiplierDEA`."""
    return MultiplierDEA(
        orientation=orientation,
        rts=rts,
        assurance_regions=assurance_regions,
        weight_floor=weight_floor,
        solver=solver,
        tolerance=tolerance,
    ).fit(data, reference=reference)
