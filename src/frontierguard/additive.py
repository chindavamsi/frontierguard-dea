"""Weighted additive DEA with explicit, auditable slack weights."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from frontierguard._data import DEAData
from frontierguard._enums import ReturnsToScale, SolverStatus, coerce_enum
from frontierguard._errors import ModelValidationError, NumericalError, SolverError
from frontierguard._solver import LinearProgram, LinearSolver, ScipyHighsSolver

WeightingMode = Literal["unit_normalized", "raw"]


def _readonly(values: ArrayLike) -> NDArray[np.float64]:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _rows_or_none(
    rows: list[NDArray[np.float64]],
) -> NDArray[np.float64] | None:
    return None if not rows else np.vstack(rows)


def _values_or_none(values: list[float]) -> NDArray[np.float64] | None:
    return None if not values else np.asarray(values, dtype=float)


def _scaled_absolute(
    residual: ArrayLike | float,
    *scales: ArrayLike | float,
) -> float:
    values = np.asarray(residual, dtype=float)
    numerator = float(np.max(np.abs(values), initial=0.0))
    denominator = 1.0
    for scale in scales:
        denominator = max(
            denominator,
            1.0
            + float(
                np.max(
                    np.abs(np.asarray(scale, dtype=float)),
                    initial=0.0,
                )
            ),
        )
    return numerator / denominator


def _scaled_positive(
    residual: ArrayLike | float,
    *scales: ArrayLike | float,
) -> float:
    return _scaled_absolute(
        np.maximum(np.asarray(residual, dtype=float), 0.0),
        *scales,
    )


def _validate_reference(data: DEAData, reference: DEAData) -> None:
    if data.n_inputs != reference.n_inputs:
        raise ModelValidationError(
            "Evaluation and reference data must have the same number of inputs."
        )
    if data.n_outputs != reference.n_outputs:
        raise ModelValidationError(
            "Evaluation and reference data must have the same number of outputs."
        )
    if tuple(data.input_names) != tuple(reference.input_names):
        raise ModelValidationError(
            "Evaluation and reference input names must match exactly."
        )
    if tuple(data.output_names) != tuple(reference.output_names):
        raise ModelValidationError(
            "Evaluation and reference output names must match exactly."
        )


def _weight_vector(
    values: ArrayLike,
    *,
    count: int,
    label: str,
) -> NDArray[np.float64]:
    try:
        weights = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{label} must be a numeric vector.") from exc
    if weights.ndim != 1 or weights.shape != (count,):
        raise ModelValidationError(
            f"{label} must have shape ({count},); received {weights.shape}."
        )
    if not np.isfinite(weights).all():
        raise ModelValidationError(f"{label} must contain only finite values.")
    if (weights < 0).any():
        raise ModelValidationError(f"{label} must be non-negative.")
    return np.array(weights, dtype=float, copy=True)


def _add_rts(
    *,
    returns_to_scale: ReturnsToScale,
    n_lambdas: int,
    n_variables: int,
    inequality_rows: list[NDArray[np.float64]],
    inequality_bounds: list[float],
    equality_rows: list[NDArray[np.float64]],
    equality_bounds: list[float],
) -> None:
    row = np.zeros(n_variables)
    row[:n_lambdas] = 1.0
    if returns_to_scale is ReturnsToScale.VRS:
        equality_rows.append(row)
        equality_bounds.append(1.0)
    elif returns_to_scale is ReturnsToScale.NIRS:
        inequality_rows.append(row)
        inequality_bounds.append(1.0)
    elif returns_to_scale is ReturnsToScale.NDRS:
        inequality_rows.append(-row)
        inequality_bounds.append(-1.0)


@dataclass(frozen=True, slots=True, init=False)
class AdditiveResult:
    """Immutable weighted-additive inefficiency results."""

    dmu_names: tuple[str, ...]
    reference_names: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    returns_to_scale: ReturnsToScale
    weighting: WeightingMode
    inefficiency: NDArray[np.float64]
    lambdas: NDArray[np.float64]
    input_slacks: NDArray[np.float64]
    output_slacks: NDArray[np.float64]
    input_targets: NDArray[np.float64]
    output_targets: NDArray[np.float64]
    input_weights: NDArray[np.float64]
    output_weights: NDArray[np.float64]
    statuses: tuple[SolverStatus, ...]
    solver: str
    exclude_self: bool
    data_fingerprint: str
    reference_fingerprint: str
    audit_residual: NDArray[np.float64]
    audit_tolerance: float

    def __init__(
        self,
        *,
        dmu_names: Sequence[str],
        reference_names: Sequence[str],
        input_names: Sequence[str],
        output_names: Sequence[str],
        returns_to_scale: ReturnsToScale,
        weighting: WeightingMode,
        inefficiency: ArrayLike,
        lambdas: ArrayLike,
        input_slacks: ArrayLike,
        output_slacks: ArrayLike,
        input_targets: ArrayLike,
        output_targets: ArrayLike,
        input_weights: ArrayLike,
        output_weights: ArrayLike,
        statuses: Sequence[SolverStatus],
        solver: str,
        exclude_self: bool,
        data_fingerprint: str,
        reference_fingerprint: str,
        audit_residual: ArrayLike,
        audit_tolerance: float,
    ) -> None:
        """Copy all numeric fields into read-only arrays."""
        object.__setattr__(self, "dmu_names", tuple(dmu_names))
        object.__setattr__(self, "reference_names", tuple(reference_names))
        object.__setattr__(self, "input_names", tuple(input_names))
        object.__setattr__(self, "output_names", tuple(output_names))
        object.__setattr__(self, "returns_to_scale", returns_to_scale)
        object.__setattr__(self, "weighting", weighting)
        object.__setattr__(self, "inefficiency", _readonly(inefficiency))
        object.__setattr__(self, "lambdas", _readonly(lambdas))
        object.__setattr__(self, "input_slacks", _readonly(input_slacks))
        object.__setattr__(self, "output_slacks", _readonly(output_slacks))
        object.__setattr__(self, "input_targets", _readonly(input_targets))
        object.__setattr__(self, "output_targets", _readonly(output_targets))
        object.__setattr__(self, "input_weights", _readonly(input_weights))
        object.__setattr__(self, "output_weights", _readonly(output_weights))
        object.__setattr__(self, "statuses", tuple(statuses))
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "exclude_self", exclude_self)
        object.__setattr__(self, "data_fingerprint", data_fingerprint)
        object.__setattr__(
            self,
            "reference_fingerprint",
            reference_fingerprint,
        )
        object.__setattr__(self, "audit_residual", _readonly(audit_residual))
        object.__setattr__(self, "audit_tolerance", float(audit_tolerance))

    def to_frame(self) -> pd.DataFrame:
        """Return one row per evaluated DMU."""
        return pd.DataFrame(
            {
                "dmu": self.dmu_names,
                "additive_inefficiency": self.inefficiency,
                "status": [status.value for status in self.statuses],
            }
        ).set_index("dmu")

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible representation."""
        return {
            "schema_version": "0.1.0",
            "model": "weighted_additive",
            "returns_to_scale": self.returns_to_scale.value,
            "weighting": self.weighting,
            "solver": self.solver,
            "exclude_self": self.exclude_self,
            "data_fingerprint": self.data_fingerprint,
            "reference_fingerprint": self.reference_fingerprint,
            "audit_residual": self.audit_residual.tolist(),
            "audit_tolerance": self.audit_tolerance,
            "dmu_names": list(self.dmu_names),
            "reference_names": list(self.reference_names),
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "inefficiency": self.inefficiency.tolist(),
            "lambdas": self.lambdas.tolist(),
            "input_slacks": self.input_slacks.tolist(),
            "output_slacks": self.output_slacks.tolist(),
            "input_targets": self.input_targets.tolist(),
            "output_targets": self.output_targets.tolist(),
            "input_weights": self.input_weights.tolist(),
            "output_weights": self.output_weights.tolist(),
            "statuses": [status.value for status in self.statuses],
        }


@dataclass(frozen=True, slots=True)
class _AdditiveFit:
    inefficiency: float
    lambdas: NDArray[np.float64]
    input_slacks: NDArray[np.float64]
    output_slacks: NDArray[np.float64]
    input_targets: NDArray[np.float64]
    output_targets: NDArray[np.float64]
    audit_residual: float


class WeightedAdditiveDEA:
    """Weighted additive DEA under a declared returns-to-scale technology.

    ``unit_normalized`` weights each slack by the inverse of its column maximum
    across the evaluation and reference samples. ``raw`` uses the caller's
    weights unchanged, so its score retains the resulting units.
    """

    def __init__(
        self,
        *,
        rts: ReturnsToScale | str = ReturnsToScale.VRS,
        weighting: WeightingMode = "unit_normalized",
        input_weights: ArrayLike | None = None,
        output_weights: ArrayLike | None = None,
        solver: LinearSolver | None = None,
        tolerance: float = 1e-9,
    ) -> None:
        self.returns_to_scale = coerce_enum(
            rts,
            ReturnsToScale,
            label="rts",
        )
        if weighting not in ("unit_normalized", "raw"):
            raise ModelValidationError("weighting must be 'unit_normalized' or 'raw'.")
        if weighting == "unit_normalized":
            if input_weights is not None or output_weights is not None:
                raise ModelValidationError("Explicit weights require weighting='raw'.")
        elif input_weights is None or output_weights is None:
            raise ModelValidationError(
                "weighting='raw' requires input_weights and output_weights."
            )
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ModelValidationError("tolerance must be a positive finite number.")
        self.weighting: WeightingMode = weighting
        self._input_weights = input_weights
        self._output_weights = output_weights
        self.tolerance = float(tolerance)
        self.solver = ScipyHighsSolver() if solver is None else solver

    def fit(
        self,
        data: DEAData,
        *,
        reference: DEAData | None = None,
        exclude_self: bool = False,
    ) -> AdditiveResult:
        """Maximize weighted input and output slacks for every DMU."""
        if not isinstance(data, DEAData):
            raise ModelValidationError("data must be a DEAData instance.")
        reference_data = data if reference is None else reference
        if not isinstance(reference_data, DEAData):
            raise ModelValidationError("reference must be a DEAData instance.")
        _validate_reference(data, reference_data)
        input_weights, output_weights = self._resolve_weights(data, reference_data)

        n_eval = data.n_dmus
        n_ref = reference_data.n_dmus
        inefficiency = np.empty(n_eval)
        lambdas = np.zeros((n_eval, n_ref))
        input_slacks = np.empty((n_eval, data.n_inputs))
        output_slacks = np.empty((n_eval, data.n_outputs))
        input_targets = np.empty_like(input_slacks)
        output_targets = np.empty_like(output_slacks)
        audit_residuals = np.empty(n_eval)
        statuses: list[SolverStatus] = []
        reference_names = tuple(reference_data.dmu_names)

        for evaluated in range(n_eval):
            included = np.ones(n_ref, dtype=bool)
            if exclude_self:
                name = str(data.dmu_names[evaluated])
                included &= np.asarray(
                    [str(candidate) != name for candidate in reference_names],
                    dtype=bool,
                )
            if not included.any():
                raise ModelValidationError(
                    f"DMU {data.dmu_names[evaluated]!r} has an empty reference set."
                )
            fitted = self._fit_one(
                inputs=reference_data.inputs[included],
                outputs=reference_data.outputs[included],
                evaluated_inputs=data.inputs[evaluated],
                evaluated_outputs=data.outputs[evaluated],
                input_weights=input_weights,
                output_weights=output_weights,
                label=str(data.dmu_names[evaluated]),
            )
            inefficiency[evaluated] = fitted.inefficiency
            lambdas[evaluated, included] = fitted.lambdas
            input_slacks[evaluated] = fitted.input_slacks
            output_slacks[evaluated] = fitted.output_slacks
            input_targets[evaluated] = fitted.input_targets
            output_targets[evaluated] = fitted.output_targets
            audit_residuals[evaluated] = fitted.audit_residual
            statuses.append(SolverStatus.OPTIMAL)

        arrays = (
            inefficiency,
            lambdas,
            input_slacks,
            output_slacks,
            input_targets,
            output_targets,
            input_weights,
            output_weights,
            audit_residuals,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise NumericalError("The additive result contains non-finite values.")
        return AdditiveResult(
            dmu_names=tuple(data.dmu_names),
            reference_names=reference_names,
            input_names=tuple(data.input_names),
            output_names=tuple(data.output_names),
            returns_to_scale=self.returns_to_scale,
            weighting=self.weighting,
            inefficiency=inefficiency,
            lambdas=lambdas,
            input_slacks=input_slacks,
            output_slacks=output_slacks,
            input_targets=input_targets,
            output_targets=output_targets,
            input_weights=input_weights,
            output_weights=output_weights,
            statuses=statuses,
            solver=self.solver.name,
            exclude_self=exclude_self,
            data_fingerprint=data.fingerprint,
            reference_fingerprint=reference_data.fingerprint,
            audit_residual=audit_residuals,
            audit_tolerance=max(10.0 * self.tolerance, 1e-7),
        )

    def _resolve_weights(
        self,
        data: DEAData,
        reference: DEAData,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self.weighting == "unit_normalized":
            input_scale = np.maximum(
                data.inputs.max(axis=0),
                reference.inputs.max(axis=0),
            )
            output_scale = np.maximum(
                data.outputs.max(axis=0),
                reference.outputs.max(axis=0),
            )
            return 1.0 / input_scale, 1.0 / output_scale

        if self._input_weights is None or self._output_weights is None:
            raise ModelValidationError("Raw weights are missing.")
        input_weights = _weight_vector(
            self._input_weights,
            count=data.n_inputs,
            label="input_weights",
        )
        output_weights = _weight_vector(
            self._output_weights,
            count=data.n_outputs,
            label="output_weights",
        )
        if not np.any(input_weights > 0) and not np.any(output_weights > 0):
            raise ModelValidationError(
                "At least one raw input or output weight must be positive."
            )
        return input_weights, output_weights

    def _fit_one(
        self,
        *,
        inputs: NDArray[np.float64],
        outputs: NDArray[np.float64],
        evaluated_inputs: NDArray[np.float64],
        evaluated_outputs: NDArray[np.float64],
        input_weights: NDArray[np.float64],
        output_weights: NDArray[np.float64],
        label: str,
    ) -> _AdditiveFit:
        n_ref = inputs.shape[0]
        n_inputs = inputs.shape[1]
        n_outputs = outputs.shape[1]
        n_variables = n_ref + n_inputs + n_outputs
        inequality_rows: list[NDArray[np.float64]] = []
        inequality_bounds: list[float] = []
        equality_rows: list[NDArray[np.float64]] = []
        equality_bounds: list[float] = []

        for column in range(n_inputs):
            row = np.zeros(n_variables)
            row[:n_ref] = inputs[:, column]
            row[n_ref + column] = 1.0
            equality_rows.append(row)
            equality_bounds.append(float(evaluated_inputs[column]))
        for column in range(n_outputs):
            row = np.zeros(n_variables)
            row[:n_ref] = outputs[:, column]
            row[n_ref + n_inputs + column] = -1.0
            equality_rows.append(row)
            equality_bounds.append(float(evaluated_outputs[column]))
        _add_rts(
            returns_to_scale=self.returns_to_scale,
            n_lambdas=n_ref,
            n_variables=n_variables,
            inequality_rows=inequality_rows,
            inequality_bounds=inequality_bounds,
            equality_rows=equality_rows,
            equality_bounds=equality_bounds,
        )
        objective = np.zeros(n_variables)
        objective[n_ref : n_ref + n_inputs] = -input_weights
        objective[n_ref + n_inputs :] = -output_weights
        solved = self.solver.solve(
            LinearProgram(
                objective=objective,
                inequalities=_rows_or_none(inequality_rows),
                inequality_bounds=_values_or_none(inequality_bounds),
                equalities=_rows_or_none(equality_rows),
                equality_bounds=_values_or_none(equality_bounds),
                variable_bounds=[(0.0, None)] * n_variables,
            )
        )
        if solved.status is not SolverStatus.OPTIMAL:
            raise SolverError(
                f"Weighted additive model for {label!r} failed with "
                f"{solved.status.value}: {solved.message}"
            )
        lambdas = solved.variables[:n_ref].copy()
        input_slacks = solved.variables[n_ref : n_ref + n_inputs].copy()
        output_slacks = solved.variables[n_ref + n_inputs :].copy()
        for array in (lambdas, input_slacks, output_slacks):
            array[np.abs(array) <= self.tolerance] = 0.0
        if any(
            (array < -self.tolerance).any()
            for array in (lambdas, input_slacks, output_slacks)
        ):
            raise NumericalError(
                f"Weighted additive model for {label!r} produced negative "
                "non-negative variables."
            )
        lambdas = np.maximum(lambdas, 0.0)
        input_slacks = np.maximum(input_slacks, 0.0)
        output_slacks = np.maximum(output_slacks, 0.0)
        input_targets = inputs.T @ lambdas
        output_targets = outputs.T @ lambdas
        inefficiency = float(
            input_weights @ input_slacks + output_weights @ output_slacks
        )
        audit_residual = self._audit(
            inputs=inputs,
            outputs=outputs,
            evaluated_inputs=evaluated_inputs,
            evaluated_outputs=evaluated_outputs,
            lambdas=lambdas,
            input_slacks=input_slacks,
            output_slacks=output_slacks,
            input_targets=input_targets,
            output_targets=output_targets,
            input_weights=input_weights,
            output_weights=output_weights,
            inefficiency=inefficiency,
            solver_objective=solved.objective,
        )
        audit_tolerance = max(10.0 * self.tolerance, 1e-7)
        if not np.isfinite(audit_residual) or audit_residual > audit_tolerance:
            raise NumericalError(
                f"Post-solve audit for {label!r} failed with scaled residual "
                f"{audit_residual:.3e} above {audit_tolerance:.3e}."
            )
        return _AdditiveFit(
            inefficiency=inefficiency,
            lambdas=lambdas,
            input_slacks=input_slacks,
            output_slacks=output_slacks,
            input_targets=input_targets,
            output_targets=output_targets,
            audit_residual=audit_residual,
        )

    def _audit(
        self,
        *,
        inputs: NDArray[np.float64],
        outputs: NDArray[np.float64],
        evaluated_inputs: NDArray[np.float64],
        evaluated_outputs: NDArray[np.float64],
        lambdas: NDArray[np.float64],
        input_slacks: NDArray[np.float64],
        output_slacks: NDArray[np.float64],
        input_targets: NDArray[np.float64],
        output_targets: NDArray[np.float64],
        input_weights: NDArray[np.float64],
        output_weights: NDArray[np.float64],
        inefficiency: float,
        solver_objective: float,
    ) -> float:
        projected_inputs = inputs.T @ lambdas
        projected_outputs = outputs.T @ lambdas
        residuals = [
            _scaled_positive(-lambdas, lambdas),
            _scaled_positive(-input_slacks, evaluated_inputs),
            _scaled_positive(-output_slacks, evaluated_outputs),
            _scaled_absolute(
                projected_inputs + input_slacks - evaluated_inputs,
                projected_inputs,
                evaluated_inputs,
            ),
            _scaled_absolute(
                projected_outputs - output_slacks - evaluated_outputs,
                projected_outputs,
                evaluated_outputs,
            ),
            _scaled_absolute(
                input_targets - projected_inputs,
                input_targets,
                projected_inputs,
            ),
            _scaled_absolute(
                output_targets - projected_outputs,
                output_targets,
                projected_outputs,
            ),
            _scaled_absolute(
                inefficiency
                - float(input_weights @ input_slacks + output_weights @ output_slacks),
                inefficiency,
            ),
            _scaled_absolute(
                solver_objective + inefficiency,
                solver_objective,
                inefficiency,
            ),
        ]
        intensity_sum = float(lambdas.sum())
        if self.returns_to_scale is ReturnsToScale.VRS:
            residuals.append(_scaled_absolute(intensity_sum - 1.0, intensity_sum))
        elif self.returns_to_scale is ReturnsToScale.NIRS:
            residuals.append(_scaled_positive(intensity_sum - 1.0, intensity_sum))
        elif self.returns_to_scale is ReturnsToScale.NDRS:
            residuals.append(_scaled_positive(1.0 - intensity_sum, intensity_sum))
        return max(residuals, default=0.0)
