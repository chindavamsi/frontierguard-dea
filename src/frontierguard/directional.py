"""Directional-distance DEA with explicit contraction and expansion directions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from frontierguard._data import DEAData
from frontierguard._enums import ReturnsToScale, SolverStatus, coerce_enum
from frontierguard._errors import ModelValidationError, NumericalError, SolverError
from frontierguard._solver import LinearProgram, LinearSolver, ScipyHighsSolver


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


def _direction_matrix(
    values: ArrayLike,
    *,
    n_dmus: int,
    n_dimensions: int,
    label: str,
) -> NDArray[np.float64]:
    try:
        direction = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(
            f"{label} must be a numeric vector or matrix."
        ) from exc
    if direction.ndim == 1:
        if direction.shape != (n_dimensions,):
            raise ModelValidationError(
                f"{label} vector must have shape ({n_dimensions},); "
                f"received {direction.shape}."
            )
        direction = np.broadcast_to(direction, (n_dmus, n_dimensions))
    elif direction.ndim == 2:
        if direction.shape != (n_dmus, n_dimensions):
            raise ModelValidationError(
                f"{label} matrix must have shape ({n_dmus}, {n_dimensions}); "
                f"received {direction.shape}."
            )
    else:
        raise ModelValidationError(f"{label} must be a one- or two-dimensional array.")
    if not np.isfinite(direction).all():
        raise ModelValidationError(f"{label} must contain only finite values.")
    if (direction < 0).any():
        raise ModelValidationError(f"{label} must be non-negative.")
    return np.array(direction, dtype=float, copy=True)


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
class DirectionalResult:
    """Immutable directional-distance results."""

    dmu_names: tuple[str, ...]
    reference_names: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    returns_to_scale: ReturnsToScale
    beta: NDArray[np.float64]
    lambdas: NDArray[np.float64]
    input_directions: NDArray[np.float64]
    output_directions: NDArray[np.float64]
    input_slacks: NDArray[np.float64]
    output_slacks: NDArray[np.float64]
    input_targets: NDArray[np.float64]
    output_targets: NDArray[np.float64]
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
        beta: ArrayLike,
        lambdas: ArrayLike,
        input_directions: ArrayLike,
        output_directions: ArrayLike,
        input_slacks: ArrayLike,
        output_slacks: ArrayLike,
        input_targets: ArrayLike,
        output_targets: ArrayLike,
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
        object.__setattr__(self, "beta", _readonly(beta))
        object.__setattr__(self, "lambdas", _readonly(lambdas))
        object.__setattr__(
            self,
            "input_directions",
            _readonly(input_directions),
        )
        object.__setattr__(
            self,
            "output_directions",
            _readonly(output_directions),
        )
        object.__setattr__(self, "input_slacks", _readonly(input_slacks))
        object.__setattr__(self, "output_slacks", _readonly(output_slacks))
        object.__setattr__(self, "input_targets", _readonly(input_targets))
        object.__setattr__(self, "output_targets", _readonly(output_targets))
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
        """Return one directional distance per evaluated DMU."""
        return pd.DataFrame(
            {
                "dmu": self.dmu_names,
                "directional_distance": self.beta,
                "status": [status.value for status in self.statuses],
            }
        ).set_index("dmu")

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible representation."""
        return {
            "schema_version": "0.1.0",
            "model": "directional_distance",
            "returns_to_scale": self.returns_to_scale.value,
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
            "beta": self.beta.tolist(),
            "lambdas": self.lambdas.tolist(),
            "input_directions": self.input_directions.tolist(),
            "output_directions": self.output_directions.tolist(),
            "input_slacks": self.input_slacks.tolist(),
            "output_slacks": self.output_slacks.tolist(),
            "input_targets": self.input_targets.tolist(),
            "output_targets": self.output_targets.tolist(),
            "statuses": [status.value for status in self.statuses],
        }


@dataclass(frozen=True, slots=True)
class _DirectionalFit:
    beta: float
    lambdas: NDArray[np.float64]
    input_slacks: NDArray[np.float64]
    output_slacks: NDArray[np.float64]
    input_targets: NDArray[np.float64]
    output_targets: NDArray[np.float64]
    audit_residual: float


class DirectionalDEA:
    """Directional-distance DEA for non-negative contraction/expansion paths."""

    def __init__(
        self,
        *,
        rts: ReturnsToScale | str = ReturnsToScale.VRS,
        solver: LinearSolver | None = None,
        tolerance: float = 1e-9,
    ) -> None:
        self.returns_to_scale = coerce_enum(
            rts,
            ReturnsToScale,
            label="rts",
        )
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ModelValidationError("tolerance must be a positive finite number.")
        self.tolerance = float(tolerance)
        self.solver = ScipyHighsSolver() if solver is None else solver

    def fit(
        self,
        data: DEAData,
        *,
        input_direction: ArrayLike,
        output_direction: ArrayLike,
        reference: DEAData | None = None,
        exclude_self: bool = False,
    ) -> DirectionalResult:
        """Maximize beta along the explicitly supplied direction for each DMU."""
        if not isinstance(data, DEAData):
            raise ModelValidationError("data must be a DEAData instance.")
        reference_data = data if reference is None else reference
        if not isinstance(reference_data, DEAData):
            raise ModelValidationError("reference must be a DEAData instance.")
        _validate_reference(data, reference_data)
        input_directions = _direction_matrix(
            input_direction,
            n_dmus=data.n_dmus,
            n_dimensions=data.n_inputs,
            label="input_direction",
        )
        output_directions = _direction_matrix(
            output_direction,
            n_dmus=data.n_dmus,
            n_dimensions=data.n_outputs,
            label="output_direction",
        )
        inactive = np.logical_and(
            np.all(input_directions <= self.tolerance, axis=1),
            np.all(output_directions <= self.tolerance, axis=1),
        )
        if inactive.any():
            names = ", ".join(
                repr(data.dmu_names[index]) for index in np.flatnonzero(inactive)
            )
            raise ModelValidationError(
                "Each DMU needs at least one positive directional component; "
                f"zero direction for {names}."
            )

        n_eval = data.n_dmus
        n_ref = reference_data.n_dmus
        betas = np.empty(n_eval)
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
                input_direction=input_directions[evaluated],
                output_direction=output_directions[evaluated],
                label=str(data.dmu_names[evaluated]),
            )
            betas[evaluated] = fitted.beta
            lambdas[evaluated, included] = fitted.lambdas
            input_slacks[evaluated] = fitted.input_slacks
            output_slacks[evaluated] = fitted.output_slacks
            input_targets[evaluated] = fitted.input_targets
            output_targets[evaluated] = fitted.output_targets
            audit_residuals[evaluated] = fitted.audit_residual
            statuses.append(SolverStatus.OPTIMAL)

        arrays = (
            betas,
            lambdas,
            input_directions,
            output_directions,
            input_slacks,
            output_slacks,
            input_targets,
            output_targets,
            audit_residuals,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise NumericalError("The directional result contains non-finite values.")
        return DirectionalResult(
            dmu_names=tuple(data.dmu_names),
            reference_names=reference_names,
            input_names=tuple(data.input_names),
            output_names=tuple(data.output_names),
            returns_to_scale=self.returns_to_scale,
            beta=betas,
            lambdas=lambdas,
            input_directions=input_directions,
            output_directions=output_directions,
            input_slacks=input_slacks,
            output_slacks=output_slacks,
            input_targets=input_targets,
            output_targets=output_targets,
            statuses=statuses,
            solver=self.solver.name,
            exclude_self=exclude_self,
            data_fingerprint=data.fingerprint,
            reference_fingerprint=reference_data.fingerprint,
            audit_residual=audit_residuals,
            audit_tolerance=max(10.0 * self.tolerance, 1e-7),
        )

    def _fit_one(
        self,
        *,
        inputs: NDArray[np.float64],
        outputs: NDArray[np.float64],
        evaluated_inputs: NDArray[np.float64],
        evaluated_outputs: NDArray[np.float64],
        input_direction: NDArray[np.float64],
        output_direction: NDArray[np.float64],
        label: str,
    ) -> _DirectionalFit:
        n_ref = inputs.shape[0]
        n_variables = n_ref + 1
        inequality_rows: list[NDArray[np.float64]] = []
        inequality_bounds: list[float] = []
        equality_rows: list[NDArray[np.float64]] = []
        equality_bounds: list[float] = []

        for column in range(inputs.shape[1]):
            row = np.zeros(n_variables)
            row[:n_ref] = inputs[:, column]
            row[-1] = input_direction[column]
            inequality_rows.append(row)
            inequality_bounds.append(float(evaluated_inputs[column]))
        for column in range(outputs.shape[1]):
            row = np.zeros(n_variables)
            row[:n_ref] = -outputs[:, column]
            row[-1] = output_direction[column]
            inequality_rows.append(row)
            inequality_bounds.append(-float(evaluated_outputs[column]))
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
        objective[-1] = -1.0
        solved = self.solver.solve(
            LinearProgram(
                objective=objective,
                inequalities=_rows_or_none(inequality_rows),
                inequality_bounds=_values_or_none(inequality_bounds),
                equalities=_rows_or_none(equality_rows),
                equality_bounds=_values_or_none(equality_bounds),
                variable_bounds=[(0.0, None)] * n_ref + [(None, None)],
            )
        )
        if solved.status is not SolverStatus.OPTIMAL:
            raise SolverError(
                f"Directional model for {label!r} failed with "
                f"{solved.status.value}: {solved.message}"
            )
        lambdas = solved.variables[:n_ref].copy()
        beta = float(solved.variables[-1])
        lambdas[np.abs(lambdas) <= self.tolerance] = 0.0
        if (lambdas < -self.tolerance).any():
            raise NumericalError(
                f"Directional model for {label!r} produced negative "
                "reference intensities."
            )
        lambdas = np.maximum(lambdas, 0.0)
        input_targets = inputs.T @ lambdas
        output_targets = outputs.T @ lambdas
        directional_inputs = evaluated_inputs - beta * input_direction
        directional_outputs = evaluated_outputs + beta * output_direction
        input_slacks = directional_inputs - input_targets
        output_slacks = output_targets - directional_outputs
        input_slacks[np.abs(input_slacks) <= self.tolerance] = 0.0
        output_slacks[np.abs(output_slacks) <= self.tolerance] = 0.0
        if (input_slacks < -self.tolerance).any() or (
            output_slacks < -self.tolerance
        ).any():
            raise NumericalError(
                f"Directional slack reconstruction for {label!r} produced "
                "negative slacks."
            )
        input_slacks = np.maximum(input_slacks, 0.0)
        output_slacks = np.maximum(output_slacks, 0.0)
        audit_residual = self._audit(
            inputs=inputs,
            outputs=outputs,
            evaluated_inputs=evaluated_inputs,
            evaluated_outputs=evaluated_outputs,
            input_direction=input_direction,
            output_direction=output_direction,
            beta=beta,
            lambdas=lambdas,
            input_slacks=input_slacks,
            output_slacks=output_slacks,
            input_targets=input_targets,
            output_targets=output_targets,
            solver_objective=solved.objective,
        )
        audit_tolerance = max(10.0 * self.tolerance, 1e-7)
        if not np.isfinite(audit_residual) or audit_residual > audit_tolerance:
            raise NumericalError(
                f"Post-solve audit for {label!r} failed with scaled residual "
                f"{audit_residual:.3e} above {audit_tolerance:.3e}."
            )
        return _DirectionalFit(
            beta=beta,
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
        input_direction: NDArray[np.float64],
        output_direction: NDArray[np.float64],
        beta: float,
        lambdas: NDArray[np.float64],
        input_slacks: NDArray[np.float64],
        output_slacks: NDArray[np.float64],
        input_targets: NDArray[np.float64],
        output_targets: NDArray[np.float64],
        solver_objective: float,
    ) -> float:
        projected_inputs = inputs.T @ lambdas
        projected_outputs = outputs.T @ lambdas
        directional_inputs = evaluated_inputs - beta * input_direction
        directional_outputs = evaluated_outputs + beta * output_direction
        residuals = [
            _scaled_positive(-lambdas, lambdas),
            _scaled_positive(-input_slacks, evaluated_inputs),
            _scaled_positive(-output_slacks, evaluated_outputs),
            _scaled_positive(
                projected_inputs - directional_inputs,
                projected_inputs,
                directional_inputs,
            ),
            _scaled_positive(
                directional_outputs - projected_outputs,
                directional_outputs,
                projected_outputs,
            ),
            _scaled_absolute(
                input_slacks - (directional_inputs - projected_inputs),
                input_slacks,
                directional_inputs,
            ),
            _scaled_absolute(
                output_slacks - (projected_outputs - directional_outputs),
                output_slacks,
                directional_outputs,
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
                solver_objective + beta,
                solver_objective,
                beta,
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
