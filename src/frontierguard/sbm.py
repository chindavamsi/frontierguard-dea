"""Tone's non-oriented slack-based measure with an audited LP formulation.

The implementation uses the Charnes--Cooper transformation of the canonical
fractional SBM objective.  Every evaluated input and output must be strictly
positive because those values appear in the slack-ratio denominators.  No
epsilon replacement or translation is performed.
"""

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


def _validate_positive_denominators(data: DEAData) -> None:
    if np.any(data.inputs <= 0.0):
        raise ModelValidationError(
            "Tone's canonical SBM requires every evaluated input to be strictly "
            "positive because inputs are slack-ratio denominators."
        )
    if np.any(data.outputs <= 0.0):
        raise ModelValidationError(
            "Tone's canonical SBM requires every evaluated output to be strictly "
            "positive because outputs are slack-ratio denominators."
        )


def _ratio_residual(
    actual: ArrayLike | float,
    expected: ArrayLike | float,
    scale: ArrayLike | float,
) -> float:
    actual_array = np.asarray(actual, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    scale_array = np.maximum(
        np.abs(np.asarray(scale, dtype=float)), np.finfo(float).tiny
    )
    return float(
        np.max(
            np.abs(actual_array - expected_array) / scale_array,
            initial=0.0,
        )
    )


def _positive_violation(values: ArrayLike | float) -> float:
    return float(
        np.max(
            np.maximum(np.asarray(values, dtype=float), 0.0),
            initial=0.0,
        )
    )


@dataclass(frozen=True, slots=True)
class _SBMAudit:
    max_residual: float
    reconstructed_rho: float


def _audit_sbm_solution(
    *,
    inputs: NDArray[np.float64],
    outputs: NDArray[np.float64],
    evaluated_inputs: NDArray[np.float64],
    evaluated_outputs: NDArray[np.float64],
    returns_to_scale: ReturnsToScale,
    rho: float,
    lambdas: NDArray[np.float64],
    input_slacks: NDArray[np.float64],
    output_slacks: NDArray[np.float64],
    input_targets: NDArray[np.float64],
    output_targets: NDArray[np.float64],
    t: float,
    solver_objective: float,
) -> _SBMAudit:
    """Reconstruct every canonical SBM relation independently of the LP builder."""
    projected_inputs = inputs.T @ lambdas
    projected_outputs = outputs.T @ lambdas
    numerator = 1.0 - float(np.mean(input_slacks / evaluated_inputs))
    denominator = 1.0 + float(np.mean(output_slacks / evaluated_outputs))
    reconstructed_rho = numerator / denominator
    intensity_sum = float(np.sum(lambdas))

    residuals = [
        _ratio_residual(
            projected_inputs + input_slacks,
            evaluated_inputs,
            evaluated_inputs,
        ),
        _ratio_residual(
            projected_outputs - output_slacks,
            evaluated_outputs,
            evaluated_outputs,
        ),
        _ratio_residual(
            input_targets,
            projected_inputs,
            np.maximum(np.abs(projected_inputs), evaluated_inputs),
        ),
        _ratio_residual(
            output_targets,
            projected_outputs,
            np.maximum(np.abs(projected_outputs), evaluated_outputs),
        ),
        abs(t * denominator - 1.0),
        abs(rho - reconstructed_rho),
        abs(rho - solver_objective),
        _positive_violation(-lambdas),
        _positive_violation(-input_slacks / evaluated_inputs),
        _positive_violation(-output_slacks / evaluated_outputs),
        _positive_violation(-t),
        _positive_violation(-numerator),
        _positive_violation(-rho),
        _positive_violation(rho - 1.0),
    ]
    if returns_to_scale is ReturnsToScale.VRS:
        residuals.append(abs(intensity_sum - 1.0))
    elif returns_to_scale is ReturnsToScale.NIRS:
        residuals.append(_positive_violation(intensity_sum - 1.0))
    elif returns_to_scale is ReturnsToScale.NDRS:
        residuals.append(_positive_violation(1.0 - intensity_sum))

    return _SBMAudit(
        max_residual=max(residuals),
        reconstructed_rho=reconstructed_rho,
    )


@dataclass(frozen=True, slots=True, init=False)
class SBMResult:
    """Immutable results for Tone's non-oriented slack-based measure."""

    dmu_names: tuple[str, ...]
    reference_names: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    returns_to_scale: ReturnsToScale
    rho: NDArray[np.float64]
    lambdas: NDArray[np.float64]
    input_slacks: NDArray[np.float64]
    output_slacks: NDArray[np.float64]
    input_targets: NDArray[np.float64]
    output_targets: NDArray[np.float64]
    t: NDArray[np.float64]
    statuses: tuple[SolverStatus, ...]
    solver: str
    formulation: str
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
        rho: ArrayLike,
        lambdas: ArrayLike,
        input_slacks: ArrayLike,
        output_slacks: ArrayLike,
        input_targets: ArrayLike,
        output_targets: ArrayLike,
        t: ArrayLike,
        statuses: Sequence[SolverStatus],
        solver: str,
        exclude_self: bool,
        data_fingerprint: str,
        reference_fingerprint: str,
        audit_residual: ArrayLike,
        audit_tolerance: float,
    ) -> None:
        """Copy all numeric result values into read-only arrays."""
        object.__setattr__(self, "dmu_names", tuple(dmu_names))
        object.__setattr__(self, "reference_names", tuple(reference_names))
        object.__setattr__(self, "input_names", tuple(input_names))
        object.__setattr__(self, "output_names", tuple(output_names))
        object.__setattr__(self, "returns_to_scale", returns_to_scale)
        object.__setattr__(self, "rho", _readonly(rho))
        object.__setattr__(self, "lambdas", _readonly(lambdas))
        object.__setattr__(self, "input_slacks", _readonly(input_slacks))
        object.__setattr__(self, "output_slacks", _readonly(output_slacks))
        object.__setattr__(self, "input_targets", _readonly(input_targets))
        object.__setattr__(self, "output_targets", _readonly(output_targets))
        object.__setattr__(self, "t", _readonly(t))
        object.__setattr__(self, "statuses", tuple(statuses))
        object.__setattr__(self, "solver", str(solver))
        object.__setattr__(
            self,
            "formulation",
            "tone_non_oriented_sbm_charnes_cooper",
        )
        object.__setattr__(self, "exclude_self", bool(exclude_self))
        object.__setattr__(self, "data_fingerprint", str(data_fingerprint))
        object.__setattr__(
            self,
            "reference_fingerprint",
            str(reference_fingerprint),
        )
        object.__setattr__(self, "audit_residual", _readonly(audit_residual))
        object.__setattr__(self, "audit_tolerance", float(audit_tolerance))

    def to_frame(self) -> pd.DataFrame:
        """Return primary scores, transformations, and statuses by DMU."""
        return pd.DataFrame(
            {
                "dmu": self.dmu_names,
                "sbm_efficiency": self.rho,
                "t": self.t,
                "status": [status.value for status in self.statuses],
            }
        ).set_index("dmu")

    def peers(
        self,
        dmu: str | int,
        *,
        tolerance: float = 1e-8,
    ) -> dict[str, float]:
        """Return positive recovered reference intensities for one DMU."""
        position = self._position(dmu)
        return {
            name: float(weight)
            for name, weight in zip(
                self.reference_names,
                self.lambdas[position],
                strict=True,
            )
            if weight > tolerance
        }

    def targets(self, dmu: str | int) -> dict[str, dict[str, float]]:
        """Return the input and output projection targets for one DMU."""
        position = self._position(dmu)
        return {
            "inputs": {
                name: float(value)
                for name, value in zip(
                    self.input_names,
                    self.input_targets[position],
                    strict=True,
                )
            },
            "outputs": {
                name: float(value)
                for name, value in zip(
                    self.output_names,
                    self.output_targets[position],
                    strict=True,
                )
            },
        }

    def _position(self, dmu: str | int) -> int:
        if isinstance(dmu, str):
            try:
                position = self.dmu_names.index(dmu)
            except ValueError as exc:
                raise ModelValidationError(f"Unknown DMU {dmu!r}.") from exc
        else:
            position = dmu
        if not 0 <= position < len(self.dmu_names):
            raise ModelValidationError(f"DMU position {position} is out of range.")
        return position

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible result with solver provenance."""
        return {
            "schema_version": "0.1.0",
            "model": "slack_based_measure",
            "formulation": self.formulation,
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
            "rho": self.rho.tolist(),
            "lambdas": self.lambdas.tolist(),
            "input_slacks": self.input_slacks.tolist(),
            "output_slacks": self.output_slacks.tolist(),
            "input_targets": self.input_targets.tolist(),
            "output_targets": self.output_targets.tolist(),
            "t": self.t.tolist(),
            "statuses": [status.value for status in self.statuses],
        }


@dataclass(frozen=True, slots=True)
class _SBMFit:
    rho: float
    lambdas: NDArray[np.float64]
    input_slacks: NDArray[np.float64]
    output_slacks: NDArray[np.float64]
    input_targets: NDArray[np.float64]
    output_targets: NDArray[np.float64]
    t: float
    audit_residual: float


class SBMDEA:
    """Tone's canonical non-oriented slack-based measure.

    This is the standard SBM, not super-SBM.  ``exclude_self=True`` only changes
    the reference set and can make a model infeasible; it does not implement
    Tone's super-efficiency objective.
    """

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
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ModelValidationError("tolerance must be a positive finite number.")
        self.tolerance = float(tolerance)
        self.solver = ScipyHighsSolver() if solver is None else solver

    def fit(
        self,
        data: DEAData,
        *,
        reference: DEAData | None = None,
        exclude_self: bool = False,
    ) -> SBMResult:
        """Evaluate each DMU against the declared positive SBM technology."""
        if not isinstance(data, DEAData):
            raise ModelValidationError("data must be a DEAData instance.")
        reference_data = data if reference is None else reference
        if not isinstance(reference_data, DEAData):
            raise ModelValidationError("reference must be a DEAData instance.")
        _validate_reference(data, reference_data)
        _validate_positive_denominators(data)

        n_eval = data.n_dmus
        n_ref = reference_data.n_dmus
        rho = np.empty(n_eval)
        lambdas = np.zeros((n_eval, n_ref))
        input_slacks = np.empty((n_eval, data.n_inputs))
        output_slacks = np.empty((n_eval, data.n_outputs))
        input_targets = np.empty_like(input_slacks)
        output_targets = np.empty_like(output_slacks)
        transformations = np.empty(n_eval)
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
                label=str(data.dmu_names[evaluated]),
            )
            rho[evaluated] = fitted.rho
            lambdas[evaluated, included] = fitted.lambdas
            input_slacks[evaluated] = fitted.input_slacks
            output_slacks[evaluated] = fitted.output_slacks
            input_targets[evaluated] = fitted.input_targets
            output_targets[evaluated] = fitted.output_targets
            transformations[evaluated] = fitted.t
            audit_residuals[evaluated] = fitted.audit_residual
            statuses.append(SolverStatus.OPTIMAL)

        arrays = (
            rho,
            lambdas,
            input_slacks,
            output_slacks,
            input_targets,
            output_targets,
            transformations,
            audit_residuals,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise NumericalError("The completed SBM result contains non-finite values.")
        return SBMResult(
            dmu_names=tuple(data.dmu_names),
            reference_names=reference_names,
            input_names=tuple(data.input_names),
            output_names=tuple(data.output_names),
            returns_to_scale=self.returns_to_scale,
            rho=rho,
            lambdas=lambdas,
            input_slacks=input_slacks,
            output_slacks=output_slacks,
            input_targets=input_targets,
            output_targets=output_targets,
            t=transformations,
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
        label: str,
    ) -> _SBMFit:
        n_ref = inputs.shape[0]
        n_inputs = inputs.shape[1]
        n_outputs = outputs.shape[1]
        input_start = n_ref
        output_start = input_start + n_inputs
        t_position = output_start + n_outputs
        n_variables = t_position + 1
        equality_rows: list[NDArray[np.float64]] = []
        equality_bounds: list[float] = []
        inequality_rows: list[NDArray[np.float64]] = []
        inequality_bounds: list[float] = []

        # X * Lambda + S- = t * x_o.
        for column in range(n_inputs):
            row = np.zeros(n_variables)
            row[:n_ref] = inputs[:, column]
            row[input_start + column] = 1.0
            row[t_position] = -evaluated_inputs[column]
            equality_rows.append(row)
            equality_bounds.append(0.0)

        # Y * Lambda - S+ = t * y_o.
        for column in range(n_outputs):
            row = np.zeros(n_variables)
            row[:n_ref] = outputs[:, column]
            row[output_start + column] = -1.0
            row[t_position] = -evaluated_outputs[column]
            equality_rows.append(row)
            equality_bounds.append(0.0)

        # t + mean(S+ / y_o) = 1.
        normalization = np.zeros(n_variables)
        normalization[output_start:t_position] = 1.0 / (n_outputs * evaluated_outputs)
        normalization[t_position] = 1.0
        equality_rows.append(normalization)
        equality_bounds.append(1.0)

        # Transform sum(lambda) {=, <=, >=} 1 into
        # sum(Lambda) {=, <=, >=} t.
        rts_row = np.zeros(n_variables)
        rts_row[:n_ref] = 1.0
        rts_row[t_position] = -1.0
        if self.returns_to_scale is ReturnsToScale.VRS:
            equality_rows.append(rts_row)
            equality_bounds.append(0.0)
        elif self.returns_to_scale is ReturnsToScale.NIRS:
            inequality_rows.append(rts_row)
            inequality_bounds.append(0.0)
        elif self.returns_to_scale is ReturnsToScale.NDRS:
            inequality_rows.append(-rts_row)
            inequality_bounds.append(0.0)

        # min t - mean(S- / x_o).
        objective = np.zeros(n_variables)
        objective[input_start:output_start] = -1.0 / (n_inputs * evaluated_inputs)
        objective[t_position] = 1.0
        solved = self.solver.solve(
            LinearProgram(
                objective=objective,
                inequalities=_rows_or_none(inequality_rows),
                inequality_bounds=_values_or_none(inequality_bounds),
                equalities=np.vstack(equality_rows),
                equality_bounds=np.asarray(equality_bounds, dtype=float),
                variable_bounds=[(0.0, None)] * n_variables,
            )
        )
        if solved.status is not SolverStatus.OPTIMAL:
            raise SolverError(
                f"SBM model for {label!r} failed with {solved.status.value}: "
                f"{solved.message}"
            )

        transformed_lambdas = solved.variables[:n_ref].copy()
        transformed_input_slacks = solved.variables[input_start:output_start].copy()
        transformed_output_slacks = solved.variables[output_start:t_position].copy()
        t = float(solved.variables[t_position])
        transformed_arrays = (
            transformed_lambdas,
            transformed_input_slacks,
            transformed_output_slacks,
        )
        for values in transformed_arrays:
            values[np.abs(values) <= self.tolerance] = 0.0
        if t <= self.tolerance or not np.isfinite(t):
            raise NumericalError(
                f"SBM model for {label!r} produced a non-positive "
                "Charnes-Cooper transformation."
            )
        if any((values < -self.tolerance).any() for values in transformed_arrays):
            raise NumericalError(
                f"SBM model for {label!r} produced negative non-negative variables."
            )

        lambdas = np.maximum(transformed_lambdas, 0.0) / t
        input_slacks = np.maximum(transformed_input_slacks, 0.0) / t
        output_slacks = np.maximum(transformed_output_slacks, 0.0) / t
        for values in (lambdas, input_slacks, output_slacks):
            values[np.abs(values) <= self.tolerance] = 0.0
        input_targets = inputs.T @ lambdas
        output_targets = outputs.T @ lambdas
        numerator = 1.0 - float(np.mean(input_slacks / evaluated_inputs))
        denominator = 1.0 + float(np.mean(output_slacks / evaluated_outputs))
        rho = numerator / denominator
        audit_tolerance = max(10.0 * self.tolerance, 1e-7)
        if abs(rho) <= audit_tolerance:
            rho = 0.0
        elif abs(rho - 1.0) <= audit_tolerance:
            rho = 1.0

        audit = _audit_sbm_solution(
            inputs=inputs,
            outputs=outputs,
            evaluated_inputs=evaluated_inputs,
            evaluated_outputs=evaluated_outputs,
            returns_to_scale=self.returns_to_scale,
            rho=rho,
            lambdas=lambdas,
            input_slacks=input_slacks,
            output_slacks=output_slacks,
            input_targets=input_targets,
            output_targets=output_targets,
            t=t,
            solver_objective=solved.objective,
        )
        if not np.isfinite(audit.max_residual) or audit.max_residual > audit_tolerance:
            raise NumericalError(
                f"Post-solve SBM audit for {label!r} failed with scaled residual "
                f"{audit.max_residual:.3e} above {audit_tolerance:.3e}."
            )
        return _SBMFit(
            rho=rho,
            lambdas=lambdas,
            input_slacks=input_slacks,
            output_slacks=output_slacks,
            input_targets=input_targets,
            output_targets=output_targets,
            t=t,
            audit_residual=audit.max_residual,
        )


def sbm(
    data: DEAData,
    *,
    rts: ReturnsToScale | str = ReturnsToScale.VRS,
    reference: DEAData | None = None,
    exclude_self: bool = False,
) -> SBMResult:
    """Fit Tone's non-oriented SBM through a concise functional API."""
    return SBMDEA(rts=rts).fit(
        data,
        reference=reference,
        exclude_self=exclude_self,
    )
