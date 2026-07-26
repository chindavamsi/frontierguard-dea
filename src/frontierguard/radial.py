"""Radial input- and output-oriented envelopment models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from frontierguard._audit import audit_radial_solution
from frontierguard._data import DEAData
from frontierguard._enums import (
    Orientation,
    ReturnsToScale,
    SlackStrategy,
    SolverStatus,
    coerce_enum,
)
from frontierguard._errors import ModelValidationError, NumericalError, SolverError
from frontierguard._results import RadialResult
from frontierguard._solver import (
    LinearProgram,
    LinearSolution,
    LinearSolver,
    ScipyHighsSolver,
)


def _empty_rows(n_variables: int) -> tuple[list[NDArray[np.float64]], list[float]]:
    return [], []


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


def _matrix_or_none(rows: list[NDArray[np.float64]]) -> NDArray[np.float64] | None:
    return None if not rows else np.vstack(rows)


def _vector_or_none(values: list[float]) -> NDArray[np.float64] | None:
    return None if not values else np.asarray(values, dtype=float)


@dataclass(frozen=True, slots=True)
class _FitOne:
    radial_factor: float
    lambdas: NDArray[np.float64]
    input_slacks: NDArray[np.float64]
    output_slacks: NDArray[np.float64]
    input_targets: NDArray[np.float64]
    output_targets: NDArray[np.float64]
    status: SolverStatus
    audit_residual: float


class RadialDEA:
    """Farrell radial DEA with an optional lexicographic slack phase."""

    def __init__(
        self,
        *,
        orientation: Orientation | str = Orientation.INPUT,
        rts: ReturnsToScale | str = ReturnsToScale.VRS,
        slacks: SlackStrategy | str = SlackStrategy.NORMALIZED,
        solver: LinearSolver | None = None,
        tolerance: float = 1e-9,
    ) -> None:
        self.orientation = coerce_enum(
            orientation,
            Orientation,
            label="orientation",
        )
        self.returns_to_scale = coerce_enum(
            rts,
            ReturnsToScale,
            label="rts",
        )
        self.slack_strategy = coerce_enum(
            slacks,
            SlackStrategy,
            label="slacks",
        )
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ModelValidationError("tolerance must be a positive finite number.")
        self.tolerance = float(tolerance)
        self.solver = ScipyHighsSolver() if solver is None else solver

    def fit(
        self,
        data: DEAData,
        *,
        reference: DEAData | None = None,
        exclude_self: bool = False,
    ) -> RadialResult:
        """Fit every DMU against the declared reference technology."""
        if not isinstance(data, DEAData):
            raise ModelValidationError("data must be a DEAData instance.")
        reference_data = data if reference is None else reference
        if not isinstance(reference_data, DEAData):
            raise ModelValidationError("reference must be a DEAData instance.")
        if data.n_inputs != reference_data.n_inputs:
            raise ModelValidationError(
                "Evaluation and reference data must have the same number of inputs."
            )
        if data.n_outputs != reference_data.n_outputs:
            raise ModelValidationError(
                "Evaluation and reference data must have the same number of outputs."
            )
        if tuple(data.input_names) != tuple(reference_data.input_names):
            raise ModelValidationError(
                "Evaluation and reference input names must match exactly."
            )
        if tuple(data.output_names) != tuple(reference_data.output_names):
            raise ModelValidationError(
                "Evaluation and reference output names must match exactly."
            )

        n_eval = data.n_dmus
        n_ref = reference_data.n_dmus
        factors = np.empty(n_eval)
        efficiencies = np.empty(n_eval)
        lambdas = np.zeros((n_eval, n_ref))
        input_slacks = np.empty((n_eval, data.n_inputs))
        output_slacks = np.empty((n_eval, data.n_outputs))
        input_targets = np.empty_like(input_slacks)
        output_targets = np.empty_like(output_slacks)
        audit_residuals = np.empty(n_eval)
        statuses: list[SolverStatus] = []

        reference_names = tuple(reference_data.dmu_names)
        for position in range(n_eval):
            included = np.ones(n_ref, dtype=bool)
            if exclude_self:
                current_name = str(data.dmu_names[position])
                included &= np.asarray(
                    [str(name) != current_name for name in reference_names],
                    dtype=bool,
                )
            if not included.any():
                raise ModelValidationError(
                    f"DMU {data.dmu_names[position]!r} has an empty reference set."
                )
            fit = self._fit_one(
                inputs=reference_data.inputs[included],
                outputs=reference_data.outputs[included],
                evaluated_inputs=data.inputs[position],
                evaluated_outputs=data.outputs[position],
                label=str(data.dmu_names[position]),
            )
            factors[position] = fit.radial_factor
            efficiencies[position] = (
                fit.radial_factor
                if self.orientation is Orientation.INPUT
                else 1.0 / fit.radial_factor
            )
            lambdas[position, included] = fit.lambdas
            input_slacks[position] = fit.input_slacks
            output_slacks[position] = fit.output_slacks
            input_targets[position] = fit.input_targets
            output_targets[position] = fit.output_targets
            audit_residuals[position] = fit.audit_residual
            statuses.append(fit.status)

        arrays = (
            factors,
            efficiencies,
            lambdas,
            input_slacks,
            output_slacks,
            input_targets,
            output_targets,
            audit_residuals,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise NumericalError(
                "The completed radial result contains non-finite values."
            )
        return RadialResult(
            dmu_names=tuple(data.dmu_names),
            reference_names=reference_names,
            input_names=tuple(data.input_names),
            output_names=tuple(data.output_names),
            orientation=self.orientation,
            returns_to_scale=self.returns_to_scale,
            radial_factor=factors,
            technical_efficiency=efficiencies,
            lambdas=lambdas,
            input_slacks=input_slacks,
            output_slacks=output_slacks,
            input_targets=input_targets,
            output_targets=output_targets,
            statuses=tuple(statuses),
            solver=self.solver.name,
            phase_two=self.slack_strategy is not SlackStrategy.NONE,
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
    ) -> _FitOne:
        first = self._phase_one(
            inputs=inputs,
            outputs=outputs,
            evaluated_inputs=evaluated_inputs,
            evaluated_outputs=evaluated_outputs,
        )
        if first.status is not SolverStatus.OPTIMAL:
            raise SolverError(
                f"Radial model for {label!r} failed with {first.status.value}: "
                f"{first.message}"
            )
        factor = float(first.variables[-1])
        if self.orientation is Orientation.OUTPUT and factor <= self.tolerance:
            raise NumericalError(
                f"Output expansion factor for {label!r} is not positive."
            )

        if self.slack_strategy is SlackStrategy.NONE:
            intensities = first.variables[:-1].copy()
            input_targets = inputs.T @ intensities
            output_targets = outputs.T @ intensities
            if self.orientation is Orientation.INPUT:
                input_slacks = factor * evaluated_inputs - input_targets
                output_slacks = output_targets - evaluated_outputs
            else:
                input_slacks = evaluated_inputs - input_targets
                output_slacks = output_targets - factor * evaluated_outputs
        else:
            second = self._phase_two(
                inputs=inputs,
                outputs=outputs,
                evaluated_inputs=evaluated_inputs,
                evaluated_outputs=evaluated_outputs,
                radial_factor=factor,
            )
            if second.status is not SolverStatus.OPTIMAL:
                raise SolverError(
                    f"Slack phase for {label!r} failed with {second.status.value}: "
                    f"{second.message}"
                )
            n_ref = inputs.shape[0]
            n_inputs = inputs.shape[1]
            intensities = second.variables[:n_ref].copy()
            input_slacks = second.variables[n_ref : n_ref + n_inputs].copy()
            output_slacks = second.variables[n_ref + n_inputs :].copy()
            input_targets = inputs.T @ intensities
            output_targets = outputs.T @ intensities

        input_slacks[np.abs(input_slacks) <= self.tolerance] = 0.0
        output_slacks[np.abs(output_slacks) <= self.tolerance] = 0.0
        intensities[np.abs(intensities) <= self.tolerance] = 0.0
        if (input_slacks < -self.tolerance).any() or (
            output_slacks < -self.tolerance
        ).any():
            raise NumericalError(
                f"Slack reconstruction for {label!r} produced negative slacks."
            )
        audit = audit_radial_solution(
            inputs=inputs,
            outputs=outputs,
            evaluated_inputs=evaluated_inputs,
            evaluated_outputs=evaluated_outputs,
            orientation=self.orientation,
            returns_to_scale=self.returns_to_scale,
            radial_factor=factor,
            lambdas=intensities,
            input_slacks=np.maximum(input_slacks, 0.0),
            output_slacks=np.maximum(output_slacks, 0.0),
            input_targets=input_targets,
            output_targets=output_targets,
            solver_objective=first.objective,
            phase_two=self.slack_strategy is not SlackStrategy.NONE,
            tolerance=max(10.0 * self.tolerance, 1e-7),
        )
        if not audit.passed:
            raise NumericalError(
                f"Post-solve audit for {label!r} failed with scaled residual "
                f"{audit.max_residual:.3e} above {audit.tolerance:.3e}."
            )
        return _FitOne(
            radial_factor=factor,
            lambdas=intensities,
            input_slacks=np.maximum(input_slacks, 0.0),
            output_slacks=np.maximum(output_slacks, 0.0),
            input_targets=input_targets,
            output_targets=output_targets,
            status=SolverStatus.OPTIMAL,
            audit_residual=audit.max_residual,
        )

    def _phase_one(
        self,
        *,
        inputs: NDArray[np.float64],
        outputs: NDArray[np.float64],
        evaluated_inputs: NDArray[np.float64],
        evaluated_outputs: NDArray[np.float64],
    ) -> LinearSolution:
        n_ref = inputs.shape[0]
        n_variables = n_ref + 1
        inequality_rows, inequality_bounds = _empty_rows(n_variables)
        equality_rows, equality_bounds = _empty_rows(n_variables)

        if self.orientation is Orientation.INPUT:
            for column in range(inputs.shape[1]):
                row = np.zeros(n_variables)
                row[:n_ref] = inputs[:, column]
                row[-1] = -evaluated_inputs[column]
                inequality_rows.append(row)
                inequality_bounds.append(0.0)
            for column in range(outputs.shape[1]):
                row = np.zeros(n_variables)
                row[:n_ref] = -outputs[:, column]
                inequality_rows.append(row)
                inequality_bounds.append(-evaluated_outputs[column])
            objective = np.zeros(n_variables)
            objective[-1] = 1.0
        else:
            for column in range(inputs.shape[1]):
                row = np.zeros(n_variables)
                row[:n_ref] = inputs[:, column]
                inequality_rows.append(row)
                inequality_bounds.append(evaluated_inputs[column])
            for column in range(outputs.shape[1]):
                row = np.zeros(n_variables)
                row[:n_ref] = -outputs[:, column]
                row[-1] = evaluated_outputs[column]
                inequality_rows.append(row)
                inequality_bounds.append(0.0)
            objective = np.zeros(n_variables)
            objective[-1] = -1.0

        _add_rts(
            returns_to_scale=self.returns_to_scale,
            n_lambdas=n_ref,
            n_variables=n_variables,
            inequality_rows=inequality_rows,
            inequality_bounds=inequality_bounds,
            equality_rows=equality_rows,
            equality_bounds=equality_bounds,
        )
        return self.solver.solve(
            LinearProgram(
                objective=objective,
                inequalities=_matrix_or_none(inequality_rows),
                inequality_bounds=_vector_or_none(inequality_bounds),
                equalities=_matrix_or_none(equality_rows),
                equality_bounds=_vector_or_none(equality_bounds),
                variable_bounds=[(0.0, None)] * n_variables,
            )
        )

    def _phase_two(
        self,
        *,
        inputs: NDArray[np.float64],
        outputs: NDArray[np.float64],
        evaluated_inputs: NDArray[np.float64],
        evaluated_outputs: NDArray[np.float64],
        radial_factor: float,
    ) -> LinearSolution:
        n_ref = inputs.shape[0]
        n_inputs = inputs.shape[1]
        n_outputs = outputs.shape[1]
        n_variables = n_ref + n_inputs + n_outputs
        inequality_rows, inequality_bounds = _empty_rows(n_variables)
        equality_rows, equality_bounds = _empty_rows(n_variables)

        for column in range(n_inputs):
            row = np.zeros(n_variables)
            row[:n_ref] = inputs[:, column]
            row[n_ref + column] = 1.0
            equality_rows.append(row)
            equality_bounds.append(
                radial_factor * evaluated_inputs[column]
                if self.orientation is Orientation.INPUT
                else evaluated_inputs[column]
            )
        for column in range(n_outputs):
            row = np.zeros(n_variables)
            row[:n_ref] = outputs[:, column]
            row[n_ref + n_inputs + column] = -1.0
            equality_rows.append(row)
            equality_bounds.append(
                evaluated_outputs[column]
                if self.orientation is Orientation.INPUT
                else radial_factor * evaluated_outputs[column]
            )

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
        if self.slack_strategy is SlackStrategy.MAX_SUM:
            objective[n_ref:] = -1.0
        else:
            input_scale = np.maximum(
                np.maximum(inputs.max(axis=0), evaluated_inputs),
                self.tolerance,
            )
            output_scale = np.maximum(
                np.maximum(outputs.max(axis=0), evaluated_outputs),
                self.tolerance,
            )
            objective[n_ref : n_ref + n_inputs] = -1.0 / input_scale
            objective[n_ref + n_inputs :] = -1.0 / output_scale

        return self.solver.solve(
            LinearProgram(
                objective=objective,
                inequalities=_matrix_or_none(inequality_rows),
                inequality_bounds=_vector_or_none(inequality_bounds),
                equalities=_matrix_or_none(equality_rows),
                equality_bounds=_vector_or_none(equality_bounds),
                variable_bounds=[(0.0, None)] * n_variables,
            )
        )


def dea(
    data: DEAData,
    *,
    orientation: Orientation | str = Orientation.INPUT,
    rts: ReturnsToScale | str = ReturnsToScale.VRS,
    slacks: SlackStrategy | str = SlackStrategy.NORMALIZED,
    reference: DEAData | None = None,
) -> RadialResult:
    """Fit the standard radial model with a concise functional API."""
    return RadialDEA(
        orientation=orientation,
        rts=rts,
        slacks=slacks,
    ).fit(data, reference=reference)
