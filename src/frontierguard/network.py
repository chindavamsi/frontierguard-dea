"""Relational multi-stage network DEA."""

from __future__ import annotations

import numpy as np

from frontierguard._audit import audit_serial_multiplier_solution
from frontierguard._data import SerialNetworkData
from frontierguard._enums import SolverStatus
from frontierguard._errors import ModelValidationError, NumericalError, SolverError
from frontierguard._results import SerialNetworkResult
from frontierguard._solver import LinearProgram, LinearSolver, ScipyHighsSolver


class SerialNetworkDEA:
    """CRS relational DEA for a serial network with any number of stages.

    The model uses one multiplier system across the entire network. Overall
    efficiency is the product of the reported stage efficiencies.
    """

    def __init__(
        self,
        *,
        weight_floor: float = 0.0,
        solver: LinearSolver | None = None,
        tolerance: float = 1e-8,
    ) -> None:
        if not np.isfinite(weight_floor) or weight_floor < 0:
            raise ModelValidationError(
                "weight_floor must be a non-negative finite number."
            )
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ModelValidationError("tolerance must be a positive finite number.")
        self.weight_floor = float(weight_floor)
        self.tolerance = float(tolerance)
        self.solver = ScipyHighsSolver() if solver is None else solver

    def fit(self, data: SerialNetworkData) -> SerialNetworkResult:
        """Fit the centralized relational network model for every DMU."""
        if not isinstance(data, SerialNetworkData):
            raise ModelValidationError("data must be a SerialNetworkData instance.")

        inputs = np.asarray(data.inputs)
        links = tuple(np.asarray(link) for link in data.links)
        outputs = np.asarray(data.outputs)
        input_scale = np.maximum(inputs.mean(axis=0), self.tolerance)
        link_scale = tuple(
            np.maximum(link.mean(axis=0), self.tolerance) for link in links
        )
        output_scale = np.maximum(outputs.mean(axis=0), self.tolerance)
        scaled_inputs = inputs / input_scale
        scaled_links = tuple(
            link / scale for link, scale in zip(links, link_scale, strict=True)
        )
        scaled_outputs = outputs / output_scale

        n_dmus = data.n_dmus
        n_stages = data.n_stages
        n_inputs = inputs.shape[1]
        n_outputs = outputs.shape[1]
        link_widths = [link.shape[1] for link in links]

        input_slice = slice(0, n_inputs)
        link_slices: list[slice] = []
        cursor = n_inputs
        for width in link_widths:
            link_slices.append(slice(cursor, cursor + width))
            cursor += width
        output_slice = slice(cursor, cursor + n_outputs)
        n_variables = output_slice.stop

        overall = np.empty(n_dmus)
        stages = np.empty((n_dmus, n_stages))
        input_weights = np.empty((n_dmus, n_inputs))
        link_weights = tuple(np.empty((n_dmus, width)) for width in link_widths)
        output_weights = np.empty((n_dmus, n_outputs))
        closure = np.empty(n_dmus)
        audit_residuals = np.empty(n_dmus)
        statuses: list[SolverStatus] = []

        for evaluated in range(n_dmus):
            objective = np.zeros(n_variables)
            objective[output_slice] = -scaled_outputs[evaluated]
            equalities = np.zeros((1, n_variables))
            equalities[0, input_slice] = scaled_inputs[evaluated]

            inequalities: list[np.ndarray] = []
            for reference in range(n_dmus):
                first = np.zeros(n_variables)
                first[input_slice] = -scaled_inputs[reference]
                first[link_slices[0]] = scaled_links[0][reference]
                inequalities.append(first)

                for stage in range(1, len(links)):
                    middle = np.zeros(n_variables)
                    middle[link_slices[stage - 1]] = -scaled_links[stage - 1][reference]
                    middle[link_slices[stage]] = scaled_links[stage][reference]
                    inequalities.append(middle)

                final = np.zeros(n_variables)
                final[link_slices[-1]] = -scaled_links[-1][reference]
                final[output_slice] = scaled_outputs[reference]
                inequalities.append(final)

            solution = self.solver.solve(
                LinearProgram(
                    objective=objective,
                    inequalities=np.vstack(inequalities),
                    inequality_bounds=np.zeros(len(inequalities)),
                    equalities=equalities,
                    equality_bounds=np.ones(1),
                    variable_bounds=[(self.weight_floor, None)] * n_variables,
                )
            )
            if solution.status is not SolverStatus.OPTIMAL:
                raise SolverError(
                    f"Serial network model for {data.dmu_names[evaluated]!r} "
                    f"failed with {solution.status.value}: {solution.message}"
                )

            scaled_input_weights = solution.variables[input_slice]
            scaled_link_weights = tuple(
                solution.variables[link_slice] for link_slice in link_slices
            )
            scaled_output_weights = solution.variables[output_slice]
            denominator = float(scaled_input_weights @ scaled_inputs[evaluated])
            aggregates = [
                float(weights @ values[evaluated])
                for weights, values in zip(
                    scaled_link_weights,
                    scaled_links,
                    strict=True,
                )
            ]
            final_output = float(scaled_output_weights @ scaled_outputs[evaluated])
            numerators = [*aggregates, final_output]
            denominators = [denominator, *aggregates]
            stage_scores = np.divide(
                np.asarray(numerators),
                np.asarray(denominators),
                out=np.full(n_stages, np.nan),
                where=np.asarray(denominators) > self.tolerance,
            )
            score = final_output / denominator
            closure_error = np.nan
            if np.isfinite(stage_scores).all():
                stage_product = float(np.prod(stage_scores))
                closure_error = score - stage_product
                if not np.isclose(
                    score,
                    stage_product,
                    rtol=1e-9,
                    atol=1e-11,
                ):
                    raise NumericalError(
                        f"Stage efficiencies for {data.dmu_names[evaluated]!r} "
                        "do not close to overall efficiency."
                    )

            audit = audit_serial_multiplier_solution(
                inputs=scaled_inputs,
                links=scaled_links,
                outputs=scaled_outputs,
                evaluated=evaluated,
                input_weights=scaled_input_weights,
                link_weights=scaled_link_weights,
                output_weights=scaled_output_weights,
                solver_objective=solution.objective,
                overall_efficiency=score,
                stage_efficiency=stage_scores,
                tolerance=max(10.0 * self.tolerance, 1e-7),
            )
            if not audit.passed:
                raise NumericalError(
                    f"Post-solve audit for {data.dmu_names[evaluated]!r} failed "
                    f"with scaled residual {audit.max_residual:.3e} above "
                    f"{audit.tolerance:.3e}."
                )

            overall[evaluated] = score
            stages[evaluated] = stage_scores
            input_weights[evaluated] = scaled_input_weights / input_scale
            for index, weights in enumerate(scaled_link_weights):
                link_weights[index][evaluated] = weights / link_scale[index]
            output_weights[evaluated] = scaled_output_weights / output_scale
            closure[evaluated] = closure_error
            audit_residuals[evaluated] = audit.max_residual
            statuses.append(solution.status)

        if not all(
            np.isfinite(values).all()
            for values in (
                overall,
                input_weights,
                *link_weights,
                output_weights,
                audit_residuals,
            )
        ):
            raise NumericalError("The network result contains non-finite values.")
        return SerialNetworkResult(
            dmu_names=tuple(data.dmu_names),
            stage_names=tuple(data.stage_names),
            overall_efficiency=overall,
            stage_efficiency=stages,
            input_weights=input_weights,
            link_weights=link_weights,
            output_weights=output_weights,
            closure_error=closure,
            statuses=tuple(statuses),
            solver=self.solver.name,
            data_fingerprint=data.fingerprint,
            audit_residual=audit_residuals,
            audit_tolerance=max(10.0 * self.tolerance, 1e-7),
        )
