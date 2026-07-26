"""Independent mathematical audits for solved DEA programs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from frontierguard._enums import Orientation, ReturnsToScale


@dataclass(frozen=True, slots=True)
class AuditOutcome:
    """Largest scale-adjusted residual observed by an auditor."""

    max_residual: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return bool(
            np.isfinite(self.max_residual) and self.max_residual <= self.tolerance
        )


def _scaled_absolute(
    residual: NDArray[np.float64] | float,
    *scales: NDArray[np.float64] | float,
) -> float:
    residual_array = np.asarray(residual, dtype=float)
    numerator = float(np.max(np.abs(residual_array), initial=0.0))
    denominator = 1.0
    for scale in scales:
        denominator = max(
            denominator,
            1.0 + float(np.max(np.abs(np.asarray(scale, dtype=float)), initial=0.0)),
        )
    return numerator / denominator


def _scaled_positive(
    residual: NDArray[np.float64] | float,
    *scales: NDArray[np.float64] | float,
) -> float:
    return _scaled_absolute(np.maximum(np.asarray(residual, dtype=float), 0.0), *scales)


def audit_radial_solution(
    *,
    inputs: NDArray[np.float64],
    outputs: NDArray[np.float64],
    evaluated_inputs: NDArray[np.float64],
    evaluated_outputs: NDArray[np.float64],
    orientation: Orientation,
    returns_to_scale: ReturnsToScale,
    radial_factor: float,
    lambdas: NDArray[np.float64],
    input_slacks: NDArray[np.float64],
    output_slacks: NDArray[np.float64],
    input_targets: NDArray[np.float64],
    output_targets: NDArray[np.float64],
    solver_objective: float,
    phase_two: bool,
    tolerance: float,
) -> AuditOutcome:
    """Recompute radial feasibility without reusing the LP matrices."""
    projected_inputs = inputs.T @ lambdas
    projected_outputs = outputs.T @ lambdas
    residuals = [
        _scaled_positive(-lambdas, lambdas),
        _scaled_positive(-input_slacks, evaluated_inputs),
        _scaled_positive(-output_slacks, evaluated_outputs),
        _scaled_absolute(
            input_targets - projected_inputs, input_targets, projected_inputs
        ),
        _scaled_absolute(
            output_targets - projected_outputs,
            output_targets,
            projected_outputs,
        ),
    ]

    if orientation is Orientation.INPUT:
        radial_inputs = radial_factor * evaluated_inputs
        residuals.extend(
            [
                _scaled_positive(
                    projected_inputs - radial_inputs, projected_inputs, radial_inputs
                ),
                _scaled_positive(
                    evaluated_outputs - projected_outputs,
                    evaluated_outputs,
                    projected_outputs,
                ),
                _scaled_absolute(
                    solver_objective - radial_factor,
                    solver_objective,
                    radial_factor,
                ),
            ]
        )
        if phase_two:
            residuals.extend(
                [
                    _scaled_absolute(
                        projected_inputs + input_slacks - radial_inputs,
                        projected_inputs,
                        radial_inputs,
                    ),
                    _scaled_absolute(
                        projected_outputs - output_slacks - evaluated_outputs,
                        projected_outputs,
                        evaluated_outputs,
                    ),
                ]
            )
    else:
        radial_outputs = radial_factor * evaluated_outputs
        residuals.extend(
            [
                _scaled_positive(
                    projected_inputs - evaluated_inputs,
                    projected_inputs,
                    evaluated_inputs,
                ),
                _scaled_positive(
                    radial_outputs - projected_outputs,
                    radial_outputs,
                    projected_outputs,
                ),
                _scaled_absolute(
                    solver_objective + radial_factor,
                    solver_objective,
                    radial_factor,
                ),
            ]
        )
        if phase_two:
            residuals.extend(
                [
                    _scaled_absolute(
                        projected_inputs + input_slacks - evaluated_inputs,
                        projected_inputs,
                        evaluated_inputs,
                    ),
                    _scaled_absolute(
                        projected_outputs - output_slacks - radial_outputs,
                        projected_outputs,
                        radial_outputs,
                    ),
                ]
            )

    intensity_sum = float(lambdas.sum())
    if returns_to_scale is ReturnsToScale.VRS:
        residuals.append(_scaled_absolute(intensity_sum - 1.0, intensity_sum))
    elif returns_to_scale is ReturnsToScale.NIRS:
        residuals.append(_scaled_positive(intensity_sum - 1.0, intensity_sum))
    elif returns_to_scale is ReturnsToScale.NDRS:
        residuals.append(_scaled_positive(1.0 - intensity_sum, intensity_sum))

    return AuditOutcome(max(residuals, default=0.0), tolerance)


def audit_serial_multiplier_solution(
    *,
    inputs: NDArray[np.float64],
    links: tuple[NDArray[np.float64], ...],
    outputs: NDArray[np.float64],
    evaluated: int,
    input_weights: NDArray[np.float64],
    link_weights: tuple[NDArray[np.float64], ...],
    output_weights: NDArray[np.float64],
    solver_objective: float,
    overall_efficiency: float,
    stage_efficiency: NDArray[np.float64],
    tolerance: float,
) -> AuditOutcome:
    """Recompute a serial relational multiplier solution and its identities."""
    virtual_input = float(input_weights @ inputs[evaluated])
    virtual_links = tuple(
        float(weights @ values[evaluated])
        for weights, values in zip(link_weights, links, strict=True)
    )
    virtual_output = float(output_weights @ outputs[evaluated])
    residuals = [
        _scaled_positive(-input_weights, input_weights),
        _scaled_positive(-output_weights, output_weights),
        _scaled_absolute(virtual_input - 1.0, virtual_input),
        _scaled_absolute(
            solver_objective + virtual_output,
            solver_objective,
            virtual_output,
        ),
        _scaled_absolute(
            overall_efficiency - virtual_output / virtual_input,
            overall_efficiency,
            virtual_output,
        ),
    ]
    residuals.extend(_scaled_positive(-weights, weights) for weights in link_weights)

    for reference in range(inputs.shape[0]):
        stage_numerators = [
            *(
                float(weights @ values[reference])
                for weights, values in zip(
                    link_weights,
                    links,
                    strict=True,
                )
            ),
            float(output_weights @ outputs[reference]),
        ]
        stage_denominators = [
            float(input_weights @ inputs[reference]),
            *(
                float(weights @ values[reference])
                for weights, values in zip(
                    link_weights,
                    links,
                    strict=True,
                )
            ),
        ]
        residuals.extend(
            _scaled_positive(numerator - denominator, numerator, denominator)
            for numerator, denominator in zip(
                stage_numerators,
                stage_denominators,
                strict=True,
            )
        )

    if np.isfinite(stage_efficiency).all():
        selected_numerators = np.asarray([*virtual_links, virtual_output])
        selected_denominators = np.asarray([virtual_input, *virtual_links])
        residuals.extend(
            [
                _scaled_absolute(
                    stage_efficiency - selected_numerators / selected_denominators,
                    stage_efficiency,
                ),
                _scaled_absolute(
                    overall_efficiency - float(np.prod(stage_efficiency)),
                    overall_efficiency,
                ),
            ]
        )

    return AuditOutcome(max(residuals, default=0.0), tolerance)
