"""Cost and revenue efficiency under observed factor prices.

This module deliberately excludes profit efficiency. Profit ratios require a
separate normalization and boundedness contract when observed or optimal profit
is zero or negative.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from frontierguard._data import DEAData
from frontierguard._enums import (
    ReturnsToScale,
    SlackStrategy,
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
from frontierguard.radial import RadialDEA

__all__ = [
    "CostEfficiencyDEA",
    "CostEfficiencyResult",
    "RevenueEfficiencyDEA",
    "RevenueEfficiencyResult",
]


def _readonly(values: ArrayLike) -> NDArray[np.float64]:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _fingerprint(values: NDArray[np.float64]) -> str:
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(values.dtype.str.encode())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _resolve_prices(
    values: ArrayLike,
    *,
    n_dmus: int,
    n_factors: int,
    label: str,
) -> tuple[NDArray[np.float64], str]:
    try:
        prices = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{label} must be numeric.") from exc
    if prices.ndim == 0 and n_factors == 1:
        prices = prices.reshape(1)
    if prices.ndim == 1:
        if prices.shape != (n_factors,):
            raise ModelValidationError(
                f"{label} must contain exactly {n_factors} factor prices."
            )
        resolved = np.broadcast_to(prices, (n_dmus, n_factors)).copy()
        scope = "common"
    elif prices.ndim == 2:
        if prices.shape != (n_dmus, n_factors):
            raise ModelValidationError(
                f"Per-DMU {label} must have shape {(n_dmus, n_factors)}; "
                f"received {prices.shape}."
            )
        resolved = prices.copy()
        scope = "per_dmu"
    else:
        raise ModelValidationError(
            f"{label} must be a factor vector or a DMU-by-factor matrix."
        )
    if not np.isfinite(resolved).all():
        raise ModelValidationError(f"{label} must contain only finite values.")
    if (resolved < 0).any():
        raise ModelValidationError(f"{label} must be non-negative.")
    resolved.setflags(write=False)
    return resolved, scope


def _validate_data_and_reference(
    data: DEAData,
    reference: DEAData | None,
) -> DEAData:
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
    if data.input_names != reference_data.input_names:
        raise ModelValidationError(
            "Evaluation and reference input names must match exactly."
        )
    if data.output_names != reference_data.output_names:
        raise ModelValidationError(
            "Evaluation and reference output names must match exactly."
        )
    return reference_data


def _positive_observed_aggregates(
    prices: NDArray[np.float64],
    quantities: NDArray[np.float64],
    *,
    dmu_names: tuple[str, ...],
    label: str,
) -> NDArray[np.float64]:
    with np.errstate(over="ignore", invalid="ignore"):
        aggregates: NDArray[np.float64] = np.asarray(
            np.einsum("ij,ij->i", prices, quantities),
            dtype=float,
        )
    invalid = (~np.isfinite(aggregates)) | (aggregates <= 0)
    if invalid.any():
        names = ", ".join(
            name for name, failed in zip(dmu_names, invalid, strict=True) if failed
        )
        raise ModelValidationError(
            f"Observed {label} must be positive and finite for every DMU; "
            f"invalid for: {names}."
        )
    return aggregates


def _rts_constraints(
    n_lambdas: int,
    returns_to_scale: ReturnsToScale,
) -> tuple[
    list[NDArray[np.float64]],
    list[float],
    list[NDArray[np.float64]],
    list[float],
]:
    inequality_rows: list[NDArray[np.float64]] = []
    inequality_bounds: list[float] = []
    equality_rows: list[NDArray[np.float64]] = []
    equality_bounds: list[float] = []
    row = np.ones(n_lambdas, dtype=float)
    if returns_to_scale is ReturnsToScale.VRS:
        equality_rows.append(row)
        equality_bounds.append(1.0)
    elif returns_to_scale is ReturnsToScale.NIRS:
        inequality_rows.append(row)
        inequality_bounds.append(1.0)
    elif returns_to_scale is ReturnsToScale.NDRS:
        inequality_rows.append(-row)
        inequality_bounds.append(-1.0)
    return inequality_rows, inequality_bounds, equality_rows, equality_bounds


def _matrix_or_none(
    rows: list[NDArray[np.float64]],
) -> NDArray[np.float64] | None:
    return None if not rows else np.vstack(rows)


def _vector_or_none(values: list[float]) -> NDArray[np.float64] | None:
    return None if not values else np.asarray(values, dtype=float)


def _scaled_absolute(
    residual: NDArray[np.float64] | float,
    *scales: NDArray[np.float64] | float,
) -> float:
    numerator = float(np.max(np.abs(np.asarray(residual, dtype=float)), initial=0.0))
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
    return _scaled_absolute(np.maximum(np.asarray(residual), 0.0), *scales)


def _rts_audit(
    lambdas: NDArray[np.float64],
    returns_to_scale: ReturnsToScale,
) -> float:
    intensity_sum = float(lambdas.sum())
    if returns_to_scale is ReturnsToScale.VRS:
        return _scaled_absolute(intensity_sum - 1.0, intensity_sum)
    if returns_to_scale is ReturnsToScale.NIRS:
        return _scaled_positive(intensity_sum - 1.0, intensity_sum)
    if returns_to_scale is ReturnsToScale.NDRS:
        return _scaled_positive(1.0 - intensity_sum, intensity_sum)
    return 0.0


def _position(dmu_names: tuple[str, ...], dmu: str | int) -> int:
    if isinstance(dmu, str):
        try:
            return dmu_names.index(dmu)
        except ValueError as exc:
            raise DataValidationError(f"Unknown DMU {dmu!r}.") from exc
    if not 0 <= dmu < len(dmu_names):
        raise DataValidationError(f"DMU position {dmu} is out of range.")
    return dmu


def _peers(
    *,
    dmu_names: tuple[str, ...],
    reference_names: tuple[str, ...],
    lambdas: NDArray[np.float64],
    dmu: str | int,
    tolerance: float,
) -> dict[str, float]:
    position = _position(dmu_names, dmu)
    return {
        name: float(weight)
        for name, weight in zip(
            reference_names,
            lambdas[position],
            strict=True,
        )
        if weight > tolerance
    }


def _targets(
    *,
    dmu_names: tuple[str, ...],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_targets: NDArray[np.float64],
    output_targets: NDArray[np.float64],
    dmu: str | int,
) -> dict[str, dict[str, float]]:
    position = _position(dmu_names, dmu)
    return {
        "inputs": {
            name: float(value)
            for name, value in zip(
                input_names,
                input_targets[position],
                strict=True,
            )
        },
        "outputs": {
            name: float(value)
            for name, value in zip(
                output_names,
                output_targets[position],
                strict=True,
            )
        },
    }


@dataclass(frozen=True, slots=True, init=False)
class CostEfficiencyResult:
    """Immutable cost, technical, and allocative efficiency result.

    ``cost_efficiency`` is minimum attainable input cost divided by observed
    input cost. ``allocative_efficiency`` is cost efficiency divided by
    input-oriented Farrell technical efficiency.
    """

    dmu_names: tuple[str, ...]
    reference_names: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    returns_to_scale: ReturnsToScale
    input_prices: NDArray[np.float64]
    price_scope: str
    observed_cost: NDArray[np.float64]
    minimum_cost: NDArray[np.float64]
    cost_efficiency: NDArray[np.float64]
    technical_efficiency: NDArray[np.float64]
    allocative_efficiency: NDArray[np.float64]
    lambdas: NDArray[np.float64]
    input_targets: NDArray[np.float64]
    output_targets: NDArray[np.float64]
    statuses: tuple[SolverStatus, ...]
    technical_statuses: tuple[SolverStatus, ...]
    solver: str
    data_fingerprint: str
    reference_fingerprint: str
    price_fingerprint: str
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
        input_prices: ArrayLike,
        price_scope: str,
        observed_cost: ArrayLike,
        minimum_cost: ArrayLike,
        cost_efficiency: ArrayLike,
        technical_efficiency: ArrayLike,
        allocative_efficiency: ArrayLike,
        lambdas: ArrayLike,
        input_targets: ArrayLike,
        output_targets: ArrayLike,
        statuses: Sequence[SolverStatus],
        technical_statuses: Sequence[SolverStatus],
        solver: str,
        data_fingerprint: str,
        reference_fingerprint: str,
        price_fingerprint: str,
        audit_residual: ArrayLike,
        audit_tolerance: float,
    ) -> None:
        object.__setattr__(self, "dmu_names", tuple(dmu_names))
        object.__setattr__(self, "reference_names", tuple(reference_names))
        object.__setattr__(self, "input_names", tuple(input_names))
        object.__setattr__(self, "output_names", tuple(output_names))
        object.__setattr__(self, "returns_to_scale", returns_to_scale)
        object.__setattr__(self, "input_prices", _readonly(input_prices))
        object.__setattr__(self, "price_scope", price_scope)
        object.__setattr__(self, "observed_cost", _readonly(observed_cost))
        object.__setattr__(self, "minimum_cost", _readonly(minimum_cost))
        object.__setattr__(self, "cost_efficiency", _readonly(cost_efficiency))
        object.__setattr__(
            self,
            "technical_efficiency",
            _readonly(technical_efficiency),
        )
        object.__setattr__(
            self,
            "allocative_efficiency",
            _readonly(allocative_efficiency),
        )
        object.__setattr__(self, "lambdas", _readonly(lambdas))
        object.__setattr__(self, "input_targets", _readonly(input_targets))
        object.__setattr__(self, "output_targets", _readonly(output_targets))
        object.__setattr__(self, "statuses", tuple(statuses))
        object.__setattr__(self, "technical_statuses", tuple(technical_statuses))
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "data_fingerprint", data_fingerprint)
        object.__setattr__(
            self,
            "reference_fingerprint",
            reference_fingerprint,
        )
        object.__setattr__(self, "price_fingerprint", price_fingerprint)
        object.__setattr__(self, "audit_residual", _readonly(audit_residual))
        object.__setattr__(self, "audit_tolerance", float(audit_tolerance))

    @property
    def economic_efficiency(self) -> NDArray[np.float64]:
        """Alias for the model-specific cost efficiency score."""
        return self.cost_efficiency

    @property
    def score_convention(self) -> str:
        """Human-readable numerator and denominator convention."""
        return "minimum_cost / observed_cost"

    def peers(
        self,
        dmu: str | int,
        *,
        tolerance: float = 1e-8,
    ) -> dict[str, float]:
        """Return positive cost-minimizing reference intensities."""
        return _peers(
            dmu_names=self.dmu_names,
            reference_names=self.reference_names,
            lambdas=self.lambdas,
            dmu=dmu,
            tolerance=tolerance,
        )

    def targets(self, dmu: str | int) -> dict[str, dict[str, float]]:
        """Return the cost-minimizing input and associated output target."""
        return _targets(
            dmu_names=self.dmu_names,
            input_names=self.input_names,
            output_names=self.output_names,
            input_targets=self.input_targets,
            output_targets=self.output_targets,
            dmu=dmu,
        )

    def to_frame(self) -> pd.DataFrame:
        """Return primary cost-efficiency quantities in a tidy table."""
        return pd.DataFrame(
            {
                "dmu": self.dmu_names,
                "observed_cost": self.observed_cost,
                "minimum_cost": self.minimum_cost,
                "cost_efficiency": self.cost_efficiency,
                "technical_efficiency": self.technical_efficiency,
                "allocative_efficiency": self.allocative_efficiency,
                "audit_residual": self.audit_residual,
                "status": [status.value for status in self.statuses],
            }
        ).set_index("dmu")

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible result and provenance record."""
        return {
            "schema_version": "0.1.0",
            "model": "cost_efficiency",
            "score_convention": self.score_convention,
            "returns_to_scale": self.returns_to_scale.value,
            "solver": self.solver,
            "price_scope": self.price_scope,
            "data_fingerprint": self.data_fingerprint,
            "reference_fingerprint": self.reference_fingerprint,
            "price_fingerprint": self.price_fingerprint,
            "technical_model": "farrell_radial_input_envelopment",
            "audit_residual": self.audit_residual.tolist(),
            "audit_tolerance": self.audit_tolerance,
            "dmu_names": list(self.dmu_names),
            "reference_names": list(self.reference_names),
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "input_prices": self.input_prices.tolist(),
            "observed_cost": self.observed_cost.tolist(),
            "minimum_cost": self.minimum_cost.tolist(),
            "cost_efficiency": self.cost_efficiency.tolist(),
            "technical_efficiency": self.technical_efficiency.tolist(),
            "allocative_efficiency": self.allocative_efficiency.tolist(),
            "lambdas": self.lambdas.tolist(),
            "input_targets": self.input_targets.tolist(),
            "output_targets": self.output_targets.tolist(),
            "statuses": [status.value for status in self.statuses],
            "technical_statuses": [status.value for status in self.technical_statuses],
        }


@dataclass(frozen=True, slots=True, init=False)
class RevenueEfficiencyResult:
    """Immutable revenue, technical, and allocative efficiency result.

    ``revenue_efficiency`` is observed output revenue divided by maximum
    attainable output revenue. ``allocative_efficiency`` is revenue efficiency
    divided by output-oriented Farrell technical efficiency.
    """

    dmu_names: tuple[str, ...]
    reference_names: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    returns_to_scale: ReturnsToScale
    output_prices: NDArray[np.float64]
    price_scope: str
    observed_revenue: NDArray[np.float64]
    maximum_revenue: NDArray[np.float64]
    revenue_efficiency: NDArray[np.float64]
    technical_efficiency: NDArray[np.float64]
    allocative_efficiency: NDArray[np.float64]
    lambdas: NDArray[np.float64]
    input_targets: NDArray[np.float64]
    output_targets: NDArray[np.float64]
    statuses: tuple[SolverStatus, ...]
    technical_statuses: tuple[SolverStatus, ...]
    solver: str
    data_fingerprint: str
    reference_fingerprint: str
    price_fingerprint: str
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
        output_prices: ArrayLike,
        price_scope: str,
        observed_revenue: ArrayLike,
        maximum_revenue: ArrayLike,
        revenue_efficiency: ArrayLike,
        technical_efficiency: ArrayLike,
        allocative_efficiency: ArrayLike,
        lambdas: ArrayLike,
        input_targets: ArrayLike,
        output_targets: ArrayLike,
        statuses: Sequence[SolverStatus],
        technical_statuses: Sequence[SolverStatus],
        solver: str,
        data_fingerprint: str,
        reference_fingerprint: str,
        price_fingerprint: str,
        audit_residual: ArrayLike,
        audit_tolerance: float,
    ) -> None:
        object.__setattr__(self, "dmu_names", tuple(dmu_names))
        object.__setattr__(self, "reference_names", tuple(reference_names))
        object.__setattr__(self, "input_names", tuple(input_names))
        object.__setattr__(self, "output_names", tuple(output_names))
        object.__setattr__(self, "returns_to_scale", returns_to_scale)
        object.__setattr__(self, "output_prices", _readonly(output_prices))
        object.__setattr__(self, "price_scope", price_scope)
        object.__setattr__(self, "observed_revenue", _readonly(observed_revenue))
        object.__setattr__(self, "maximum_revenue", _readonly(maximum_revenue))
        object.__setattr__(
            self,
            "revenue_efficiency",
            _readonly(revenue_efficiency),
        )
        object.__setattr__(
            self,
            "technical_efficiency",
            _readonly(technical_efficiency),
        )
        object.__setattr__(
            self,
            "allocative_efficiency",
            _readonly(allocative_efficiency),
        )
        object.__setattr__(self, "lambdas", _readonly(lambdas))
        object.__setattr__(self, "input_targets", _readonly(input_targets))
        object.__setattr__(self, "output_targets", _readonly(output_targets))
        object.__setattr__(self, "statuses", tuple(statuses))
        object.__setattr__(self, "technical_statuses", tuple(technical_statuses))
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "data_fingerprint", data_fingerprint)
        object.__setattr__(
            self,
            "reference_fingerprint",
            reference_fingerprint,
        )
        object.__setattr__(self, "price_fingerprint", price_fingerprint)
        object.__setattr__(self, "audit_residual", _readonly(audit_residual))
        object.__setattr__(self, "audit_tolerance", float(audit_tolerance))

    @property
    def economic_efficiency(self) -> NDArray[np.float64]:
        """Alias for the model-specific revenue efficiency score."""
        return self.revenue_efficiency

    @property
    def score_convention(self) -> str:
        """Human-readable numerator and denominator convention."""
        return "observed_revenue / maximum_revenue"

    def peers(
        self,
        dmu: str | int,
        *,
        tolerance: float = 1e-8,
    ) -> dict[str, float]:
        """Return positive revenue-maximizing reference intensities."""
        return _peers(
            dmu_names=self.dmu_names,
            reference_names=self.reference_names,
            lambdas=self.lambdas,
            dmu=dmu,
            tolerance=tolerance,
        )

    def targets(self, dmu: str | int) -> dict[str, dict[str, float]]:
        """Return the resource-feasible input and revenue-maximizing output target."""
        return _targets(
            dmu_names=self.dmu_names,
            input_names=self.input_names,
            output_names=self.output_names,
            input_targets=self.input_targets,
            output_targets=self.output_targets,
            dmu=dmu,
        )

    def to_frame(self) -> pd.DataFrame:
        """Return primary revenue-efficiency quantities in a tidy table."""
        return pd.DataFrame(
            {
                "dmu": self.dmu_names,
                "observed_revenue": self.observed_revenue,
                "maximum_revenue": self.maximum_revenue,
                "revenue_efficiency": self.revenue_efficiency,
                "technical_efficiency": self.technical_efficiency,
                "allocative_efficiency": self.allocative_efficiency,
                "audit_residual": self.audit_residual,
                "status": [status.value for status in self.statuses],
            }
        ).set_index("dmu")

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible result and provenance record."""
        return {
            "schema_version": "0.1.0",
            "model": "revenue_efficiency",
            "score_convention": self.score_convention,
            "returns_to_scale": self.returns_to_scale.value,
            "solver": self.solver,
            "price_scope": self.price_scope,
            "data_fingerprint": self.data_fingerprint,
            "reference_fingerprint": self.reference_fingerprint,
            "price_fingerprint": self.price_fingerprint,
            "technical_model": "farrell_radial_output_envelopment",
            "audit_residual": self.audit_residual.tolist(),
            "audit_tolerance": self.audit_tolerance,
            "dmu_names": list(self.dmu_names),
            "reference_names": list(self.reference_names),
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "output_prices": self.output_prices.tolist(),
            "observed_revenue": self.observed_revenue.tolist(),
            "maximum_revenue": self.maximum_revenue.tolist(),
            "revenue_efficiency": self.revenue_efficiency.tolist(),
            "technical_efficiency": self.technical_efficiency.tolist(),
            "allocative_efficiency": self.allocative_efficiency.tolist(),
            "lambdas": self.lambdas.tolist(),
            "input_targets": self.input_targets.tolist(),
            "output_targets": self.output_targets.tolist(),
            "statuses": [status.value for status in self.statuses],
            "technical_statuses": [status.value for status in self.technical_statuses],
        }


class CostEfficiencyDEA:
    """Minimize input expenditure while producing each evaluated output vector."""

    def __init__(
        self,
        input_prices: ArrayLike,
        *,
        rts: ReturnsToScale | str = ReturnsToScale.VRS,
        solver: LinearSolver | None = None,
        tolerance: float = 1e-9,
    ) -> None:
        self.input_prices = input_prices
        self.returns_to_scale = coerce_enum(rts, ReturnsToScale, label="rts")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ModelValidationError("tolerance must be a positive finite number.")
        self.tolerance = float(tolerance)
        self.solver = ScipyHighsSolver() if solver is None else solver

    def fit(
        self,
        data: DEAData,
        *,
        reference: DEAData | None = None,
    ) -> CostEfficiencyResult:
        """Fit price-specific cost frontiers for every evaluated DMU."""
        reference_data = _validate_data_and_reference(data, reference)
        prices, price_scope = _resolve_prices(
            self.input_prices,
            n_dmus=data.n_dmus,
            n_factors=data.n_inputs,
            label="input_prices",
        )
        observed_cost = _positive_observed_aggregates(
            prices,
            data.inputs,
            dmu_names=data.dmu_names,
            label="input cost",
        )

        n_eval = data.n_dmus
        n_ref = reference_data.n_dmus
        minimum_cost = np.empty(n_eval)
        lambdas = np.empty((n_eval, n_ref))
        input_targets = np.empty((n_eval, data.n_inputs))
        output_targets = np.empty((n_eval, data.n_outputs))
        solver_objectives = np.empty(n_eval)
        statuses: list[SolverStatus] = []

        for position in range(n_eval):
            (
                minimum_cost[position],
                lambdas[position],
                input_targets[position],
                output_targets[position],
                solver_objectives[position],
                status,
            ) = self._fit_one(
                inputs=reference_data.inputs,
                outputs=reference_data.outputs,
                evaluated_outputs=data.outputs[position],
                prices=prices[position],
                label=data.dmu_names[position],
            )
            statuses.append(status)

        cost_efficiency = minimum_cost / observed_cost
        technical = RadialDEA(
            orientation="input",
            rts=self.returns_to_scale,
            slacks=SlackStrategy.NONE,
            solver=self.solver,
            tolerance=self.tolerance,
        ).fit(data, reference=reference_data)
        if (technical.technical_efficiency <= 0).any():
            raise NumericalError(
                "Input-oriented technical efficiency must be positive to "
                "define allocative efficiency."
            )
        allocative_efficiency = cost_efficiency / technical.technical_efficiency
        audit_tolerance = max(10.0 * self.tolerance, 1e-7)
        audit_residual = np.empty(n_eval)
        for position in range(n_eval):
            audit_residual[position] = max(
                float(technical.audit_residual[position]),
                self._audit_one(
                    inputs=reference_data.inputs,
                    outputs=reference_data.outputs,
                    evaluated_inputs=data.inputs[position],
                    evaluated_outputs=data.outputs[position],
                    prices=prices[position],
                    observed_cost=float(observed_cost[position]),
                    minimum_cost=float(minimum_cost[position]),
                    cost_efficiency=float(cost_efficiency[position]),
                    technical_efficiency=float(
                        technical.technical_efficiency[position]
                    ),
                    allocative_efficiency=float(allocative_efficiency[position]),
                    lambdas=lambdas[position],
                    input_target=input_targets[position],
                    output_target=output_targets[position],
                    solver_objective=float(solver_objectives[position]),
                ),
            )
            if audit_residual[position] > audit_tolerance:
                raise NumericalError(
                    f"Post-solve cost audit for {data.dmu_names[position]!r} "
                    f"failed with scaled residual {audit_residual[position]:.3e}."
                )

        arrays = (
            minimum_cost,
            cost_efficiency,
            technical.technical_efficiency,
            allocative_efficiency,
            lambdas,
            input_targets,
            output_targets,
            audit_residual,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise NumericalError(
                "The completed cost result contains non-finite values."
            )
        return CostEfficiencyResult(
            dmu_names=data.dmu_names,
            reference_names=reference_data.dmu_names,
            input_names=data.input_names,
            output_names=data.output_names,
            returns_to_scale=self.returns_to_scale,
            input_prices=prices,
            price_scope=price_scope,
            observed_cost=observed_cost,
            minimum_cost=minimum_cost,
            cost_efficiency=cost_efficiency,
            technical_efficiency=technical.technical_efficiency,
            allocative_efficiency=allocative_efficiency,
            lambdas=lambdas,
            input_targets=input_targets,
            output_targets=output_targets,
            statuses=statuses,
            technical_statuses=technical.statuses,
            solver=self.solver.name,
            data_fingerprint=data.fingerprint,
            reference_fingerprint=reference_data.fingerprint,
            price_fingerprint=_fingerprint(prices),
            audit_residual=audit_residual,
            audit_tolerance=audit_tolerance,
        )

    def _fit_one(
        self,
        *,
        inputs: NDArray[np.float64],
        outputs: NDArray[np.float64],
        evaluated_outputs: NDArray[np.float64],
        prices: NDArray[np.float64],
        label: str,
    ) -> tuple[
        float,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
        SolverStatus,
    ]:
        objective = inputs @ prices
        if not np.isfinite(objective).all():
            raise ModelValidationError(
                f"Input prices produce non-finite reference costs for {label!r}."
            )
        (
            inequality_rows,
            inequality_bounds,
            equality_rows,
            equality_bounds,
        ) = _rts_constraints(inputs.shape[0], self.returns_to_scale)
        inequality_rows.extend(
            -outputs[:, column] for column in range(outputs.shape[1])
        )
        inequality_bounds.extend(-float(value) for value in evaluated_outputs)
        solution = self.solver.solve(
            LinearProgram(
                objective=objective,
                inequalities=_matrix_or_none(inequality_rows),
                inequality_bounds=_vector_or_none(inequality_bounds),
                equalities=_matrix_or_none(equality_rows),
                equality_bounds=_vector_or_none(equality_bounds),
            )
        )
        if solution.status is not SolverStatus.OPTIMAL:
            raise SolverError(
                f"Cost model for {label!r} failed with {solution.status.value}: "
                f"{solution.message}"
            )
        intensities = np.asarray(solution.variables, dtype=float).copy()
        intensities[np.abs(intensities) <= self.tolerance] = 0.0
        input_target = inputs.T @ intensities
        output_target = outputs.T @ intensities
        minimum_cost = float(prices @ input_target)
        if minimum_cost < -self.tolerance:
            raise NumericalError(
                f"Minimum cost for {label!r} is unexpectedly negative."
            )
        minimum_cost = max(minimum_cost, 0.0)
        return (
            minimum_cost,
            intensities,
            input_target,
            output_target,
            float(solution.objective),
            solution.status,
        )

    def _audit_one(
        self,
        *,
        inputs: NDArray[np.float64],
        outputs: NDArray[np.float64],
        evaluated_inputs: NDArray[np.float64],
        evaluated_outputs: NDArray[np.float64],
        prices: NDArray[np.float64],
        observed_cost: float,
        minimum_cost: float,
        cost_efficiency: float,
        technical_efficiency: float,
        allocative_efficiency: float,
        lambdas: NDArray[np.float64],
        input_target: NDArray[np.float64],
        output_target: NDArray[np.float64],
        solver_objective: float,
    ) -> float:
        reconstructed_inputs = inputs.T @ lambdas
        reconstructed_outputs = outputs.T @ lambdas
        residuals = [
            _scaled_positive(-lambdas, lambdas),
            _scaled_absolute(
                input_target - reconstructed_inputs,
                input_target,
                reconstructed_inputs,
            ),
            _scaled_absolute(
                output_target - reconstructed_outputs,
                output_target,
                reconstructed_outputs,
            ),
            _scaled_positive(
                evaluated_outputs - output_target,
                evaluated_outputs,
                output_target,
            ),
            _scaled_absolute(
                observed_cost - float(prices @ evaluated_inputs),
                observed_cost,
            ),
            _scaled_absolute(
                minimum_cost - float(prices @ input_target),
                minimum_cost,
            ),
            _scaled_absolute(
                solver_objective - minimum_cost,
                solver_objective,
                minimum_cost,
            ),
            _scaled_absolute(
                cost_efficiency - minimum_cost / observed_cost,
                cost_efficiency,
            ),
            _scaled_absolute(
                cost_efficiency - technical_efficiency * allocative_efficiency,
                cost_efficiency,
            ),
            _rts_audit(lambdas, self.returns_to_scale),
        ]
        return max(residuals)


class RevenueEfficiencyDEA:
    """Maximize output revenue within each evaluated input vector."""

    def __init__(
        self,
        output_prices: ArrayLike,
        *,
        rts: ReturnsToScale | str = ReturnsToScale.VRS,
        solver: LinearSolver | None = None,
        tolerance: float = 1e-9,
    ) -> None:
        self.output_prices = output_prices
        self.returns_to_scale = coerce_enum(rts, ReturnsToScale, label="rts")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ModelValidationError("tolerance must be a positive finite number.")
        self.tolerance = float(tolerance)
        self.solver = ScipyHighsSolver() if solver is None else solver

    def fit(
        self,
        data: DEAData,
        *,
        reference: DEAData | None = None,
    ) -> RevenueEfficiencyResult:
        """Fit price-specific revenue frontiers for every evaluated DMU."""
        reference_data = _validate_data_and_reference(data, reference)
        prices, price_scope = _resolve_prices(
            self.output_prices,
            n_dmus=data.n_dmus,
            n_factors=data.n_outputs,
            label="output_prices",
        )
        observed_revenue = _positive_observed_aggregates(
            prices,
            data.outputs,
            dmu_names=data.dmu_names,
            label="output revenue",
        )

        n_eval = data.n_dmus
        n_ref = reference_data.n_dmus
        maximum_revenue = np.empty(n_eval)
        lambdas = np.empty((n_eval, n_ref))
        input_targets = np.empty((n_eval, data.n_inputs))
        output_targets = np.empty((n_eval, data.n_outputs))
        solver_objectives = np.empty(n_eval)
        statuses: list[SolverStatus] = []

        for position in range(n_eval):
            (
                maximum_revenue[position],
                lambdas[position],
                input_targets[position],
                output_targets[position],
                solver_objectives[position],
                status,
            ) = self._fit_one(
                inputs=reference_data.inputs,
                outputs=reference_data.outputs,
                evaluated_inputs=data.inputs[position],
                prices=prices[position],
                label=data.dmu_names[position],
            )
            statuses.append(status)

        if (maximum_revenue <= 0).any():
            names = ", ".join(
                name
                for name, invalid in zip(
                    data.dmu_names,
                    maximum_revenue <= 0,
                    strict=True,
                )
                if invalid
            )
            raise NumericalError(
                "Maximum attainable revenue must be positive to define revenue "
                f"efficiency; invalid for: {names}."
            )
        revenue_efficiency = observed_revenue / maximum_revenue
        technical = RadialDEA(
            orientation="output",
            rts=self.returns_to_scale,
            slacks=SlackStrategy.NONE,
            solver=self.solver,
            tolerance=self.tolerance,
        ).fit(data, reference=reference_data)
        if (technical.technical_efficiency <= 0).any():
            raise NumericalError(
                "Output-oriented technical efficiency must be positive to "
                "define allocative efficiency."
            )
        allocative_efficiency = revenue_efficiency / technical.technical_efficiency
        audit_tolerance = max(10.0 * self.tolerance, 1e-7)
        audit_residual = np.empty(n_eval)
        for position in range(n_eval):
            audit_residual[position] = max(
                float(technical.audit_residual[position]),
                self._audit_one(
                    inputs=reference_data.inputs,
                    outputs=reference_data.outputs,
                    evaluated_inputs=data.inputs[position],
                    evaluated_outputs=data.outputs[position],
                    prices=prices[position],
                    observed_revenue=float(observed_revenue[position]),
                    maximum_revenue=float(maximum_revenue[position]),
                    revenue_efficiency=float(revenue_efficiency[position]),
                    technical_efficiency=float(
                        technical.technical_efficiency[position]
                    ),
                    allocative_efficiency=float(allocative_efficiency[position]),
                    lambdas=lambdas[position],
                    input_target=input_targets[position],
                    output_target=output_targets[position],
                    solver_objective=float(solver_objectives[position]),
                ),
            )
            if audit_residual[position] > audit_tolerance:
                raise NumericalError(
                    f"Post-solve revenue audit for {data.dmu_names[position]!r} "
                    f"failed with scaled residual {audit_residual[position]:.3e}."
                )

        arrays = (
            maximum_revenue,
            revenue_efficiency,
            technical.technical_efficiency,
            allocative_efficiency,
            lambdas,
            input_targets,
            output_targets,
            audit_residual,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise NumericalError(
                "The completed revenue result contains non-finite values."
            )
        return RevenueEfficiencyResult(
            dmu_names=data.dmu_names,
            reference_names=reference_data.dmu_names,
            input_names=data.input_names,
            output_names=data.output_names,
            returns_to_scale=self.returns_to_scale,
            output_prices=prices,
            price_scope=price_scope,
            observed_revenue=observed_revenue,
            maximum_revenue=maximum_revenue,
            revenue_efficiency=revenue_efficiency,
            technical_efficiency=technical.technical_efficiency,
            allocative_efficiency=allocative_efficiency,
            lambdas=lambdas,
            input_targets=input_targets,
            output_targets=output_targets,
            statuses=statuses,
            technical_statuses=technical.statuses,
            solver=self.solver.name,
            data_fingerprint=data.fingerprint,
            reference_fingerprint=reference_data.fingerprint,
            price_fingerprint=_fingerprint(prices),
            audit_residual=audit_residual,
            audit_tolerance=audit_tolerance,
        )

    def _fit_one(
        self,
        *,
        inputs: NDArray[np.float64],
        outputs: NDArray[np.float64],
        evaluated_inputs: NDArray[np.float64],
        prices: NDArray[np.float64],
        label: str,
    ) -> tuple[
        float,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
        SolverStatus,
    ]:
        objective = -(outputs @ prices)
        if not np.isfinite(objective).all():
            raise ModelValidationError(
                f"Output prices produce non-finite reference revenues for {label!r}."
            )
        (
            inequality_rows,
            inequality_bounds,
            equality_rows,
            equality_bounds,
        ) = _rts_constraints(inputs.shape[0], self.returns_to_scale)
        inequality_rows.extend(inputs[:, column] for column in range(inputs.shape[1]))
        inequality_bounds.extend(float(value) for value in evaluated_inputs)
        solution = self.solver.solve(
            LinearProgram(
                objective=objective,
                inequalities=_matrix_or_none(inequality_rows),
                inequality_bounds=_vector_or_none(inequality_bounds),
                equalities=_matrix_or_none(equality_rows),
                equality_bounds=_vector_or_none(equality_bounds),
            )
        )
        if solution.status is not SolverStatus.OPTIMAL:
            raise SolverError(
                f"Revenue model for {label!r} failed with {solution.status.value}: "
                f"{solution.message}"
            )
        intensities = np.asarray(solution.variables, dtype=float).copy()
        intensities[np.abs(intensities) <= self.tolerance] = 0.0
        input_target = inputs.T @ intensities
        output_target = outputs.T @ intensities
        maximum_revenue = float(prices @ output_target)
        if maximum_revenue < -self.tolerance:
            raise NumericalError(
                f"Maximum revenue for {label!r} is unexpectedly negative."
            )
        maximum_revenue = max(maximum_revenue, 0.0)
        return (
            maximum_revenue,
            intensities,
            input_target,
            output_target,
            float(solution.objective),
            solution.status,
        )

    def _audit_one(
        self,
        *,
        inputs: NDArray[np.float64],
        outputs: NDArray[np.float64],
        evaluated_inputs: NDArray[np.float64],
        evaluated_outputs: NDArray[np.float64],
        prices: NDArray[np.float64],
        observed_revenue: float,
        maximum_revenue: float,
        revenue_efficiency: float,
        technical_efficiency: float,
        allocative_efficiency: float,
        lambdas: NDArray[np.float64],
        input_target: NDArray[np.float64],
        output_target: NDArray[np.float64],
        solver_objective: float,
    ) -> float:
        reconstructed_inputs = inputs.T @ lambdas
        reconstructed_outputs = outputs.T @ lambdas
        residuals = [
            _scaled_positive(-lambdas, lambdas),
            _scaled_absolute(
                input_target - reconstructed_inputs,
                input_target,
                reconstructed_inputs,
            ),
            _scaled_absolute(
                output_target - reconstructed_outputs,
                output_target,
                reconstructed_outputs,
            ),
            _scaled_positive(
                input_target - evaluated_inputs,
                input_target,
                evaluated_inputs,
            ),
            _scaled_absolute(
                observed_revenue - float(prices @ evaluated_outputs),
                observed_revenue,
            ),
            _scaled_absolute(
                maximum_revenue - float(prices @ output_target),
                maximum_revenue,
            ),
            _scaled_absolute(
                solver_objective + maximum_revenue,
                solver_objective,
                maximum_revenue,
            ),
            _scaled_absolute(
                revenue_efficiency - observed_revenue / maximum_revenue,
                revenue_efficiency,
            ),
            _scaled_absolute(
                revenue_efficiency - technical_efficiency * allocative_efficiency,
                revenue_efficiency,
            ),
            _rts_audit(lambdas, self.returns_to_scale),
        ]
        return max(residuals)
