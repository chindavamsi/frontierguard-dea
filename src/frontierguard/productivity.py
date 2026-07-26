"""Adjacent-period Malmquist productivity measurement.

This module implements the standard four-distance, adjacent-period Malmquist
index.  It deliberately does not implement a global Malmquist index,
Malmquist--Luenberger environmental productivity, or dynamic/network DEA.
Those methods use different technologies and must not be selected by changing
an option on this estimator.

Distance convention
-------------------
For an observation from period ``a`` evaluated on period ``b``'s technology,
``distance_a_at_b`` is the normalized radial efficiency ``E_b(x_a, y_a)``:

* input orientation: the Farrell contraction factor ``theta``;
* output orientation: the reciprocal expansion factor ``1 / phi``.

An observation on its own frontier therefore has value one.  A cross-period
value can exceed one when the evaluated observation lies beyond the reference
technology.  With this convention, a Malmquist index above one denotes
productivity growth.

The CRS calculation is the conventional scale-inclusive Malmquist total-factor
productivity index.  VRS is exposed as an explicitly VRS-frontier productivity
index; it measures change relative to the variable-returns technologies and
must not be interpreted as the CRS scale-inclusive index.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from frontierguard._data import DEAData
from frontierguard._enums import Orientation, ReturnsToScale, coerce_enum
from frontierguard._errors import ModelValidationError, NumericalError, SolverError
from frontierguard._solver import LinearSolver
from frontierguard.radial import RadialDEA

__all__ = [
    "AdjacentMalmquistResult",
    "MalmquistDEA",
    "MalmquistPanelResult",
    "malmquist",
    "malmquist_panel",
]

_DISTANCE_TERMS = (
    "distance_t_at_t",
    "distance_t1_at_t",
    "distance_t_at_t1",
    "distance_t1_at_t1",
)


def _readonly_vector(
    values: ArrayLike,
    *,
    length: int,
    label: str,
    positive: bool = False,
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.shape != (length,):
        raise ModelValidationError(f"{label} must have shape ({length},).")
    if not np.isfinite(array).all():
        raise NumericalError(f"{label} contains non-finite values.")
    if positive and np.any(array <= 0):
        raise NumericalError(f"{label} must contain strictly positive values.")
    array.setflags(write=False)
    return array


def _readonly_matrix(
    values: ArrayLike,
    *,
    shape: tuple[int, ...],
    label: str,
    positive: bool = False,
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.shape != shape:
        raise ModelValidationError(f"{label} must have shape {shape}.")
    if not np.isfinite(array).all():
        raise NumericalError(f"{label} contains non-finite values.")
    if positive and np.any(array <= 0):
        raise NumericalError(f"{label} must contain strictly positive values.")
    array.setflags(write=False)
    return array


def _labels(values: Sequence[object], *, expected: int) -> tuple[str, ...]:
    labels = tuple(str(value) for value in values)
    if len(labels) != expected:
        raise ModelValidationError(
            f"period_labels must contain exactly {expected} labels."
        )
    if any(not label.strip() for label in labels):
        raise ModelValidationError("period_labels cannot contain blank labels.")
    if len(set(labels)) != len(labels):
        raise ModelValidationError("period_labels must be unique.")
    return labels


def _require_aligned(period_t: DEAData, period_t1: DEAData) -> None:
    if not isinstance(period_t, DEAData) or not isinstance(period_t1, DEAData):
        raise ModelValidationError("Every period must be a DEAData instance.")
    checks = (
        ("DMU names and order", period_t.dmu_names, period_t1.dmu_names),
        ("input names and order", period_t.input_names, period_t1.input_names),
        ("output names and order", period_t.output_names, period_t1.output_names),
    )
    for label, left, right in checks:
        if tuple(left) != tuple(right):
            raise ModelValidationError(
                f"Adjacent periods require exactly aligned {label}; "
                f"received {tuple(left)!r} and {tuple(right)!r}."
            )


def _safe_exp(values: NDArray[np.float64], *, label: str) -> NDArray[np.float64]:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.exp(values)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise NumericalError(
            f"{label} could not be represented as finite positive values."
        )
    return result


def _scaled_identity_residual(
    malmquist_index: NDArray[np.float64],
    efficiency_change: NDArray[np.float64],
    technical_change: NDArray[np.float64],
) -> NDArray[np.float64]:
    product = efficiency_change * technical_change
    scale = np.maximum.reduce(
        (
            np.ones_like(malmquist_index),
            np.abs(malmquist_index),
            np.abs(product),
        )
    )
    return np.asarray(np.abs(malmquist_index - product) / scale, dtype=float)


@dataclass(frozen=True, slots=True, init=False)
class AdjacentMalmquistResult:
    """Audited four-distance decomposition for one adjacent transition."""

    dmu_names: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    period_t: str
    period_t1: str
    transition_label: str
    orientation: Orientation
    returns_to_scale: ReturnsToScale
    distance_t_at_t: NDArray[np.float64]
    distance_t1_at_t: NDArray[np.float64]
    distance_t_at_t1: NDArray[np.float64]
    distance_t1_at_t1: NDArray[np.float64]
    efficiency_change: NDArray[np.float64]
    technical_change: NDArray[np.float64]
    malmquist_index: NDArray[np.float64]
    distance_audit_residual: NDArray[np.float64]
    decomposition_residual: NDArray[np.float64]
    audit_residual: NDArray[np.float64]
    audit_tolerance: float
    solver: str
    period_fingerprints: tuple[str, str]

    def __init__(
        self,
        *,
        dmu_names: Sequence[str],
        input_names: Sequence[str],
        output_names: Sequence[str],
        period_t: str,
        period_t1: str,
        orientation: Orientation,
        returns_to_scale: ReturnsToScale,
        distance_t_at_t: ArrayLike,
        distance_t1_at_t: ArrayLike,
        distance_t_at_t1: ArrayLike,
        distance_t1_at_t1: ArrayLike,
        efficiency_change: ArrayLike,
        technical_change: ArrayLike,
        malmquist_index: ArrayLike,
        distance_audit_residual: ArrayLike,
        decomposition_residual: ArrayLike,
        audit_tolerance: float,
        solver: str,
        period_fingerprints: Sequence[str],
    ) -> None:
        names = tuple(dmu_names)
        n_dmus = len(names)
        if not names:
            raise ModelValidationError("A Malmquist result requires at least one DMU.")
        labels = _labels((period_t, period_t1), expected=2)
        fingerprints = tuple(period_fingerprints)
        if len(fingerprints) != 2:
            raise ModelValidationError(
                "period_fingerprints must contain exactly two values."
            )
        if not np.isfinite(audit_tolerance) or audit_tolerance <= 0:
            raise ModelValidationError(
                "audit_tolerance must be a positive finite number."
            )

        object.__setattr__(self, "dmu_names", names)
        object.__setattr__(self, "input_names", tuple(input_names))
        object.__setattr__(self, "output_names", tuple(output_names))
        object.__setattr__(self, "period_t", labels[0])
        object.__setattr__(self, "period_t1", labels[1])
        object.__setattr__(self, "transition_label", f"{labels[0]}->{labels[1]}")
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "returns_to_scale", returns_to_scale)

        positive_vectors = {
            "distance_t_at_t": distance_t_at_t,
            "distance_t1_at_t": distance_t1_at_t,
            "distance_t_at_t1": distance_t_at_t1,
            "distance_t1_at_t1": distance_t1_at_t1,
            "efficiency_change": efficiency_change,
            "technical_change": technical_change,
            "malmquist_index": malmquist_index,
        }
        for label, values in positive_vectors.items():
            object.__setattr__(
                self,
                label,
                _readonly_vector(
                    values,
                    length=n_dmus,
                    label=label,
                    positive=True,
                ),
            )

        distance_audit = _readonly_matrix(
            distance_audit_residual,
            shape=(n_dmus, len(_DISTANCE_TERMS)),
            label="distance_audit_residual",
        )
        decomposition = _readonly_vector(
            decomposition_residual,
            length=n_dmus,
            label="decomposition_residual",
        )
        if np.any(distance_audit < 0) or np.any(decomposition < 0):
            raise NumericalError("Audit residuals cannot be negative.")
        aggregate = np.maximum(distance_audit.max(axis=1), decomposition)
        aggregate.setflags(write=False)
        object.__setattr__(self, "distance_audit_residual", distance_audit)
        object.__setattr__(self, "decomposition_residual", decomposition)
        object.__setattr__(self, "audit_residual", aggregate)
        object.__setattr__(self, "audit_tolerance", float(audit_tolerance))
        object.__setattr__(self, "solver", str(solver))
        object.__setattr__(self, "period_fingerprints", fingerprints)

    @property
    def ec(self) -> NDArray[np.float64]:
        """Alias for :attr:`efficiency_change`."""
        return self.efficiency_change

    @property
    def tc(self) -> NDArray[np.float64]:
        """Alias for :attr:`technical_change`."""
        return self.technical_change

    @property
    def m(self) -> NDArray[np.float64]:
        """Alias for :attr:`malmquist_index`."""
        return self.malmquist_index

    @property
    def distance_convention(self) -> str:
        """Machine-readable name of the normalized efficiency convention."""
        return "normalized_radial_efficiency"

    @property
    def distance_terms(self) -> tuple[str, ...]:
        """Four distance fields in their audit-matrix column order."""
        return _DISTANCE_TERMS

    def to_frame(self) -> pd.DataFrame:
        """Return one tidy row per DMU for this transition."""
        return pd.DataFrame(
            {
                "transition": self.transition_label,
                "from_period": self.period_t,
                "to_period": self.period_t1,
                "dmu": self.dmu_names,
                "distance_t_at_t": self.distance_t_at_t,
                "distance_t1_at_t": self.distance_t1_at_t,
                "distance_t_at_t1": self.distance_t_at_t1,
                "distance_t1_at_t1": self.distance_t1_at_t1,
                "efficiency_change": self.efficiency_change,
                "technical_change": self.technical_change,
                "malmquist_index": self.malmquist_index,
                "decomposition_residual": self.decomposition_residual,
                "audit_residual": self.audit_residual,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible adjacent-period result."""
        return {
            "schema_version": "0.1.0",
            "model": "adjacent_malmquist",
            "distance_convention": "normalized_radial_efficiency",
            "orientation": self.orientation.value,
            "returns_to_scale": self.returns_to_scale.value,
            "period_t": self.period_t,
            "period_t1": self.period_t1,
            "transition_label": self.transition_label,
            "solver": self.solver,
            "dmu_names": list(self.dmu_names),
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "period_fingerprints": list(self.period_fingerprints),
            "distance_terms": list(_DISTANCE_TERMS),
            "distance_t_at_t": self.distance_t_at_t.tolist(),
            "distance_t1_at_t": self.distance_t1_at_t.tolist(),
            "distance_t_at_t1": self.distance_t_at_t1.tolist(),
            "distance_t1_at_t1": self.distance_t1_at_t1.tolist(),
            "efficiency_change": self.efficiency_change.tolist(),
            "technical_change": self.technical_change.tolist(),
            "malmquist_index": self.malmquist_index.tolist(),
            "distance_audit_residual": self.distance_audit_residual.tolist(),
            "decomposition_residual": self.decomposition_residual.tolist(),
            "audit_residual": self.audit_residual.tolist(),
            "audit_tolerance": self.audit_tolerance,
        }


@dataclass(frozen=True, slots=True, init=False)
class MalmquistPanelResult:
    """Immutable adjacent-transition results for an aligned DEA panel."""

    period_labels: tuple[str, ...]
    transition_labels: tuple[str, ...]
    dmu_names: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    orientation: Orientation
    returns_to_scale: ReturnsToScale
    distance_t_at_t: NDArray[np.float64]
    distance_t1_at_t: NDArray[np.float64]
    distance_t_at_t1: NDArray[np.float64]
    distance_t1_at_t1: NDArray[np.float64]
    efficiency_change: NDArray[np.float64]
    technical_change: NDArray[np.float64]
    malmquist_index: NDArray[np.float64]
    distance_audit_residual: NDArray[np.float64]
    decomposition_residual: NDArray[np.float64]
    audit_residual: NDArray[np.float64]
    audit_tolerance: float
    solver: str
    period_fingerprints: tuple[str, ...]

    def __init__(
        self,
        *,
        period_labels: Sequence[str],
        dmu_names: Sequence[str],
        input_names: Sequence[str],
        output_names: Sequence[str],
        orientation: Orientation,
        returns_to_scale: ReturnsToScale,
        distance_t_at_t: ArrayLike,
        distance_t1_at_t: ArrayLike,
        distance_t_at_t1: ArrayLike,
        distance_t1_at_t1: ArrayLike,
        efficiency_change: ArrayLike,
        technical_change: ArrayLike,
        malmquist_index: ArrayLike,
        distance_audit_residual: ArrayLike,
        decomposition_residual: ArrayLike,
        audit_residual: ArrayLike,
        audit_tolerance: float,
        solver: str,
        period_fingerprints: Sequence[str],
    ) -> None:
        labels = _labels(period_labels, expected=len(period_labels))
        if len(labels) < 2:
            raise ModelValidationError(
                "A Malmquist panel requires at least two periods."
            )
        names = tuple(dmu_names)
        if not names:
            raise ModelValidationError("A Malmquist panel requires at least one DMU.")
        n_transitions = len(labels) - 1
        shape = (n_transitions, len(names))
        fingerprints = tuple(period_fingerprints)
        if len(fingerprints) != len(labels):
            raise ModelValidationError(
                "period_fingerprints must contain one value per period."
            )
        if not np.isfinite(audit_tolerance) or audit_tolerance <= 0:
            raise ModelValidationError(
                "audit_tolerance must be a positive finite number."
            )

        object.__setattr__(self, "period_labels", labels)
        object.__setattr__(
            self,
            "transition_labels",
            tuple(f"{left}->{right}" for left, right in pairwise(labels)),
        )
        object.__setattr__(self, "dmu_names", names)
        object.__setattr__(self, "input_names", tuple(input_names))
        object.__setattr__(self, "output_names", tuple(output_names))
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "returns_to_scale", returns_to_scale)

        positive_matrices = {
            "distance_t_at_t": distance_t_at_t,
            "distance_t1_at_t": distance_t1_at_t,
            "distance_t_at_t1": distance_t_at_t1,
            "distance_t1_at_t1": distance_t1_at_t1,
            "efficiency_change": efficiency_change,
            "technical_change": technical_change,
            "malmquist_index": malmquist_index,
        }
        for label, values in positive_matrices.items():
            object.__setattr__(
                self,
                label,
                _readonly_matrix(
                    values,
                    shape=shape,
                    label=label,
                    positive=True,
                ),
            )

        distance_audit = _readonly_matrix(
            distance_audit_residual,
            shape=(*shape, len(_DISTANCE_TERMS)),
            label="distance_audit_residual",
        )
        decomposition = _readonly_matrix(
            decomposition_residual,
            shape=shape,
            label="decomposition_residual",
        )
        aggregate = _readonly_matrix(
            audit_residual,
            shape=shape,
            label="audit_residual",
        )
        if (
            np.any(distance_audit < 0)
            or np.any(decomposition < 0)
            or np.any(aggregate < 0)
        ):
            raise NumericalError("Audit residuals cannot be negative.")
        expected_aggregate = np.maximum(distance_audit.max(axis=2), decomposition)
        if not np.allclose(aggregate, expected_aggregate, rtol=0.0, atol=1e-15):
            raise NumericalError(
                "Panel audit_residual does not aggregate its component audits."
            )
        object.__setattr__(self, "distance_audit_residual", distance_audit)
        object.__setattr__(self, "decomposition_residual", decomposition)
        object.__setattr__(self, "audit_residual", aggregate)
        object.__setattr__(self, "audit_tolerance", float(audit_tolerance))
        object.__setattr__(self, "solver", str(solver))
        object.__setattr__(self, "period_fingerprints", fingerprints)

    @property
    def ec(self) -> NDArray[np.float64]:
        """Alias for :attr:`efficiency_change`."""
        return self.efficiency_change

    @property
    def tc(self) -> NDArray[np.float64]:
        """Alias for :attr:`technical_change`."""
        return self.technical_change

    @property
    def m(self) -> NDArray[np.float64]:
        """Alias for :attr:`malmquist_index`."""
        return self.malmquist_index

    @property
    def distance_convention(self) -> str:
        """Machine-readable name of the normalized efficiency convention."""
        return "normalized_radial_efficiency"

    @property
    def distance_terms(self) -> tuple[str, ...]:
        """Four distance fields in their audit-matrix final-axis order."""
        return _DISTANCE_TERMS

    def to_frame(self) -> pd.DataFrame:
        """Return a tidy frame with one row per transition and DMU."""
        n_dmus = len(self.dmu_names)
        return pd.DataFrame(
            {
                "transition": np.repeat(self.transition_labels, n_dmus),
                "from_period": np.repeat(self.period_labels[:-1], n_dmus),
                "to_period": np.repeat(self.period_labels[1:], n_dmus),
                "dmu": np.tile(self.dmu_names, len(self.transition_labels)),
                "distance_t_at_t": self.distance_t_at_t.reshape(-1),
                "distance_t1_at_t": self.distance_t1_at_t.reshape(-1),
                "distance_t_at_t1": self.distance_t_at_t1.reshape(-1),
                "distance_t1_at_t1": self.distance_t1_at_t1.reshape(-1),
                "efficiency_change": self.efficiency_change.reshape(-1),
                "technical_change": self.technical_change.reshape(-1),
                "malmquist_index": self.malmquist_index.reshape(-1),
                "decomposition_residual": self.decomposition_residual.reshape(-1),
                "audit_residual": self.audit_residual.reshape(-1),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible panel result."""
        return {
            "schema_version": "0.1.0",
            "model": "adjacent_malmquist_panel",
            "distance_convention": "normalized_radial_efficiency",
            "orientation": self.orientation.value,
            "returns_to_scale": self.returns_to_scale.value,
            "period_labels": list(self.period_labels),
            "transition_labels": list(self.transition_labels),
            "solver": self.solver,
            "dmu_names": list(self.dmu_names),
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "period_fingerprints": list(self.period_fingerprints),
            "distance_terms": list(_DISTANCE_TERMS),
            "distance_t_at_t": self.distance_t_at_t.tolist(),
            "distance_t1_at_t": self.distance_t1_at_t.tolist(),
            "distance_t_at_t1": self.distance_t_at_t1.tolist(),
            "distance_t1_at_t1": self.distance_t1_at_t1.tolist(),
            "efficiency_change": self.efficiency_change.tolist(),
            "technical_change": self.technical_change.tolist(),
            "malmquist_index": self.malmquist_index.tolist(),
            "distance_audit_residual": self.distance_audit_residual.tolist(),
            "decomposition_residual": self.decomposition_residual.tolist(),
            "audit_residual": self.audit_residual.tolist(),
            "audit_tolerance": self.audit_tolerance,
        }


class MalmquistDEA:
    """Four-distance adjacent-period Malmquist estimator.

    ``rts="crs"`` is the standard scale-inclusive specification.  ``rts="vrs"``
    calculates the same four-distance decomposition on VRS technologies.
    NIRS and NDRS are intentionally rejected because their cross-period
    productivity interpretation requires a separate method contract.
    """

    def __init__(
        self,
        *,
        orientation: Orientation | str = Orientation.INPUT,
        rts: ReturnsToScale | str = ReturnsToScale.CRS,
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
        if self.returns_to_scale not in {
            ReturnsToScale.CRS,
            ReturnsToScale.VRS,
        }:
            raise ModelValidationError(
                "Adjacent Malmquist supports only 'crs' and 'vrs' technologies."
            )
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ModelValidationError("tolerance must be a positive finite number.")
        self.solver = solver
        self.tolerance = float(tolerance)

    def _radial_model(self) -> RadialDEA:
        return RadialDEA(
            orientation=self.orientation,
            rts=self.returns_to_scale,
            slacks="none",
            solver=self.solver,
            tolerance=self.tolerance,
        )

    def _distance(
        self,
        model: RadialDEA,
        *,
        evaluation: DEAData,
        reference: DEAData,
        label: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, str]:
        try:
            result = model.fit(evaluation, reference=reference)
        except SolverError as exc:
            raise SolverError(f"Malmquist term {label} failed: {exc}") from exc
        return (
            np.asarray(result.technical_efficiency, dtype=float),
            np.asarray(result.audit_residual, dtype=float),
            result.audit_tolerance,
            result.solver,
        )

    def fit(
        self,
        period_t: DEAData,
        period_t1: DEAData,
        *,
        period_labels: Sequence[object] = ("t", "t+1"),
    ) -> AdjacentMalmquistResult:
        """Fit one adjacent transition using four cross-period technologies."""
        _require_aligned(period_t, period_t1)
        labels = _labels(period_labels, expected=2)
        model = self._radial_model()
        specifications = (
            ("distance_t_at_t", period_t, period_t),
            ("distance_t1_at_t", period_t1, period_t),
            ("distance_t_at_t1", period_t, period_t1),
            ("distance_t1_at_t1", period_t1, period_t1),
        )
        distances: list[NDArray[np.float64]] = []
        audits: list[NDArray[np.float64]] = []
        tolerances: list[float] = []
        solvers: list[str] = []
        for label, evaluation, reference in specifications:
            distance, audit, audit_tolerance, solver_name = self._distance(
                model,
                evaluation=evaluation,
                reference=reference,
                label=label,
            )
            distances.append(distance)
            audits.append(audit)
            tolerances.append(audit_tolerance)
            solvers.append(solver_name)

        distance_t_at_t, distance_t1_at_t, distance_t_at_t1, distance_t1_at_t1 = (
            distances
        )
        distance_matrix = np.column_stack(distances)
        if not np.isfinite(distance_matrix).all() or np.any(distance_matrix <= 0):
            raise NumericalError(
                "Malmquist distance terms must be finite and strictly positive."
            )
        logs = np.log(distance_matrix)
        log_efficiency_change = logs[:, 3] - logs[:, 0]
        log_technical_change = 0.5 * (
            (logs[:, 1] - logs[:, 3]) + (logs[:, 0] - logs[:, 2])
        )
        log_malmquist = 0.5 * ((logs[:, 1] - logs[:, 0]) + (logs[:, 3] - logs[:, 2]))
        efficiency_change = _safe_exp(
            log_efficiency_change,
            label="efficiency_change",
        )
        technical_change = _safe_exp(
            log_technical_change,
            label="technical_change",
        )
        malmquist_index = _safe_exp(log_malmquist, label="malmquist_index")
        decomposition_residual = _scaled_identity_residual(
            malmquist_index,
            efficiency_change,
            technical_change,
        )
        audit_tolerance = max(*tolerances, 100.0 * self.tolerance, 1e-7)
        distance_audit = np.column_stack(audits)
        aggregate_audit = np.maximum(
            distance_audit.max(axis=1),
            decomposition_residual,
        )
        if np.any(aggregate_audit > audit_tolerance):
            raise NumericalError(
                "Malmquist post-solve audit failed: maximum scaled residual "
                f"{float(aggregate_audit.max()):.3e} exceeds "
                f"{audit_tolerance:.3e}."
            )
        if len(set(solvers)) != 1:
            raise NumericalError("Malmquist distance terms used different solvers.")

        return AdjacentMalmquistResult(
            dmu_names=period_t.dmu_names,
            input_names=period_t.input_names,
            output_names=period_t.output_names,
            period_t=labels[0],
            period_t1=labels[1],
            orientation=self.orientation,
            returns_to_scale=self.returns_to_scale,
            distance_t_at_t=distance_t_at_t,
            distance_t1_at_t=distance_t1_at_t,
            distance_t_at_t1=distance_t_at_t1,
            distance_t1_at_t1=distance_t1_at_t1,
            efficiency_change=efficiency_change,
            technical_change=technical_change,
            malmquist_index=malmquist_index,
            distance_audit_residual=distance_audit,
            decomposition_residual=decomposition_residual,
            audit_tolerance=audit_tolerance,
            solver=solvers[0],
            period_fingerprints=(period_t.fingerprint, period_t1.fingerprint),
        )

    def fit_panel(
        self,
        periods: Sequence[DEAData],
        *,
        period_labels: Sequence[object] | None = None,
    ) -> MalmquistPanelResult:
        """Fit every adjacent transition in an aligned sequence of periods."""
        resolved_periods = tuple(periods)
        if len(resolved_periods) < 2:
            raise ModelValidationError(
                "A Malmquist panel requires at least two periods."
            )
        labels = _labels(
            (
                tuple(f"t{index}" for index in range(len(resolved_periods)))
                if period_labels is None
                else period_labels
            ),
            expected=len(resolved_periods),
        )
        for period in resolved_periods:
            if not isinstance(period, DEAData):
                raise ModelValidationError("Every period must be a DEAData instance.")
        for left, right in pairwise(resolved_periods):
            _require_aligned(left, right)

        transitions = tuple(
            self.fit(
                left,
                right,
                period_labels=(labels[index], labels[index + 1]),
            )
            for index, (left, right) in enumerate(pairwise(resolved_periods))
        )
        return MalmquistPanelResult(
            period_labels=labels,
            dmu_names=resolved_periods[0].dmu_names,
            input_names=resolved_periods[0].input_names,
            output_names=resolved_periods[0].output_names,
            orientation=self.orientation,
            returns_to_scale=self.returns_to_scale,
            distance_t_at_t=np.stack(
                [result.distance_t_at_t for result in transitions]
            ),
            distance_t1_at_t=np.stack(
                [result.distance_t1_at_t for result in transitions]
            ),
            distance_t_at_t1=np.stack(
                [result.distance_t_at_t1 for result in transitions]
            ),
            distance_t1_at_t1=np.stack(
                [result.distance_t1_at_t1 for result in transitions]
            ),
            efficiency_change=np.stack(
                [result.efficiency_change for result in transitions]
            ),
            technical_change=np.stack(
                [result.technical_change for result in transitions]
            ),
            malmquist_index=np.stack(
                [result.malmquist_index for result in transitions]
            ),
            distance_audit_residual=np.stack(
                [result.distance_audit_residual for result in transitions]
            ),
            decomposition_residual=np.stack(
                [result.decomposition_residual for result in transitions]
            ),
            audit_residual=np.stack([result.audit_residual for result in transitions]),
            audit_tolerance=max(result.audit_tolerance for result in transitions),
            solver=transitions[0].solver,
            period_fingerprints=tuple(
                period.fingerprint for period in resolved_periods
            ),
        )


def malmquist(
    period_t: DEAData,
    period_t1: DEAData,
    *,
    orientation: Orientation | str = Orientation.INPUT,
    rts: ReturnsToScale | str = ReturnsToScale.CRS,
    solver: LinearSolver | None = None,
    tolerance: float = 1e-9,
    period_labels: Sequence[object] = ("t", "t+1"),
) -> AdjacentMalmquistResult:
    """Convenience function for one adjacent-period Malmquist transition."""
    return MalmquistDEA(
        orientation=orientation,
        rts=rts,
        solver=solver,
        tolerance=tolerance,
    ).fit(period_t, period_t1, period_labels=period_labels)


def malmquist_panel(
    periods: Sequence[DEAData],
    *,
    orientation: Orientation | str = Orientation.INPUT,
    rts: ReturnsToScale | str = ReturnsToScale.CRS,
    solver: LinearSolver | None = None,
    tolerance: float = 1e-9,
    period_labels: Sequence[object] | None = None,
) -> MalmquistPanelResult:
    """Convenience function for every adjacent transition in an aligned panel."""
    return MalmquistDEA(
        orientation=orientation,
        rts=rts,
        solver=solver,
        tolerance=tolerance,
    ).fit_panel(periods, period_labels=period_labels)
