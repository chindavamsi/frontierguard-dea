"""Explicit leave-one-out radial super-efficiency models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from frontierguard._data import DEAData
from frontierguard._enums import (
    Orientation,
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
from frontierguard._solver import (
    LinearProgram,
    LinearSolution,
    LinearSolver,
    ScipyHighsSolver,
)
from frontierguard.radial import RadialDEA

InfeasibilityPolicy = Literal["raise", "record"]


def _readonly(values: ArrayLike) -> NDArray[np.float64]:
    result = np.array(values, dtype=float, copy=True)
    if np.isinf(result).any():
        raise NumericalError("Super-efficiency results cannot contain infinity.")
    result.setflags(write=False)
    return result


def _json_vector(values: NDArray[np.float64]) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _json_matrix(values: NDArray[np.float64]) -> list[list[float | None]]:
    return [_json_vector(row) for row in values]


class _TrackingSolver:
    """Record the structured status returned by a delegated solver."""

    def __init__(self, solver: LinearSolver) -> None:
        self._solver = solver
        self.name = solver.name
        self.last_status: SolverStatus | None = None

    def solve(self, program: LinearProgram) -> LinearSolution:
        solution = self._solver.solve(program)
        self.last_status = solution.status
        return solution


@dataclass(frozen=True, slots=True, init=False)
class SuperEfficiencyResult:
    """Immutable leave-one-out radial super-efficiency result.

    ``super_efficiency`` always follows a larger-is-better convention. For an
    input-oriented model it equals the contraction factor ``theta``; for an
    output-oriented model it is ``1 / phi``. ``radial_factor`` retains the raw
    orientation-specific factor. Recorded infeasible fits are represented by
    NaN in numerical arrays and by ``None`` in :meth:`to_dict`.
    """

    dmu_names: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    orientation: Orientation
    returns_to_scale: ReturnsToScale
    radial_factor: NDArray[np.float64]
    super_efficiency: NDArray[np.float64]
    lambdas: NDArray[np.float64]
    input_slacks: NDArray[np.float64]
    output_slacks: NDArray[np.float64]
    input_targets: NDArray[np.float64]
    output_targets: NDArray[np.float64]
    statuses: tuple[SolverStatus, ...]
    messages: tuple[str | None, ...]
    reference_fingerprints: tuple[str | None, ...]
    audit_residual: NDArray[np.float64]
    audit_tolerance: float
    solver: str
    phase_two: bool
    infeasibility_policy: InfeasibilityPolicy
    data_fingerprint: str

    def __init__(
        self,
        *,
        dmu_names: Sequence[str],
        input_names: Sequence[str],
        output_names: Sequence[str],
        orientation: Orientation,
        returns_to_scale: ReturnsToScale,
        radial_factor: ArrayLike,
        super_efficiency: ArrayLike,
        lambdas: ArrayLike,
        input_slacks: ArrayLike,
        output_slacks: ArrayLike,
        input_targets: ArrayLike,
        output_targets: ArrayLike,
        statuses: Sequence[SolverStatus],
        messages: Sequence[str | None],
        reference_fingerprints: Sequence[str | None],
        audit_residual: ArrayLike,
        audit_tolerance: float,
        solver: str,
        phase_two: bool,
        infeasibility_policy: InfeasibilityPolicy,
        data_fingerprint: str,
    ) -> None:
        """Copy flexible values into a canonical immutable representation."""
        object.__setattr__(self, "dmu_names", tuple(dmu_names))
        object.__setattr__(self, "input_names", tuple(input_names))
        object.__setattr__(self, "output_names", tuple(output_names))
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "returns_to_scale", returns_to_scale)
        object.__setattr__(self, "radial_factor", _readonly(radial_factor))
        object.__setattr__(self, "super_efficiency", _readonly(super_efficiency))
        object.__setattr__(self, "lambdas", _readonly(lambdas))
        object.__setattr__(self, "input_slacks", _readonly(input_slacks))
        object.__setattr__(self, "output_slacks", _readonly(output_slacks))
        object.__setattr__(self, "input_targets", _readonly(input_targets))
        object.__setattr__(self, "output_targets", _readonly(output_targets))
        object.__setattr__(self, "statuses", tuple(statuses))
        object.__setattr__(self, "messages", tuple(messages))
        object.__setattr__(
            self,
            "reference_fingerprints",
            tuple(reference_fingerprints),
        )
        object.__setattr__(self, "audit_residual", _readonly(audit_residual))
        object.__setattr__(self, "audit_tolerance", float(audit_tolerance))
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "phase_two", phase_two)
        object.__setattr__(
            self,
            "infeasibility_policy",
            infeasibility_policy,
        )
        object.__setattr__(self, "data_fingerprint", data_fingerprint)

    @property
    def feasible(self) -> NDArray[np.bool_]:
        """Return a read-only mask of optimally solved DMUs."""
        result = np.asarray(
            [status is SolverStatus.OPTIMAL for status in self.statuses],
            dtype=bool,
        )
        result.setflags(write=False)
        return result

    def to_frame(self) -> pd.DataFrame:
        """Return scores and fit status by evaluated DMU."""
        return pd.DataFrame(
            {
                "dmu": self.dmu_names,
                "super_efficiency": self.super_efficiency,
                "radial_factor": self.radial_factor,
                "status": [status.value for status in self.statuses],
                "message": self.messages,
            }
        ).set_index("dmu")

    def peers(
        self,
        dmu: str | int,
        *,
        tolerance: float = 1e-8,
    ) -> dict[str, float]:
        """Return positive peer intensities for a successfully solved DMU."""
        if isinstance(dmu, str):
            try:
                position = self.dmu_names.index(dmu)
            except ValueError as exc:
                raise DataValidationError(f"Unknown DMU {dmu!r}.") from exc
        else:
            position = dmu
        if not 0 <= position < len(self.dmu_names):
            raise DataValidationError(f"DMU position {position} is out of range.")
        if self.statuses[position] is not SolverStatus.OPTIMAL:
            raise SolverError(
                f"DMU {self.dmu_names[position]!r} has no peer set because its "
                f"super-efficiency model is {self.statuses[position].value}."
            )
        return {
            name: float(weight)
            for name, weight in zip(
                self.dmu_names,
                self.lambdas[position],
                strict=True,
            )
            if weight > tolerance
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible result, using null for infeasibility."""
        return {
            "schema_version": "0.1.0",
            "model": "radial_super_efficiency",
            "orientation": self.orientation.value,
            "returns_to_scale": self.returns_to_scale.value,
            "score_convention": "larger_is_better",
            "output_radial_factor": (
                "phi_with_super_efficiency_equal_to_1_over_phi"
                if self.orientation is Orientation.OUTPUT
                else None
            ),
            "solver": self.solver,
            "phase_two": self.phase_two,
            "exclude_self": True,
            "infeasibility_policy": self.infeasibility_policy,
            "data_fingerprint": self.data_fingerprint,
            "audit_tolerance": self.audit_tolerance,
            "dmu_names": list(self.dmu_names),
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "radial_factor": _json_vector(self.radial_factor),
            "super_efficiency": _json_vector(self.super_efficiency),
            "lambdas": _json_matrix(self.lambdas),
            "input_slacks": _json_matrix(self.input_slacks),
            "output_slacks": _json_matrix(self.output_slacks),
            "input_targets": _json_matrix(self.input_targets),
            "output_targets": _json_matrix(self.output_targets),
            "audit_residual": _json_vector(self.audit_residual),
            "statuses": [status.value for status in self.statuses],
            "messages": list(self.messages),
            "reference_fingerprints": list(self.reference_fingerprints),
        }


class RadialSuperEfficiencyDEA:
    """Fit one leave-one-out radial model per DMU.

    All returns-to-scale modes supported by :class:`RadialDEA` are accepted.
    VRS and restricted-RTS super-efficiency models can be infeasible. Set
    ``on_infeasible="record"`` to retain those DMUs with an explicit status and
    missing score, or use ``"raise"`` for fail-fast behavior.
    """

    def __init__(
        self,
        *,
        orientation: Orientation | str = Orientation.INPUT,
        rts: ReturnsToScale | str = ReturnsToScale.CRS,
        slacks: SlackStrategy | str = SlackStrategy.NONE,
        on_infeasible: InfeasibilityPolicy = "raise",
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
        if on_infeasible not in ("raise", "record"):
            raise ModelValidationError(
                "on_infeasible must be either 'raise' or 'record'."
            )
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ModelValidationError("tolerance must be a positive finite number.")
        self.infeasibility_policy = on_infeasible
        self.tolerance = float(tolerance)
        self.solver = ScipyHighsSolver() if solver is None else solver

    def fit(self, data: DEAData) -> SuperEfficiencyResult:
        """Evaluate every DMU against all other DMUs in the dataset."""
        if not isinstance(data, DEAData):
            raise ModelValidationError("data must be a DEAData instance.")
        if data.n_dmus < 2:
            raise ModelValidationError(
                "Super-efficiency requires at least two DMUs after leave-one-out."
            )

        n_dmus = data.n_dmus
        radial_factors = np.full(n_dmus, np.nan)
        scores = np.full(n_dmus, np.nan)
        lambdas = np.full((n_dmus, n_dmus), np.nan)
        input_slacks = np.full((n_dmus, data.n_inputs), np.nan)
        output_slacks = np.full((n_dmus, data.n_outputs), np.nan)
        input_targets = np.full_like(input_slacks, np.nan)
        output_targets = np.full_like(output_slacks, np.nan)
        audit_residuals = np.full(n_dmus, np.nan)
        statuses: list[SolverStatus] = []
        messages: list[str | None] = []
        reference_fingerprints: list[str | None] = []

        for position, name in enumerate(data.dmu_names):
            included = np.arange(n_dmus) != position
            evaluation = DEAData(
                inputs=data.inputs[[position]],
                outputs=data.outputs[[position]],
                dmu_names=(name,),
                input_names=data.input_names,
                output_names=data.output_names,
            )
            reference = DEAData(
                inputs=data.inputs[included],
                outputs=data.outputs[included],
                dmu_names=tuple(
                    candidate
                    for index, candidate in enumerate(data.dmu_names)
                    if index != position
                ),
                input_names=data.input_names,
                output_names=data.output_names,
            )
            tracking_solver = _TrackingSolver(self.solver)
            model = RadialDEA(
                orientation=self.orientation,
                rts=self.returns_to_scale,
                slacks=self.slack_strategy,
                solver=tracking_solver,
                tolerance=self.tolerance,
            )
            try:
                result = model.fit(evaluation, reference=reference)
            except SolverError as exc:
                if tracking_solver.last_status is not SolverStatus.INFEASIBLE:
                    raise
                if self.infeasibility_policy == "raise":
                    raise SolverError(
                        f"Super-efficiency model for {name!r} is infeasible under "
                        f"{self.returns_to_scale.value.upper()} "
                        f"{self.orientation.value} orientation; set "
                        "on_infeasible='record' to retain an explicit missing score."
                    ) from exc
                statuses.append(SolverStatus.INFEASIBLE)
                messages.append(str(exc))
                reference_fingerprints.append(reference.fingerprint)
                continue

            radial_factors[position] = result.radial_factor[0]
            scores[position] = result.technical_efficiency[0]
            lambdas[position] = 0.0
            lambdas[position, included] = result.lambdas[0]
            input_slacks[position] = result.input_slacks[0]
            output_slacks[position] = result.output_slacks[0]
            input_targets[position] = result.input_targets[0]
            output_targets[position] = result.output_targets[0]
            audit_residuals[position] = result.audit_residual[0]
            statuses.append(result.statuses[0])
            messages.append(None)
            reference_fingerprints.append(reference.fingerprint)

        return SuperEfficiencyResult(
            dmu_names=data.dmu_names,
            input_names=data.input_names,
            output_names=data.output_names,
            orientation=self.orientation,
            returns_to_scale=self.returns_to_scale,
            radial_factor=radial_factors,
            super_efficiency=scores,
            lambdas=lambdas,
            input_slacks=input_slacks,
            output_slacks=output_slacks,
            input_targets=input_targets,
            output_targets=output_targets,
            statuses=tuple(statuses),
            messages=tuple(messages),
            reference_fingerprints=tuple(reference_fingerprints),
            audit_residual=audit_residuals,
            audit_tolerance=max(10.0 * self.tolerance, 1e-7),
            solver=self.solver.name,
            phase_two=self.slack_strategy is not SlackStrategy.NONE,
            infeasibility_policy=self.infeasibility_policy,
            data_fingerprint=data.fingerprint,
        )
