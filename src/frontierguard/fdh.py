"""Free Disposal Hull efficiency by exact peer enumeration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from frontierguard._data import DEAData
from frontierguard._enums import Orientation, SolverStatus, coerce_enum
from frontierguard._errors import ModelValidationError, NumericalError, SolverError


def _readonly(values: ArrayLike) -> NDArray[np.float64]:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, init=False)
class FDHResult:
    """Immutable radial efficiency result for a nonconvex FDH technology."""

    dmu_names: tuple[str, ...]
    reference_names: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    orientation: Orientation
    radial_factor: NDArray[np.float64]
    technical_efficiency: NDArray[np.float64]
    peer_position: NDArray[np.int64]
    lambdas: NDArray[np.float64]
    input_slacks: NDArray[np.float64]
    output_slacks: NDArray[np.float64]
    input_targets: NDArray[np.float64]
    output_targets: NDArray[np.float64]
    audit_residual: NDArray[np.float64]
    audit_tolerance: float
    statuses: tuple[SolverStatus, ...]
    exclude_self: bool
    data_fingerprint: str
    reference_fingerprint: str

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
        peer_position: ArrayLike,
        lambdas: ArrayLike,
        input_slacks: ArrayLike,
        output_slacks: ArrayLike,
        input_targets: ArrayLike,
        output_targets: ArrayLike,
        audit_residual: ArrayLike,
        audit_tolerance: float,
        statuses: Sequence[SolverStatus],
        exclude_self: bool,
        data_fingerprint: str,
        reference_fingerprint: str,
    ) -> None:
        object.__setattr__(self, "dmu_names", tuple(dmu_names))
        object.__setattr__(self, "reference_names", tuple(reference_names))
        object.__setattr__(self, "input_names", tuple(input_names))
        object.__setattr__(self, "output_names", tuple(output_names))
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "radial_factor", _readonly(radial_factor))
        object.__setattr__(
            self, "technical_efficiency", _readonly(technical_efficiency)
        )
        peers = np.array(peer_position, dtype=np.int64, copy=True)
        peers.setflags(write=False)
        object.__setattr__(self, "peer_position", peers)
        object.__setattr__(self, "lambdas", _readonly(lambdas))
        object.__setattr__(self, "input_slacks", _readonly(input_slacks))
        object.__setattr__(self, "output_slacks", _readonly(output_slacks))
        object.__setattr__(self, "input_targets", _readonly(input_targets))
        object.__setattr__(self, "output_targets", _readonly(output_targets))
        object.__setattr__(self, "audit_residual", _readonly(audit_residual))
        object.__setattr__(self, "audit_tolerance", float(audit_tolerance))
        object.__setattr__(self, "statuses", tuple(statuses))
        object.__setattr__(self, "exclude_self", bool(exclude_self))
        object.__setattr__(self, "data_fingerprint", data_fingerprint)
        object.__setattr__(self, "reference_fingerprint", reference_fingerprint)

    def peers(self, dmu: str | int) -> dict[str, float]:
        """Return the deterministic selected FDH peer."""
        if isinstance(dmu, str):
            try:
                position = self.dmu_names.index(dmu)
            except ValueError as exc:
                raise ModelValidationError(f"Unknown DMU {dmu!r}.") from exc
        else:
            position = dmu
        if not 0 <= position < len(self.dmu_names):
            raise ModelValidationError(f"DMU position {position} is out of range.")
        peer = int(self.peer_position[position])
        return {self.reference_names[peer]: 1.0}

    def to_frame(self) -> pd.DataFrame:
        """Return scores and selected peers."""
        return pd.DataFrame(
            {
                "dmu": self.dmu_names,
                "technical_efficiency": self.technical_efficiency,
                "radial_factor": self.radial_factor,
                "peer": [
                    self.reference_names[int(position)]
                    for position in self.peer_position
                ],
                "status": [status.value for status in self.statuses],
            }
        ).set_index("dmu")

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible result."""
        return {
            "schema_version": "0.1.0",
            "model": "free_disposal_hull",
            "orientation": self.orientation.value,
            "returns_to_scale": "fdh",
            "dmu_names": list(self.dmu_names),
            "reference_names": list(self.reference_names),
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "radial_factor": self.radial_factor.tolist(),
            "technical_efficiency": self.technical_efficiency.tolist(),
            "peer_position": self.peer_position.tolist(),
            "lambdas": self.lambdas.tolist(),
            "input_slacks": self.input_slacks.tolist(),
            "output_slacks": self.output_slacks.tolist(),
            "input_targets": self.input_targets.tolist(),
            "output_targets": self.output_targets.tolist(),
            "audit_residual": self.audit_residual.tolist(),
            "audit_tolerance": self.audit_tolerance,
            "statuses": [status.value for status in self.statuses],
            "exclude_self": self.exclude_self,
            "data_fingerprint": self.data_fingerprint,
            "reference_fingerprint": self.reference_fingerprint,
        }


class FreeDisposalHullDEA:
    """Input- or output-oriented FDH with one observed reference peer."""

    def __init__(
        self,
        *,
        orientation: Orientation | str = Orientation.INPUT,
        tolerance: float = 1e-9,
    ) -> None:
        self.orientation = coerce_enum(
            orientation,
            Orientation,
            label="orientation",
        )
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ModelValidationError("tolerance must be a positive finite number.")
        self.tolerance = float(tolerance)

    def fit(
        self,
        data: DEAData,
        *,
        reference: DEAData | None = None,
        exclude_self: bool = False,
    ) -> FDHResult:
        """Enumerate feasible observed peers and select the best radial factor."""
        if not isinstance(data, DEAData):
            raise ModelValidationError("data must be a DEAData instance.")
        reference_data = data if reference is None else reference
        if not isinstance(reference_data, DEAData):
            raise ModelValidationError("reference must be a DEAData instance.")
        _validate_reference(data, reference_data)

        n_eval = data.n_dmus
        n_ref = reference_data.n_dmus
        factors = np.empty(n_eval)
        efficiencies = np.empty(n_eval)
        peers = np.empty(n_eval, dtype=np.int64)
        lambdas = np.zeros((n_eval, n_ref))
        input_slacks = np.empty((n_eval, data.n_inputs))
        output_slacks = np.empty((n_eval, data.n_outputs))
        input_targets = np.empty_like(input_slacks)
        output_targets = np.empty_like(output_slacks)
        audits = np.empty(n_eval)
        statuses: list[SolverStatus] = []
        audit_tolerance = max(10.0 * self.tolerance, 1e-8)

        for evaluated in range(n_eval):
            included = np.ones(n_ref, dtype=bool)
            if exclude_self:
                included &= np.asarray(
                    [
                        reference_name != data.dmu_names[evaluated]
                        for reference_name in reference_data.dmu_names
                    ]
                )
            candidate_positions = np.flatnonzero(included)
            if not candidate_positions.size:
                raise ModelValidationError(
                    f"DMU {data.dmu_names[evaluated]!r} has an empty reference set."
                )
            if self.orientation is Orientation.INPUT:
                peer, factor = self._input_peer(
                    reference_data.inputs,
                    reference_data.outputs,
                    data.inputs[evaluated],
                    data.outputs[evaluated],
                    candidate_positions,
                )
                efficiency = factor
                input_slack = (
                    factor * data.inputs[evaluated] - reference_data.inputs[peer]
                )
                output_slack = reference_data.outputs[peer] - data.outputs[evaluated]
            else:
                peer, factor = self._output_peer(
                    reference_data.inputs,
                    reference_data.outputs,
                    data.inputs[evaluated],
                    data.outputs[evaluated],
                    candidate_positions,
                )
                if factor <= self.tolerance:
                    raise NumericalError(
                        f"Output FDH factor for {data.dmu_names[evaluated]!r} "
                        "is not positive."
                    )
                efficiency = 1.0 / factor
                input_slack = data.inputs[evaluated] - reference_data.inputs[peer]
                output_slack = (
                    reference_data.outputs[peer] - factor * data.outputs[evaluated]
                )

            input_slack[np.abs(input_slack) <= self.tolerance] = 0.0
            output_slack[np.abs(output_slack) <= self.tolerance] = 0.0
            if (input_slack < -self.tolerance).any() or (
                output_slack < -self.tolerance
            ).any():
                raise NumericalError(
                    f"FDH projection for {data.dmu_names[evaluated]!r} is infeasible."
                )
            input_slack = np.maximum(input_slack, 0.0)
            output_slack = np.maximum(output_slack, 0.0)
            target_input = reference_data.inputs[peer]
            target_output = reference_data.outputs[peer]
            audit = self._audit(
                evaluated_inputs=data.inputs[evaluated],
                evaluated_outputs=data.outputs[evaluated],
                target_inputs=target_input,
                target_outputs=target_output,
                factor=factor,
                input_slacks=input_slack,
                output_slacks=output_slack,
            )
            if audit > audit_tolerance:
                raise NumericalError(
                    f"FDH audit for {data.dmu_names[evaluated]!r} failed with "
                    f"scaled residual {audit:.3e} above {audit_tolerance:.3e}."
                )

            factors[evaluated] = factor
            efficiencies[evaluated] = efficiency
            peers[evaluated] = peer
            lambdas[evaluated, peer] = 1.0
            input_slacks[evaluated] = input_slack
            output_slacks[evaluated] = output_slack
            input_targets[evaluated] = target_input
            output_targets[evaluated] = target_output
            audits[evaluated] = audit
            statuses.append(SolverStatus.OPTIMAL)

        return FDHResult(
            dmu_names=data.dmu_names,
            reference_names=reference_data.dmu_names,
            input_names=data.input_names,
            output_names=data.output_names,
            orientation=self.orientation,
            radial_factor=factors,
            technical_efficiency=efficiencies,
            peer_position=peers,
            lambdas=lambdas,
            input_slacks=input_slacks,
            output_slacks=output_slacks,
            input_targets=input_targets,
            output_targets=output_targets,
            audit_residual=audits,
            audit_tolerance=audit_tolerance,
            statuses=statuses,
            exclude_self=exclude_self,
            data_fingerprint=data.fingerprint,
            reference_fingerprint=reference_data.fingerprint,
        )

    def _input_peer(
        self,
        inputs: NDArray[np.float64],
        outputs: NDArray[np.float64],
        evaluated_inputs: NDArray[np.float64],
        evaluated_outputs: NDArray[np.float64],
        positions: NDArray[np.int64],
    ) -> tuple[int, float]:
        best_peer: int | None = None
        best_factor = np.inf
        positive_inputs = evaluated_inputs > self.tolerance
        for position in positions:
            peer = int(position)
            if np.any(outputs[peer] + self.tolerance < evaluated_outputs):
                continue
            if np.any(
                np.logical_and(
                    ~positive_inputs,
                    inputs[peer] > self.tolerance,
                )
            ):
                continue
            ratios = inputs[peer, positive_inputs] / evaluated_inputs[positive_inputs]
            factor = float(np.max(ratios, initial=0.0))
            if factor < best_factor - self.tolerance:
                best_peer = peer
                best_factor = factor
        if best_peer is None:
            raise SolverError(
                "No FDH reference peer can produce the evaluated outputs."
            )
        return best_peer, best_factor

    def _output_peer(
        self,
        inputs: NDArray[np.float64],
        outputs: NDArray[np.float64],
        evaluated_inputs: NDArray[np.float64],
        evaluated_outputs: NDArray[np.float64],
        positions: NDArray[np.int64],
    ) -> tuple[int, float]:
        best_peer: int | None = None
        best_factor = -np.inf
        positive_outputs = evaluated_outputs > self.tolerance
        for position in positions:
            peer = int(position)
            if np.any(inputs[peer] > evaluated_inputs + self.tolerance):
                continue
            ratios = (
                outputs[peer, positive_outputs] / evaluated_outputs[positive_outputs]
            )
            factor = float(np.min(ratios, initial=np.inf))
            if factor > best_factor + self.tolerance:
                best_peer = peer
                best_factor = factor
        if best_peer is None:
            raise SolverError("No FDH reference peer fits the evaluated inputs.")
        return best_peer, best_factor

    def _audit(
        self,
        *,
        evaluated_inputs: NDArray[np.float64],
        evaluated_outputs: NDArray[np.float64],
        target_inputs: NDArray[np.float64],
        target_outputs: NDArray[np.float64],
        factor: float,
        input_slacks: NDArray[np.float64],
        output_slacks: NDArray[np.float64],
    ) -> float:
        if self.orientation is Orientation.INPUT:
            input_residual = target_inputs + input_slacks - factor * evaluated_inputs
            output_residual = target_outputs - output_slacks - evaluated_outputs
        else:
            input_residual = target_inputs + input_slacks - evaluated_inputs
            output_residual = (
                target_outputs - output_slacks - factor * evaluated_outputs
            )
        scale = 1.0 + max(
            float(np.max(np.abs(evaluated_inputs))),
            float(np.max(np.abs(evaluated_outputs))),
            float(np.max(np.abs(target_inputs))),
            float(np.max(np.abs(target_outputs))),
        )
        nonnegative_violation = max(
            float(np.max(-input_slacks, initial=0.0)),
            float(np.max(-output_slacks, initial=0.0)),
            0.0,
        )
        return max(
            float(np.max(np.abs(input_residual), initial=0.0)) / scale,
            float(np.max(np.abs(output_residual), initial=0.0)) / scale,
            nonnegative_violation / scale,
        )


def _validate_reference(data: DEAData, reference: DEAData) -> None:
    if data.n_inputs != reference.n_inputs or data.n_outputs != reference.n_outputs:
        raise ModelValidationError(
            "Evaluation and reference data must have the same dimensions."
        )
    if data.input_names != reference.input_names:
        raise ModelValidationError(
            "Evaluation and reference input names must match exactly."
        )
    if data.output_names != reference.output_names:
        raise ModelValidationError(
            "Evaluation and reference output names must match exactly."
        )
