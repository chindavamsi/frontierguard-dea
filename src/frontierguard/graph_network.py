"""Graph-defined relational network DEA for acyclic production systems."""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from frontierguard._enums import SolverStatus
from frontierguard._errors import (
    DataValidationError,
    ModelValidationError,
    NumericalError,
    SolverError,
)
from frontierguard._solver import LinearProgram, LinearSolver, ScipyHighsSolver


def _optional_matrix(
    values: ArrayLike | None,
    *,
    label: str,
) -> NDArray[np.float64] | None:
    if values is None:
        return None
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{label} must be a numeric matrix.") from exc
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise DataValidationError(
            f"{label} must contain at least one row and one column."
        )
    if not np.isfinite(array).all():
        raise DataValidationError(f"{label} contains missing or non-finite values.")
    if (array < 0).any():
        raise DataValidationError(f"{label} must be non-negative.")
    if np.any(array.max(axis=0) <= 0):
        raise DataValidationError(
            f"Every column of {label} must contain a positive value."
        )
    result = np.array(array, dtype=float, copy=True, order="C")
    result.setflags(write=False)
    return result


def _factor_names(
    values: Sequence[object] | None,
    *,
    count: int,
    prefix: str,
    label: str,
) -> tuple[str, ...]:
    names = (
        tuple(f"{prefix}{position + 1}" for position in range(count))
        if values is None
        else tuple(str(value) for value in values)
    )
    if len(names) != count:
        raise DataValidationError(f"{label} must contain exactly {count} names.")
    if any(not name.strip() for name in names):
        raise DataValidationError(f"{label} cannot contain blank names.")
    if len(set(names)) != len(names):
        raise DataValidationError(f"{label} must be unique.")
    return names


def _name(value: object, *, label: str) -> str:
    resolved = str(value)
    if not resolved.strip():
        raise DataValidationError(f"{label} cannot be blank.")
    return resolved


@dataclass(frozen=True, slots=True, init=False)
class NetworkStage:
    """One production stage with optional external inputs and outputs."""

    name: str
    inputs: NDArray[np.float64] | None
    outputs: NDArray[np.float64] | None
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]

    def __init__(
        self,
        name: object,
        *,
        inputs: ArrayLike | None = None,
        outputs: ArrayLike | None = None,
        input_names: Sequence[object] | None = None,
        output_names: Sequence[object] | None = None,
    ) -> None:
        resolved_name = _name(name, label="stage name")
        resolved_inputs = _optional_matrix(inputs, label=f"{resolved_name} inputs")
        resolved_outputs = _optional_matrix(outputs, label=f"{resolved_name} outputs")
        if (
            resolved_inputs is not None
            and resolved_outputs is not None
            and resolved_inputs.shape[0] != resolved_outputs.shape[0]
        ):
            raise DataValidationError(
                f"Inputs and outputs for stage {resolved_name!r} must contain "
                "the same DMUs."
            )
        if resolved_inputs is None and input_names is not None:
            raise DataValidationError(
                f"Stage {resolved_name!r} has input_names but no inputs."
            )
        if resolved_outputs is None and output_names is not None:
            raise DataValidationError(
                f"Stage {resolved_name!r} has output_names but no outputs."
            )
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "inputs", resolved_inputs)
        object.__setattr__(self, "outputs", resolved_outputs)
        object.__setattr__(
            self,
            "input_names",
            _factor_names(
                input_names,
                count=0 if resolved_inputs is None else resolved_inputs.shape[1],
                prefix=f"{resolved_name}_x",
                label=f"{resolved_name} input_names",
            ),
        )
        object.__setattr__(
            self,
            "output_names",
            _factor_names(
                output_names,
                count=0 if resolved_outputs is None else resolved_outputs.shape[1],
                prefix=f"{resolved_name}_y",
                label=f"{resolved_name} output_names",
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class NetworkLink:
    """One directed intermediate flow shared by its producer and consumer."""

    name: str
    source: str
    target: str
    values: NDArray[np.float64]
    factor_names: tuple[str, ...]

    def __init__(
        self,
        name: object,
        source: object,
        target: object,
        values: ArrayLike,
        *,
        factor_names: Sequence[object] | None = None,
    ) -> None:
        resolved_name = _name(name, label="link name")
        resolved_source = _name(source, label=f"{resolved_name} source")
        resolved_target = _name(target, label=f"{resolved_name} target")
        if resolved_source == resolved_target:
            raise DataValidationError(
                f"Link {resolved_name!r} cannot connect a stage to itself."
            )
        resolved_values = _optional_matrix(values, label=f"{resolved_name} values")
        if resolved_values is None:  # pragma: no cover - required positional argument
            raise DataValidationError(f"Link {resolved_name!r} requires values.")
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "source", resolved_source)
        object.__setattr__(self, "target", resolved_target)
        object.__setattr__(self, "values", resolved_values)
        object.__setattr__(
            self,
            "factor_names",
            _factor_names(
                factor_names,
                count=resolved_values.shape[1],
                prefix=f"{resolved_name}_z",
                label=f"{resolved_name} factor_names",
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class NetworkData:
    """Validated acyclic production graph evaluated over common DMUs."""

    stages: tuple[NetworkStage, ...]
    links: tuple[NetworkLink, ...]
    dmu_names: tuple[str, ...]
    topological_order: tuple[str, ...]
    fingerprint: str

    def __init__(
        self,
        stages: Sequence[NetworkStage],
        links: Sequence[NetworkLink],
        *,
        dmu_names: Sequence[object] | None = None,
    ) -> None:
        resolved_stages = tuple(stages)
        resolved_links = tuple(links)
        if len(resolved_stages) < 2:
            raise DataValidationError("A network requires at least two stages.")
        if not resolved_links:
            raise DataValidationError("A network requires at least one link.")
        if not all(isinstance(stage, NetworkStage) for stage in resolved_stages):
            raise DataValidationError("stages must contain NetworkStage objects.")
        if not all(isinstance(link, NetworkLink) for link in resolved_links):
            raise DataValidationError("links must contain NetworkLink objects.")

        stage_names = tuple(stage.name for stage in resolved_stages)
        if len(set(stage_names)) != len(stage_names):
            raise DataValidationError("Stage names must be unique.")
        link_names = tuple(link.name for link in resolved_links)
        if len(set(link_names)) != len(link_names):
            raise DataValidationError("Link names must be unique.")

        stage_set = set(stage_names)
        for link in resolved_links:
            if link.source not in stage_set or link.target not in stage_set:
                raise DataValidationError(
                    f"Link {link.name!r} references an unknown source or target stage."
                )

        arrays = [
            array
            for stage in resolved_stages
            for array in (stage.inputs, stage.outputs)
            if array is not None
        ]
        arrays.extend(link.values for link in resolved_links)
        n_dmus = arrays[0].shape[0]
        if any(array.shape[0] != n_dmus for array in arrays):
            raise DataValidationError(
                "Every stage factor and link must contain the same DMUs."
            )
        resolved_dmus = _factor_names(
            dmu_names,
            count=n_dmus,
            prefix="DMU",
            label="dmu_names",
        )

        incoming: dict[str, list[NetworkLink]] = defaultdict(list)
        outgoing: dict[str, list[NetworkLink]] = defaultdict(list)
        for link in resolved_links:
            outgoing[link.source].append(link)
            incoming[link.target].append(link)
        for stage in resolved_stages:
            if stage.inputs is None and not incoming[stage.name]:
                raise DataValidationError(
                    f"Stage {stage.name!r} has no external input or incoming link."
                )
            if stage.outputs is None and not outgoing[stage.name]:
                raise DataValidationError(
                    f"Stage {stage.name!r} has no external output or outgoing link."
                )

        external_input_total = np.zeros(n_dmus)
        external_output_total = np.zeros(n_dmus)
        for stage in resolved_stages:
            if stage.inputs is not None:
                external_input_total += stage.inputs.sum(axis=1)
            if stage.outputs is not None:
                external_output_total += stage.outputs.sum(axis=1)
        if np.any(external_input_total <= 0):
            raise DataValidationError(
                "Every DMU must have a positive aggregate external input."
            )
        if np.any(external_output_total <= 0):
            raise DataValidationError(
                "Every DMU must have a positive aggregate external output."
            )

        topological_order = _topological_order(stage_names, resolved_links)
        digest = hashlib.sha256()
        for part in (
            resolved_dmus,
            stage_names,
            *(stage.input_names for stage in resolved_stages),
            *(stage.output_names for stage in resolved_stages),
            tuple((link.name, link.source, link.target) for link in resolved_links),
            *(link.factor_names for link in resolved_links),
            *arrays,
        ):
            if isinstance(part, np.ndarray):
                digest.update(str(part.shape).encode())
                digest.update(part.dtype.str.encode())
                digest.update(part.tobytes(order="C"))
            else:
                digest.update(repr(part).encode())
            digest.update(b"\0")

        object.__setattr__(self, "stages", resolved_stages)
        object.__setattr__(self, "links", resolved_links)
        object.__setattr__(self, "dmu_names", resolved_dmus)
        object.__setattr__(self, "topological_order", topological_order)
        object.__setattr__(self, "fingerprint", digest.hexdigest())

    @property
    def n_dmus(self) -> int:
        return len(self.dmu_names)


def _topological_order(
    stage_names: tuple[str, ...],
    links: tuple[NetworkLink, ...],
) -> tuple[str, ...]:
    indegree = dict.fromkeys(stage_names, 0)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for link in links:
        indegree[link.target] += 1
        outgoing[link.source].append(link.target)
    queue = deque(name for name in stage_names if indegree[name] == 0)
    ordered: list[str] = []
    while queue:
        current = queue.popleft()
        ordered.append(current)
        for target in outgoing[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(ordered) != len(stage_names):
        raise DataValidationError("Network links must form an acyclic graph.")
    return tuple(ordered)


def _readonly(values: ArrayLike) -> NDArray[np.float64]:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, init=False)
class NetworkResult:
    """Audited result from a graph-defined CRS relational model."""

    dmu_names: tuple[str, ...]
    stage_names: tuple[str, ...]
    link_names: tuple[str, ...]
    overall_efficiency: NDArray[np.float64]
    stage_efficiency: NDArray[np.float64]
    stage_gap: NDArray[np.float64]
    closure_error: NDArray[np.float64]
    input_weights: tuple[NDArray[np.float64], ...]
    link_weights: tuple[NDArray[np.float64], ...]
    output_weights: tuple[NDArray[np.float64], ...]
    audit_residual: NDArray[np.float64]
    audit_tolerance: float
    statuses: tuple[SolverStatus, ...]
    solver: str
    data_fingerprint: str

    def __init__(
        self,
        *,
        dmu_names: Sequence[str],
        stage_names: Sequence[str],
        link_names: Sequence[str],
        overall_efficiency: ArrayLike,
        stage_efficiency: ArrayLike,
        stage_gap: ArrayLike,
        closure_error: ArrayLike,
        input_weights: Sequence[ArrayLike],
        link_weights: Sequence[ArrayLike],
        output_weights: Sequence[ArrayLike],
        audit_residual: ArrayLike,
        audit_tolerance: float,
        statuses: Sequence[SolverStatus],
        solver: str,
        data_fingerprint: str,
    ) -> None:
        object.__setattr__(self, "dmu_names", tuple(dmu_names))
        object.__setattr__(self, "stage_names", tuple(stage_names))
        object.__setattr__(self, "link_names", tuple(link_names))
        object.__setattr__(self, "overall_efficiency", _readonly(overall_efficiency))
        object.__setattr__(self, "stage_efficiency", _readonly(stage_efficiency))
        object.__setattr__(self, "stage_gap", _readonly(stage_gap))
        object.__setattr__(self, "closure_error", _readonly(closure_error))
        object.__setattr__(
            self,
            "input_weights",
            tuple(_readonly(weights) for weights in input_weights),
        )
        object.__setattr__(
            self,
            "link_weights",
            tuple(_readonly(weights) for weights in link_weights),
        )
        object.__setattr__(
            self,
            "output_weights",
            tuple(_readonly(weights) for weights in output_weights),
        )
        object.__setattr__(self, "audit_residual", _readonly(audit_residual))
        object.__setattr__(self, "audit_tolerance", float(audit_tolerance))
        object.__setattr__(self, "statuses", tuple(statuses))
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "data_fingerprint", data_fingerprint)

    def to_frame(self) -> pd.DataFrame:
        """Return overall, stage, decomposition, and audit values."""
        values: dict[str, object] = {
            "dmu": self.dmu_names,
            "overall_efficiency": self.overall_efficiency,
        }
        for position, name in enumerate(self.stage_names):
            values[f"stage:{name}"] = self.stage_efficiency[:, position]
            values[f"gap:{name}"] = self.stage_gap[:, position]
        values["closure_error"] = self.closure_error
        values["audit_residual"] = self.audit_residual
        values["status"] = [status.value for status in self.statuses]
        return pd.DataFrame(values).set_index("dmu")

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible result."""

        def nullable(values: NDArray[np.float64]) -> list[object]:
            return [
                (
                    [float(item) if np.isfinite(item) else None for item in value]
                    if np.ndim(value)
                    else float(value)
                    if np.isfinite(value)
                    else None
                )
                for value in values
            ]

        return {
            "schema_version": "0.1.0",
            "model": "network_dag_relational",
            "returns_to_scale": "crs",
            "solver": self.solver,
            "data_fingerprint": self.data_fingerprint,
            "dmu_names": list(self.dmu_names),
            "stage_names": list(self.stage_names),
            "link_names": list(self.link_names),
            "overall_efficiency": self.overall_efficiency.tolist(),
            "stage_efficiency": nullable(self.stage_efficiency),
            "stage_gap": self.stage_gap.tolist(),
            "closure_error": self.closure_error.tolist(),
            "input_weights": [weights.tolist() for weights in self.input_weights],
            "link_weights": [weights.tolist() for weights in self.link_weights],
            "output_weights": [weights.tolist() for weights in self.output_weights],
            "audit_residual": self.audit_residual.tolist(),
            "audit_tolerance": self.audit_tolerance,
            "statuses": [status.value for status in self.statuses],
        }


class RelationalNetworkDEA:
    """Centralized CRS relational DEA for an arbitrary acyclic stage graph."""

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

    def fit(self, data: NetworkData) -> NetworkResult:
        """Fit every DMU with shared weights on each directed intermediate link."""
        if not isinstance(data, NetworkData):
            raise ModelValidationError("data must be a NetworkData instance.")

        scaled_inputs, input_scales = _scaled_stage_values(
            tuple(stage.inputs for stage in data.stages),
            tolerance=self.tolerance,
        )
        scaled_outputs, output_scales = _scaled_stage_values(
            tuple(stage.outputs for stage in data.stages),
            tolerance=self.tolerance,
        )
        scaled_links, link_scales = _scaled_stage_values(
            tuple(link.values for link in data.links),
            tolerance=self.tolerance,
        )
        stage_index = {
            stage.name: position for position, stage in enumerate(data.stages)
        }
        incoming: list[list[int]] = [[] for _ in data.stages]
        outgoing: list[list[int]] = [[] for _ in data.stages]
        for position, link in enumerate(data.links):
            outgoing[stage_index[link.source]].append(position)
            incoming[stage_index[link.target]].append(position)

        cursor = 0
        input_slices: list[slice] = []
        for values in scaled_inputs:
            width = 0 if values is None else values.shape[1]
            input_slices.append(slice(cursor, cursor + width))
            cursor += width
        link_slices: list[slice] = []
        for values in scaled_links:
            if values is None:  # pragma: no cover - links always contain data
                raise NumericalError("A validated link unexpectedly has no values.")
            link_slices.append(slice(cursor, cursor + values.shape[1]))
            cursor += values.shape[1]
        output_slices: list[slice] = []
        for values in scaled_outputs:
            width = 0 if values is None else values.shape[1]
            output_slices.append(slice(cursor, cursor + width))
            cursor += width
        n_variables = cursor

        n_dmus = data.n_dmus
        n_stages = len(data.stages)
        overall = np.empty(n_dmus)
        stage_efficiency = np.empty((n_dmus, n_stages))
        stage_gap = np.empty((n_dmus, n_stages))
        closure = np.empty(n_dmus)
        audits = np.empty(n_dmus)
        input_weights = tuple(
            np.empty((n_dmus, 0 if values is None else values.shape[1]))
            for values in scaled_inputs
        )
        link_weights = tuple(
            np.empty((n_dmus, values.shape[1]))
            for values in scaled_links
            if values is not None
        )
        output_weights = tuple(
            np.empty((n_dmus, 0 if values is None else values.shape[1]))
            for values in scaled_outputs
        )
        statuses: list[SolverStatus] = []
        audit_tolerance = max(10.0 * self.tolerance, 1e-7)

        for evaluated in range(n_dmus):
            objective = np.zeros(n_variables)
            equality = np.zeros(n_variables)
            for position, values in enumerate(scaled_inputs):
                if values is not None:
                    equality[input_slices[position]] = values[evaluated]
            for position, values in enumerate(scaled_outputs):
                if values is not None:
                    objective[output_slices[position]] = -values[evaluated]

            inequalities: list[NDArray[np.float64]] = []
            for reference in range(n_dmus):
                for stage_position in range(n_stages):
                    row = np.zeros(n_variables)
                    inputs = scaled_inputs[stage_position]
                    outputs = scaled_outputs[stage_position]
                    if inputs is not None:
                        row[input_slices[stage_position]] -= inputs[reference]
                    if outputs is not None:
                        row[output_slices[stage_position]] += outputs[reference]
                    for link_position in incoming[stage_position]:
                        link_values = scaled_links[link_position]
                        if link_values is None:  # pragma: no cover
                            raise NumericalError("A link unexpectedly has no values.")
                        row[link_slices[link_position]] -= link_values[reference]
                    for link_position in outgoing[stage_position]:
                        link_values = scaled_links[link_position]
                        if link_values is None:  # pragma: no cover
                            raise NumericalError("A link unexpectedly has no values.")
                        row[link_slices[link_position]] += link_values[reference]
                    inequalities.append(row)

            solution = self.solver.solve(
                LinearProgram(
                    objective=objective,
                    inequalities=np.vstack(inequalities),
                    inequality_bounds=np.zeros(len(inequalities)),
                    equalities=equality.reshape(1, -1),
                    equality_bounds=np.ones(1),
                    variable_bounds=[(self.weight_floor, None)] * n_variables,
                )
            )
            if solution.status is not SolverStatus.OPTIMAL:
                raise SolverError(
                    f"Network model for {data.dmu_names[evaluated]!r} failed "
                    f"with {solution.status.value}: {solution.message}"
                )

            selected_inputs = tuple(
                solution.variables[values_slice] for values_slice in input_slices
            )
            selected_links = tuple(
                solution.variables[values_slice] for values_slice in link_slices
            )
            selected_outputs = tuple(
                solution.variables[values_slice] for values_slice in output_slices
            )
            denominators = np.empty(n_stages)
            numerators = np.empty(n_stages)
            for stage_position in range(n_stages):
                denominators[stage_position] = _virtual_value(
                    selected_inputs[stage_position],
                    scaled_inputs[stage_position],
                    evaluated,
                ) + sum(
                    _virtual_value(
                        selected_links[link_position],
                        scaled_links[link_position],
                        evaluated,
                    )
                    for link_position in incoming[stage_position]
                )
                numerators[stage_position] = _virtual_value(
                    selected_outputs[stage_position],
                    scaled_outputs[stage_position],
                    evaluated,
                ) + sum(
                    _virtual_value(
                        selected_links[link_position],
                        scaled_links[link_position],
                        evaluated,
                    )
                    for link_position in outgoing[stage_position]
                )
            scores = np.divide(
                numerators,
                denominators,
                out=np.full(n_stages, np.nan),
                where=denominators > self.tolerance,
            )
            score = -solution.objective
            gaps = denominators - numerators
            closure_error = (1.0 - score) - float(gaps.sum())
            audit_residual = _audit_graph_solution(
                data=data,
                scaled_inputs=scaled_inputs,
                scaled_links=scaled_links,
                scaled_outputs=scaled_outputs,
                input_weights=selected_inputs,
                link_weights=selected_links,
                output_weights=selected_outputs,
                incoming=incoming,
                outgoing=outgoing,
                evaluated=evaluated,
                score=score,
                stage_scores=scores,
                stage_gaps=gaps,
                solver_objective=solution.objective,
            )
            if audit_residual > audit_tolerance:
                raise NumericalError(
                    f"Post-solve network audit for "
                    f"{data.dmu_names[evaluated]!r} failed with scaled residual "
                    f"{audit_residual:.3e} above {audit_tolerance:.3e}."
                )

            overall[evaluated] = score
            stage_efficiency[evaluated] = scores
            stage_gap[evaluated] = gaps
            closure[evaluated] = closure_error
            audits[evaluated] = audit_residual
            for position, weights in enumerate(selected_inputs):
                if weights.size:
                    scale = input_scales[position]
                    if scale is None:  # pragma: no cover
                        raise NumericalError("An input scale is unexpectedly absent.")
                    input_weights[position][evaluated] = weights / scale
            for position, weights in enumerate(selected_links):
                scale = link_scales[position]
                if scale is None:  # pragma: no cover
                    raise NumericalError("A link scale is unexpectedly absent.")
                link_weights[position][evaluated] = weights / scale
            for position, weights in enumerate(selected_outputs):
                if weights.size:
                    scale = output_scales[position]
                    if scale is None:  # pragma: no cover
                        raise NumericalError("An output scale is unexpectedly absent.")
                    output_weights[position][evaluated] = weights / scale
            statuses.append(solution.status)

        if not all(
            np.isfinite(values).all()
            for values in (
                overall,
                stage_gap,
                closure,
                audits,
                *input_weights,
                *link_weights,
                *output_weights,
            )
        ):
            raise NumericalError("The network result contains non-finite core values.")
        return NetworkResult(
            dmu_names=data.dmu_names,
            stage_names=tuple(stage.name for stage in data.stages),
            link_names=tuple(link.name for link in data.links),
            overall_efficiency=overall,
            stage_efficiency=stage_efficiency,
            stage_gap=stage_gap,
            closure_error=closure,
            input_weights=input_weights,
            link_weights=link_weights,
            output_weights=output_weights,
            audit_residual=audits,
            audit_tolerance=audit_tolerance,
            statuses=statuses,
            solver=self.solver.name,
            data_fingerprint=data.fingerprint,
        )


def _scaled_stage_values(
    arrays: tuple[NDArray[np.float64] | None, ...],
    *,
    tolerance: float,
) -> tuple[
    tuple[NDArray[np.float64] | None, ...],
    tuple[NDArray[np.float64] | None, ...],
]:
    scaled: list[NDArray[np.float64] | None] = []
    scales: list[NDArray[np.float64] | None] = []
    for values in arrays:
        if values is None:
            scaled.append(None)
            scales.append(None)
        else:
            scale = np.maximum(values.mean(axis=0), tolerance)
            scaled.append(values / scale)
            scales.append(scale)
    return tuple(scaled), tuple(scales)


def _virtual_value(
    weights: NDArray[np.float64],
    values: NDArray[np.float64] | None,
    dmu: int,
) -> float:
    if values is None:
        return 0.0
    return float(weights @ values[dmu])


def _scaled_residual(
    residual: NDArray[np.float64] | float,
    *scales: NDArray[np.float64] | float,
    positive_only: bool = False,
) -> float:
    values = np.asarray(residual, dtype=float)
    if positive_only:
        values = np.maximum(values, 0.0)
    numerator = float(np.max(np.abs(values), initial=0.0))
    denominator = max(
        (
            1.0 + float(np.max(np.abs(np.asarray(scale)), initial=0.0))
            for scale in scales
        ),
        default=1.0,
    )
    return numerator / denominator


def _audit_graph_solution(
    *,
    data: NetworkData,
    scaled_inputs: tuple[NDArray[np.float64] | None, ...],
    scaled_links: tuple[NDArray[np.float64] | None, ...],
    scaled_outputs: tuple[NDArray[np.float64] | None, ...],
    input_weights: tuple[NDArray[np.float64], ...],
    link_weights: tuple[NDArray[np.float64], ...],
    output_weights: tuple[NDArray[np.float64], ...],
    incoming: list[list[int]],
    outgoing: list[list[int]],
    evaluated: int,
    score: float,
    stage_scores: NDArray[np.float64],
    stage_gaps: NDArray[np.float64],
    solver_objective: float,
) -> float:
    residuals = [
        *(
            _scaled_residual(-weights, weights, positive_only=True)
            for weights in input_weights
        ),
        *(
            _scaled_residual(-weights, weights, positive_only=True)
            for weights in link_weights
        ),
        *(
            _scaled_residual(-weights, weights, positive_only=True)
            for weights in output_weights
        ),
    ]
    external_input = sum(
        _virtual_value(weights, values, evaluated)
        for weights, values in zip(input_weights, scaled_inputs, strict=True)
    )
    external_output = sum(
        _virtual_value(weights, values, evaluated)
        for weights, values in zip(output_weights, scaled_outputs, strict=True)
    )
    residuals.extend(
        [
            _scaled_residual(external_input - 1.0, external_input),
            _scaled_residual(
                solver_objective + external_output,
                solver_objective,
                external_output,
            ),
            _scaled_residual(score - external_output, score, external_output),
        ]
    )

    selected_denominators = np.empty(len(data.stages))
    selected_numerators = np.empty(len(data.stages))
    for reference in range(data.n_dmus):
        for stage_position in range(len(data.stages)):
            denominator = _virtual_value(
                input_weights[stage_position],
                scaled_inputs[stage_position],
                reference,
            ) + sum(
                _virtual_value(
                    link_weights[link_position],
                    scaled_links[link_position],
                    reference,
                )
                for link_position in incoming[stage_position]
            )
            numerator = _virtual_value(
                output_weights[stage_position],
                scaled_outputs[stage_position],
                reference,
            ) + sum(
                _virtual_value(
                    link_weights[link_position],
                    scaled_links[link_position],
                    reference,
                )
                for link_position in outgoing[stage_position]
            )
            residuals.append(
                _scaled_residual(
                    numerator - denominator,
                    numerator,
                    denominator,
                    positive_only=True,
                )
            )
            if reference == evaluated:
                selected_denominators[stage_position] = denominator
                selected_numerators[stage_position] = numerator

    residuals.extend(
        [
            _scaled_residual(
                stage_gaps - (selected_denominators - selected_numerators),
                stage_gaps,
            ),
            _scaled_residual(
                (1.0 - score) - float(stage_gaps.sum()),
                score,
                stage_gaps,
            ),
        ]
    )
    if np.isfinite(stage_scores).all():
        residuals.append(
            _scaled_residual(
                stage_scores - selected_numerators / selected_denominators,
                stage_scores,
            )
        )
    return max(residuals, default=0.0)
