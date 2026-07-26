"""Conformance tests for CRS multiplier and cross-efficiency models."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from frontierguard import DEAData, RadialDEA
from frontierguard._enums import SolverStatus
from frontierguard._errors import (
    DataValidationError,
    ModelValidationError,
    NumericalError,
    SolverError,
)
from frontierguard._solver import LinearProgram, LinearSolution
from frontierguard.multiplier import (
    AssuranceRegion,
    MultiplierDEA,
    WeightKind,
    multiplier_dea,
)


@pytest.fixture
def analytical_data() -> DEAData:
    return DEAData(
        inputs=[[1.0], [2.0], [3.0]],
        outputs=[[1.0], [1.0], [2.0]],
        dmu_names=("A", "B", "C"),
        input_names=("staff",),
        output_names=("cases",),
    )


@pytest.fixture
def five_dmu_data() -> DEAData:
    return DEAData(
        inputs=[[2, 5], [2, 4], [6, 6], [3, 2], [6, 2]],
        outputs=[[1], [2], [3], [1], [2]],
        dmu_names=("A", "B", "C", "D", "E"),
        input_names=("x1", "x2"),
        output_names=("y1",),
    )


@pytest.mark.parametrize("orientation", ["input", "output"])
def test_multiplier_matches_crs_radial_analytical_fixture(
    analytical_data: DEAData,
    orientation: str,
) -> None:
    multiplier = MultiplierDEA(orientation=orientation).fit(analytical_data)
    radial = RadialDEA(
        orientation=orientation,
        rts="crs",
        slacks="none",
    ).fit(analytical_data)

    assert multiplier.radial_factor == pytest.approx(radial.radial_factor)
    assert multiplier.technical_efficiency == pytest.approx(radial.technical_efficiency)
    assert np.diag(multiplier.cross_efficiency) == pytest.approx(
        multiplier.technical_efficiency
    )
    assert multiplier.max_audit_residual <= multiplier.audit_tolerance


@pytest.mark.parametrize("orientation", ["input", "output"])
def test_multiplier_matches_cross_language_five_dmu_fixture(
    five_dmu_data: DEAData,
    orientation: str,
) -> None:
    result = multiplier_dea(five_dmu_data, orientation=orientation)
    expected = np.asarray([0.5, 1.0, 5 / 6, 5 / 7, 1.0])

    assert result.technical_efficiency == pytest.approx(expected)
    if orientation == "input":
        assert result.radial_factor == pytest.approx(expected)
    else:
        assert result.radial_factor == pytest.approx(1.0 / expected)
    assert np.max(result.primal_residual) <= result.audit_tolerance
    assert np.max(result.ratio_residual) <= result.audit_tolerance
    assert np.max(result.restriction_residual) <= result.audit_tolerance
    assert np.max(result.objective_residual) <= result.audit_tolerance


def test_output_orientation_uses_expansion_factor_and_ratio_score(
    analytical_data: DEAData,
) -> None:
    result = MultiplierDEA(orientation="output").fit(analytical_data)

    assert result.radial_factor == pytest.approx([1.0, 2.0, 1.5])
    assert result.technical_efficiency == pytest.approx([1.0, 0.5, 2 / 3])
    assert result.radial_factor * result.technical_efficiency == pytest.approx(
        np.ones(3)
    )


def test_selected_weights_and_full_cross_matrix_are_reconstructable(
    five_dmu_data: DEAData,
) -> None:
    result = MultiplierDEA().fit(five_dmu_data)
    reconstructed = (result.output_weights @ five_dmu_data.outputs.T) / (
        result.input_weights @ five_dmu_data.inputs.T
    )

    assert result.cross_efficiency.shape == (5, 5)
    assert result.cross_efficiency == pytest.approx(reconstructed)
    assert result.cross_frame().index.tolist() == list("ABCDE")
    assert result.cross_frame().columns.tolist() == list("ABCDE")
    assert result.selected_weights("A")["inputs"].keys() == {"x1", "x2"}
    assert result.selected_weights(0)["outputs"].keys() == {"y1"}


def test_peer_average_supports_self_exclusion_and_appraiser_weights(
    analytical_data: DEAData,
) -> None:
    result = MultiplierDEA().fit(analytical_data)

    assert result.peer_average(exclude_self=False) == pytest.approx([1.0, 0.5, 2 / 3])
    assert result.peer_average(exclude_self=True) == pytest.approx([1.0, 0.5, 2 / 3])
    assert result.peer_average(
        exclude_self=False,
        appraiser_weights={"A": 2.0},
    ) == pytest.approx(result.cross_efficiency[0])


def test_assurance_regions_compile_raw_input_ratios_after_column_scaling() -> None:
    data = DEAData(
        inputs=[[100, 1], [200, 2], [150, 3]],
        outputs=[[1], [1], [1]],
        dmu_names=("A", "B", "C"),
        input_names=("capital", "labor"),
    )
    restriction = AssuranceRegion(
        kind=WeightKind.INPUT,
        numerator="capital",
        denominator="labor",
        lower=0.01,
        upper=0.01,
    )
    result = MultiplierDEA(assurance_regions=[restriction]).fit(data)

    assert result.input_weights[:, 0] / result.input_weights[:, 1] == pytest.approx(
        np.full(3, 0.01)
    )
    assert np.max(result.restriction_residual) <= result.audit_tolerance
    assert result.restrictions == (restriction,)


def test_named_and_indexed_output_assurance_regions() -> None:
    data = DEAData(
        inputs=[[1], [2], [3]],
        outputs=[[100, 1], [200, 1], [150, 3]],
        dmu_names=("A", "B", "C"),
        output_names=("primary", "secondary"),
    )
    named = AssuranceRegion(
        "output",
        "primary",
        "secondary",
        lower=0.01,
    )
    indexed = AssuranceRegion(
        "output",
        0,
        1,
        upper=0.01,
    )
    result = MultiplierDEA(
        orientation="output",
        assurance_regions=[named, indexed],
    ).fit(data)

    assert result.output_weights[:, 0] / result.output_weights[:, 1] == pytest.approx(
        np.full(3, 0.01)
    )
    assert result.technical_efficiency == pytest.approx([1.0, 0.75, 0.75])


def test_internal_scaling_preserves_scores_and_transforms_raw_weights(
    five_dmu_data: DEAData,
) -> None:
    transformed = DEAData(
        inputs=five_dmu_data.inputs * np.asarray([100.0, 0.01]),
        outputs=five_dmu_data.outputs * np.asarray([1_000.0]),
        dmu_names=five_dmu_data.dmu_names,
        input_names=five_dmu_data.input_names,
        output_names=five_dmu_data.output_names,
    )
    baseline = MultiplierDEA().fit(five_dmu_data)
    scaled = MultiplierDEA().fit(transformed)

    assert scaled.technical_efficiency == pytest.approx(baseline.technical_efficiency)
    assert scaled.cross_efficiency == pytest.approx(baseline.cross_efficiency)
    assert scaled.input_weights == pytest.approx(
        baseline.input_weights / np.asarray([100.0, 0.01])
    )
    assert scaled.output_weights == pytest.approx(
        baseline.output_weights / np.asarray([1_000.0])
    )


def test_positive_weight_floor_is_explicitly_model_changing(
    five_dmu_data: DEAData,
) -> None:
    canonical = MultiplierDEA().fit(five_dmu_data)
    floored = MultiplierDEA(weight_floor=1e-6).fit(five_dmu_data)

    assert not canonical.weight_floor_changes_model
    assert floored.weight_floor_changes_model
    assert floored.weight_floor == pytest.approx(1e-6)
    assert np.all(floored.input_weights >= 0.0)
    assert np.all(floored.output_weights >= 0.0)


def test_separate_reference_technology_is_used_for_constraints(
    analytical_data: DEAData,
) -> None:
    reference = DEAData(
        inputs=[[1.0]],
        outputs=[[1.0]],
        dmu_names=("benchmark",),
        input_names=("staff",),
        output_names=("cases",),
    )
    result = MultiplierDEA().fit(analytical_data, reference=reference)
    radial = RadialDEA(rts="crs", slacks="none").fit(
        analytical_data,
        reference=reference,
    )

    assert result.reference_names == ("benchmark",)
    assert result.reference_fingerprint == reference.fingerprint
    assert result.technical_efficiency == pytest.approx(radial.technical_efficiency)
    assert result.cross_efficiency.shape == (3, 3)


def test_undefined_cross_denominators_are_null_in_strict_json() -> None:
    data = DEAData(
        inputs=[[1, 0], [0, 1]],
        outputs=[[1, 0], [0, 1]],
        dmu_names=("A", "B"),
    )
    result = MultiplierDEA().fit(data)
    exported = result.to_dict()

    assert np.isnan(result.cross_efficiency[0, 1])
    assert np.isnan(result.cross_efficiency[1, 0])
    assert exported["cross_efficiency"] == [[1.0, None], [None, 1.0]]
    assert exported["peer_average_excluding_self"] == [None, None]
    assert result.peer_average(skip_undefined=False) == pytest.approx(
        [np.nan, np.nan],
        nan_ok=True,
    )
    json.dumps(exported, allow_nan=False)


@pytest.mark.parametrize("rts", ["vrs", "nirs", "ndrs"])
def test_non_crs_models_are_rejected(rts: str) -> None:
    with pytest.raises(ModelValidationError, match="CRS only"):
        MultiplierDEA(rts=rts)


@pytest.mark.parametrize(
    "restriction",
    [
        AssuranceRegion("input", "x1", "x2", lower=0.0),
        AssuranceRegion("output", 0, 1, upper=2.0),
    ],
)
def test_restriction_serialization_is_strict(
    restriction: AssuranceRegion,
) -> None:
    json.dumps(restriction.to_dict(), allow_nan=False)


def test_invalid_restrictions_and_average_weights_fail(
    five_dmu_data: DEAData,
) -> None:
    with pytest.raises(ModelValidationError, match="requires"):
        AssuranceRegion("input", 0, 1)
    with pytest.raises(ModelValidationError, match="cannot exceed"):
        AssuranceRegion("input", 0, 1, lower=2.0, upper=1.0)
    with pytest.raises(ModelValidationError, match="Unknown multiplier"):
        MultiplierDEA(
            assurance_regions=[AssuranceRegion("input", "missing", "x1", lower=1.0)]
        ).fit(five_dmu_data)
    with pytest.raises(ModelValidationError, match="same multiplier"):
        MultiplierDEA(
            assurance_regions=[AssuranceRegion("input", "x1", 0, lower=1.0)]
        ).fit(five_dmu_data)

    result = MultiplierDEA().fit(five_dmu_data)
    with pytest.raises(DataValidationError, match="contain 5"):
        result.peer_average(appraiser_weights=[1.0])
    with pytest.raises(DataValidationError, match="non-negative"):
        result.peer_average(appraiser_weights=[1, 1, 1, 1, -1])
    with pytest.raises(DataValidationError, match="Unknown appraisers"):
        result.peer_average(appraiser_weights={"missing": 1.0})


def test_reference_contract_and_model_configuration_are_validated(
    five_dmu_data: DEAData,
) -> None:
    wrong_names = DEAData(
        five_dmu_data.inputs,
        five_dmu_data.outputs,
        input_names=("other", "x2"),
        output_names=("y1",),
    )
    with pytest.raises(ModelValidationError, match="input names"):
        MultiplierDEA().fit(five_dmu_data, reference=wrong_names)
    with pytest.raises(ModelValidationError, match="weight_floor"):
        MultiplierDEA(weight_floor=-1.0)
    with pytest.raises(ModelValidationError, match="tolerance"):
        MultiplierDEA(tolerance=0.0)


@dataclass
class _TerminalSolver:
    status: SolverStatus
    name: str = "terminal-test"

    def solve(self, program: LinearProgram) -> LinearSolution:
        size = np.asarray(program.objective).size
        variables = np.full(size, np.nan)
        variables.setflags(write=False)
        return LinearSolution(
            status=self.status,
            variables=variables,
            objective=float("nan"),
            message="deliberate test terminal state",
            raw_status=17,
        )


class _InvalidOptimalSolver:
    name = "invalid-optimal-test"

    def solve(self, program: LinearProgram) -> LinearSolution:
        size = np.asarray(program.objective).size
        variables: NDArray[np.float64] = np.zeros(size)
        variables.setflags(write=False)
        return LinearSolution(
            status=SolverStatus.OPTIMAL,
            variables=variables,
            objective=0.0,
            message="deliberately invalid optimum",
            raw_status=0,
        )


@pytest.mark.parametrize(
    "status",
    [
        SolverStatus.INFEASIBLE,
        SolverStatus.UNBOUNDED,
        SolverStatus.ITERATION_LIMIT,
        SolverStatus.ERROR,
    ],
)
def test_solver_terminal_states_raise_solver_error(
    analytical_data: DEAData,
    status: SolverStatus,
) -> None:
    with pytest.raises(SolverError, match=status.value):
        MultiplierDEA(solver=_TerminalSolver(status)).fit(analytical_data)


def test_independent_audit_rejects_invalid_optimal_solution(
    analytical_data: DEAData,
) -> None:
    with pytest.raises(NumericalError, match="audit"):
        MultiplierDEA(solver=_InvalidOptimalSolver()).fit(analytical_data)
