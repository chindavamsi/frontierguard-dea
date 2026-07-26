from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from frontierguard._data import DEAData
from frontierguard._enums import SolverStatus
from frontierguard._errors import ModelValidationError, NumericalError, SolverError
from frontierguard._solver import (
    HighsSolver,
    LinearProgram,
    LinearSolution,
    ScipyHighsSolver,
)
from frontierguard.sbm import SBMDEA, sbm


@pytest.fixture
def dominance_data() -> DEAData:
    return DEAData(
        inputs=[[1.0], [2.0]],
        outputs=[[1.0], [1.0]],
        dmu_names=("A", "B"),
        input_names=("cost",),
        output_names=("service",),
    )


def test_crs_sbm_hand_oracle_and_projection(dominance_data: DEAData) -> None:
    result = SBMDEA(rts="crs").fit(dominance_data)

    assert result.rho == pytest.approx([1.0, 0.5])
    assert result.lambdas[1] == pytest.approx([1.0, 0.0])
    assert result.input_slacks[1] == pytest.approx([1.0])
    assert result.output_slacks[1] == pytest.approx([0.0])
    assert result.input_targets[1] == pytest.approx([1.0])
    assert result.output_targets[1] == pytest.approx([1.0])
    assert result.t == pytest.approx([1.0, 1.0])
    assert result.statuses == (SolverStatus.OPTIMAL, SolverStatus.OPTIMAL)
    assert np.max(result.audit_residual) <= result.audit_tolerance

    assert result.input_targets + result.input_slacks == pytest.approx(
        dominance_data.inputs
    )
    assert result.output_targets - result.output_slacks == pytest.approx(
        dominance_data.outputs
    )
    assert result.peers("B") == {"A": pytest.approx(1.0)}
    assert result.targets(1) == {
        "inputs": {"cost": pytest.approx(1.0)},
        "outputs": {"service": pytest.approx(1.0)},
    }
    assert result.to_frame().loc["B", "sbm_efficiency"] == pytest.approx(0.5)


def test_two_sided_slack_score_uses_fractional_tone_objective() -> None:
    data = DEAData(
        inputs=[[1.0], [2.0]],
        outputs=[[2.0], [1.0]],
        dmu_names=("A", "B"),
    )

    result = SBMDEA(rts="vrs").fit(data)

    # B projects to A: (1 - 1/2) / (1 + 1/1) = 1/4.
    assert result.rho == pytest.approx([1.0, 0.25])
    assert result.input_slacks[1] == pytest.approx([1.0])
    assert result.output_slacks[1] == pytest.approx([1.0])
    assert result.t[1] == pytest.approx(0.5)


@pytest.mark.parametrize("rts", ["crs", "vrs", "nirs", "ndrs"])
def test_sbm_respects_each_rts_constraint(
    dominance_data: DEAData,
    rts: str,
) -> None:
    result = SBMDEA(rts=rts).fit(dominance_data)
    intensity_sum = result.lambdas.sum(axis=1)

    if rts == "vrs":
        assert intensity_sum == pytest.approx([1.0, 1.0])
    elif rts == "nirs":
        assert np.all(intensity_sum <= 1.0 + 1e-8)
    elif rts == "ndrs":
        assert np.all(intensity_sum >= 1.0 - 1e-8)
    assert np.all((result.rho > 0.0) & (result.rho <= 1.0))
    assert np.max(result.audit_residual) <= result.audit_tolerance


def test_sbm_accepts_separate_reference_and_tracks_provenance() -> None:
    evaluated = DEAData(
        inputs=[[2.0]],
        outputs=[[1.0]],
        dmu_names=("evaluated",),
        input_names=("x",),
        output_names=("y",),
    )
    reference = DEAData(
        inputs=[[1.0]],
        outputs=[[1.0]],
        dmu_names=("benchmark",),
        input_names=("x",),
        output_names=("y",),
    )

    result = SBMDEA(rts="crs").fit(evaluated, reference=reference)

    assert result.rho == pytest.approx([0.5])
    assert result.reference_names == ("benchmark",)
    assert result.lambdas.shape == (1, 1)
    assert result.data_fingerprint == evaluated.fingerprint
    assert result.reference_fingerprint == reference.fingerprint
    assert result.formulation == "tone_non_oriented_sbm_charnes_cooper"
    assert result.solver == "scipy-highs"


def test_leave_one_out_is_a_reference_option_not_super_sbm() -> None:
    data = DEAData(
        inputs=[[1.0], [1.0], [2.0]],
        outputs=[[1.0], [1.0], [1.0]],
        dmu_names=("A", "A-copy", "B"),
    )

    result = SBMDEA(rts="crs").fit(data, exclude_self=True)

    assert result.rho == pytest.approx([1.0, 1.0, 0.5])
    assert np.diag(result.lambdas) == pytest.approx([0.0, 0.0, 0.0])
    assert result.exclude_self


def test_sbm_is_invariant_to_units_and_dmu_feature_permutations() -> None:
    data = DEAData(
        inputs=[
            [1.0, 4.0],
            [2.0, 2.0],
            [4.0, 1.0],
            [4.0, 4.0],
        ],
        outputs=[
            [3.0, 1.0],
            [2.0, 2.0],
            [1.0, 3.0],
            [1.0, 1.0],
        ],
        dmu_names=("A", "B", "C", "D"),
        input_names=("labor", "capital"),
        output_names=("quality", "volume"),
    )
    baseline = SBMDEA(rts="vrs").fit(data)
    scaled = DEAData(
        inputs=np.asarray(data.inputs) * np.asarray([10.0, 0.25]),
        outputs=np.asarray(data.outputs) * np.asarray([0.1, 8.0]),
        dmu_names=data.dmu_names,
        input_names=data.input_names,
        output_names=data.output_names,
    )
    row_order = np.asarray([2, 0, 3, 1])
    permuted = DEAData(
        inputs=np.asarray(data.inputs)[row_order][:, ::-1],
        outputs=np.asarray(data.outputs)[row_order][:, ::-1],
        dmu_names=tuple(data.dmu_names[index] for index in row_order),
        input_names=tuple(reversed(data.input_names)),
        output_names=tuple(reversed(data.output_names)),
    )

    unit_result = SBMDEA(rts="vrs").fit(scaled)
    permutation_result = SBMDEA(rts="vrs").fit(permuted)
    score_by_name = dict(
        zip(
            permutation_result.dmu_names,
            permutation_result.rho,
            strict=True,
        )
    )

    assert unit_result.rho == pytest.approx(baseline.rho)
    assert unit_result.input_slacks / scaled.inputs == pytest.approx(
        baseline.input_slacks / data.inputs
    )
    assert unit_result.output_slacks / scaled.outputs == pytest.approx(
        baseline.output_slacks / data.outputs
    )
    assert [score_by_name[name] for name in data.dmu_names] == pytest.approx(
        baseline.rho
    )


def test_result_is_immutable_and_strict_json(dominance_data: DEAData) -> None:
    result = sbm(dominance_data, rts="crs")
    exported = result.to_dict()

    with pytest.raises(ValueError, match="read-only"):
        result.rho[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        result.lambdas[0, 0] = 0.0
    json.dumps(exported, allow_nan=False)
    assert exported["model"] == "slack_based_measure"
    assert exported["returns_to_scale"] == "crs"
    assert exported["statuses"] == ["optimal", "optimal"]


@pytest.mark.parametrize(
    "data",
    [
        DEAData(
            inputs=[[0.0, 1.0], [1.0, 1.0]],
            outputs=[[1.0], [1.0]],
        ),
        DEAData(
            inputs=[[1.0], [1.0]],
            outputs=[[0.0, 1.0], [1.0, 1.0]],
        ),
    ],
)
def test_sbm_rejects_zero_denominators_without_translation(data: DEAData) -> None:
    with pytest.raises(ModelValidationError, match="strictly positive"):
        SBMDEA().fit(data)


def test_reference_zeros_are_allowed_when_not_denominators() -> None:
    evaluated = DEAData(
        inputs=[[2.0, 2.0]],
        outputs=[[1.0]],
        input_names=("x1", "x2"),
        output_names=("y",),
    )
    reference = DEAData(
        inputs=[[1.0, 0.0], [0.0, 1.0]],
        outputs=[[1.0], [1.0]],
        input_names=("x1", "x2"),
        output_names=("y",),
    )

    result = SBMDEA(rts="vrs").fit(evaluated, reference=reference)

    assert result.rho == pytest.approx([0.25])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rts": "magic"},
        {"tolerance": 0.0},
        {"tolerance": np.inf},
    ],
)
def test_sbm_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ModelValidationError):
        SBMDEA(**kwargs)  # type: ignore[arg-type]


def test_sbm_validates_reference_contract(dominance_data: DEAData) -> None:
    wrong_dimension = DEAData(
        inputs=[[1.0, 1.0]],
        outputs=[[1.0]],
        input_names=("cost", "other"),
        output_names=("service",),
    )
    wrong_names = DEAData(
        inputs=[[1.0]],
        outputs=[[1.0]],
        input_names=("other",),
        output_names=("service",),
    )

    with pytest.raises(ModelValidationError, match="same number of inputs"):
        SBMDEA().fit(dominance_data, reference=wrong_dimension)
    with pytest.raises(ModelValidationError, match="names must match"):
        SBMDEA().fit(dominance_data, reference=wrong_names)
    with pytest.raises(ModelValidationError, match="data must"):
        SBMDEA().fit(object())  # type: ignore[arg-type]
    with pytest.raises(ModelValidationError, match="reference must"):
        SBMDEA().fit(dominance_data, reference=object())  # type: ignore[arg-type]


def test_sbm_reports_true_infeasibility_from_restricted_reference() -> None:
    evaluated = DEAData(
        inputs=[[0.5]],
        outputs=[[1.0]],
        input_names=("x",),
        output_names=("y",),
    )
    reference = DEAData(
        inputs=[[1.0]],
        outputs=[[1.0]],
        input_names=("x",),
        output_names=("y",),
    )

    with pytest.raises(SolverError, match=r"failed with infeasible"):
        SBMDEA(rts="vrs").fit(evaluated, reference=reference)


class _TamperingSolver:
    name = "tampered-scipy-highs"

    def solve(self, program: LinearProgram) -> LinearSolution:
        solved = ScipyHighsSolver().solve(program)
        variables = np.array(solved.variables, copy=True)
        variables[0] += 0.2
        return replace(solved, variables=variables)


def test_independent_audit_rejects_a_tampered_optimal_solution(
    dominance_data: DEAData,
) -> None:
    with pytest.raises(NumericalError, match="Post-solve SBM audit"):
        SBMDEA(rts="crs", solver=_TamperingSolver()).fit(dominance_data)


def test_native_highs_matches_scipy_for_sbm(dominance_data: DEAData) -> None:
    scipy_result = SBMDEA(rts="vrs", solver=ScipyHighsSolver()).fit(dominance_data)
    native_result = SBMDEA(rts="vrs", solver=HighsSolver()).fit(dominance_data)

    assert native_result.rho == pytest.approx(scipy_result.rho)
    assert native_result.lambdas == pytest.approx(scipy_result.lambdas)
    assert native_result.input_slacks == pytest.approx(scipy_result.input_slacks)
