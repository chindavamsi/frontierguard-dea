from __future__ import annotations

import json

import numpy as np
import pytest

from frontierguard import DEAData, ModelValidationError, SolverError
from frontierguard._enums import SolverStatus
from frontierguard._solver import LinearProgram, LinearSolution
from frontierguard.productivity import (
    MalmquistDEA,
    malmquist,
    malmquist_panel,
)


@pytest.fixture
def published_panel() -> tuple[DEAData, DEAData]:
    names = ("A", "B", "C", "D", "E")
    period_t = DEAData(
        inputs=[[2], [3], [5], [4], [4]],
        outputs=[[1], [4], [6], [3], [5]],
        dmu_names=names,
        input_names=("x",),
        output_names=("y",),
    )
    period_t1 = DEAData(
        inputs=[[1], [2], [4], [3], [4]],
        outputs=[[1], [4], [6], [3], [3]],
        dmu_names=names,
        input_names=("x",),
        output_names=("y",),
    )
    return period_t, period_t1


@pytest.mark.parametrize("orientation", ["input", "output"])
def test_published_malmquist_oracle(
    published_panel: tuple[DEAData, DEAData],
    orientation: str,
) -> None:
    period_t, period_t1 = published_panel
    result = malmquist(
        period_t,
        period_t1,
        orientation=orientation,
        rts="crs",
        period_labels=("year_1", "year_2"),
    )

    assert result.malmquist_index == pytest.approx([2, 1.5, 1.25, 4 / 3, 0.6])
    assert result.efficiency_change == pytest.approx([4 / 3, 1, 5 / 6, 8 / 9, 0.4])
    assert result.technical_change == pytest.approx([1.5] * 5)
    assert result.malmquist_index == pytest.approx(
        result.efficiency_change * result.technical_change
    )
    assert result.ec is result.efficiency_change
    assert result.tc is result.technical_change
    assert result.m is result.malmquist_index
    assert result.transition_label == "year_1->year_2"
    assert result.distance_convention == "normalized_radial_efficiency"
    assert result.distance_terms == (
        "distance_t_at_t",
        "distance_t1_at_t",
        "distance_t_at_t1",
        "distance_t1_at_t1",
    )
    assert np.max(result.audit_residual) <= result.audit_tolerance


def test_four_normalized_distance_terms_match_oracle(
    published_panel: tuple[DEAData, DEAData],
) -> None:
    period_t, period_t1 = published_panel
    result = MalmquistDEA(orientation="input", rts="crs").fit(
        period_t,
        period_t1,
    )

    assert result.distance_t_at_t == pytest.approx([0.375, 1.0, 0.9, 0.5625, 0.9375])
    assert result.distance_t1_at_t == pytest.approx([0.75, 1.5, 1.125, 0.75, 0.5625])
    assert result.distance_t_at_t1 == pytest.approx([0.25, 2 / 3, 0.6, 0.375, 0.625])
    assert result.distance_t1_at_t1 == pytest.approx([0.5, 1.0, 0.75, 0.5, 0.375])


@pytest.mark.parametrize(
    ("orientation", "rts"),
    [
        ("input", "crs"),
        ("output", "crs"),
        ("input", "vrs"),
        ("output", "vrs"),
    ],
)
def test_stationary_periods_equal_one(
    published_panel: tuple[DEAData, DEAData],
    orientation: str,
    rts: str,
) -> None:
    period_t, _ = published_panel
    result = MalmquistDEA(orientation=orientation, rts=rts).fit(
        period_t,
        period_t,
    )

    assert result.efficiency_change == pytest.approx(np.ones(5))
    assert result.technical_change == pytest.approx(np.ones(5))
    assert result.malmquist_index == pytest.approx(np.ones(5))


@pytest.mark.parametrize("orientation", ["input", "output"])
def test_common_factor_unit_changes_are_invariant(
    published_panel: tuple[DEAData, DEAData],
    orientation: str,
) -> None:
    period_t, period_t1 = published_panel
    baseline = MalmquistDEA(orientation=orientation, rts="crs").fit(
        period_t,
        period_t1,
    )
    rescaled_t = DEAData(
        inputs=np.asarray(period_t.inputs) * 1_000,
        outputs=np.asarray(period_t.outputs) * 0.01,
        dmu_names=period_t.dmu_names,
        input_names=period_t.input_names,
        output_names=period_t.output_names,
    )
    rescaled_t1 = DEAData(
        inputs=np.asarray(period_t1.inputs) * 1_000,
        outputs=np.asarray(period_t1.outputs) * 0.01,
        dmu_names=period_t1.dmu_names,
        input_names=period_t1.input_names,
        output_names=period_t1.output_names,
    )
    rescaled = MalmquistDEA(orientation=orientation, rts="crs").fit(
        rescaled_t,
        rescaled_t1,
    )

    assert rescaled.malmquist_index == pytest.approx(baseline.malmquist_index)
    assert rescaled.efficiency_change == pytest.approx(baseline.efficiency_change)
    assert rescaled.technical_change == pytest.approx(baseline.technical_change)


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        (
            DEAData(
                [[1], [2]],
                [[1], [2]],
                dmu_names=("B", "A"),
                input_names=("x",),
                output_names=("y",),
            ),
            "DMU names and order",
        ),
        (
            DEAData(
                [[1], [2]],
                [[1], [2]],
                dmu_names=("A", "B"),
                input_names=("capital",),
                output_names=("y",),
            ),
            "input names and order",
        ),
        (
            DEAData(
                [[1], [2]],
                [[1], [2]],
                dmu_names=("A", "B"),
                input_names=("x",),
                output_names=("cases",),
            ),
            "output names and order",
        ),
    ],
)
def test_period_alignment_is_exact(changed: DEAData, message: str) -> None:
    base = DEAData(
        [[1], [2]],
        [[1], [2]],
        dmu_names=("A", "B"),
        input_names=("x",),
        output_names=("y",),
    )

    with pytest.raises(ModelValidationError, match=message):
        MalmquistDEA().fit(base, changed)


def test_vrs_is_supported_but_other_rts_contracts_are_rejected(
    published_panel: tuple[DEAData, DEAData],
) -> None:
    period_t, period_t1 = published_panel
    result = MalmquistDEA(rts="vrs").fit(period_t, period_t1)

    assert result.returns_to_scale.value == "vrs"
    with pytest.raises(ModelValidationError, match="only 'crs' and 'vrs'"):
        MalmquistDEA(rts="nirs")


class _FailingSolver:
    name = "always-fails"

    def solve(self, program: LinearProgram) -> LinearSolution:
        size = np.asarray(program.objective).size
        return LinearSolution(
            status=SolverStatus.ERROR,
            variables=np.full(size, np.nan),
            objective=float("nan"),
            message="injected failure",
            raw_status=99,
        )


def test_solver_failure_identifies_the_distance_term(
    published_panel: tuple[DEAData, DEAData],
) -> None:
    period_t, period_t1 = published_panel

    with pytest.raises(
        SolverError,
        match=r"Malmquist term distance_t_at_t failed.*injected failure",
    ):
        MalmquistDEA(solver=_FailingSolver()).fit(period_t, period_t1)


def test_result_is_immutable_audited_and_strict_json(
    published_panel: tuple[DEAData, DEAData],
) -> None:
    period_t, period_t1 = published_panel
    result = malmquist(period_t, period_t1)

    with pytest.raises(ValueError, match="read-only"):
        result.malmquist_index[0] = 99
    with pytest.raises(ValueError, match="read-only"):
        result.distance_audit_residual[0, 0] = 99
    assert result.malmquist_index == pytest.approx(
        result.efficiency_change * result.technical_change
    )
    assert result.audit_residual == pytest.approx(
        np.maximum(
            result.distance_audit_residual.max(axis=1),
            result.decomposition_residual,
        )
    )
    json.dumps(result.to_dict(), allow_nan=False)
    frame = result.to_frame()
    assert frame["dmu"].tolist() == list(period_t.dmu_names)
    assert frame["transition"].unique().tolist() == ["t->t+1"]


def test_panel_convenience_returns_transition_labeled_immutable_arrays(
    published_panel: tuple[DEAData, DEAData],
) -> None:
    period_t, period_t1 = published_panel
    period_t2 = DEAData(
        inputs=np.asarray(period_t1.inputs) * 0.8,
        outputs=period_t1.outputs,
        dmu_names=period_t1.dmu_names,
        input_names=period_t1.input_names,
        output_names=period_t1.output_names,
    )
    result = malmquist_panel(
        (period_t, period_t1, period_t2),
        period_labels=("2024", "2025", "2026"),
    )

    assert result.transition_labels == ("2024->2025", "2025->2026")
    assert result.distance_convention == "normalized_radial_efficiency"
    assert len(result.distance_terms) == 4
    assert result.ec is result.efficiency_change
    assert result.tc is result.technical_change
    assert result.m is result.malmquist_index
    assert result.malmquist_index.shape == (2, 5)
    assert result.distance_audit_residual.shape == (2, 5, 4)
    assert result.malmquist_index[0] == pytest.approx([2, 1.5, 1.25, 4 / 3, 0.6])
    assert result.malmquist_index[1] == pytest.approx([1.25] * 5)
    assert np.max(result.audit_residual) <= result.audit_tolerance
    assert result.malmquist_index == pytest.approx(
        result.efficiency_change * result.technical_change
    )
    with pytest.raises(ValueError, match="read-only"):
        result.malmquist_index[0, 0] = 99

    frame = result.to_frame()
    assert frame.shape[0] == 10
    assert frame["transition"].tolist() == ["2024->2025"] * 5 + ["2025->2026"] * 5
    json.dumps(result.to_dict(), allow_nan=False)


def test_panel_validation() -> None:
    period = DEAData([[1], [2]], [[1], [2]], dmu_names=("A", "B"))

    with pytest.raises(ModelValidationError, match="at least two periods"):
        MalmquistDEA().fit_panel((period,))
    with pytest.raises(ModelValidationError, match="exactly 2 labels"):
        MalmquistDEA().fit_panel(
            (period, period),
            period_labels=("only_one",),
        )
    with pytest.raises(ModelValidationError, match="unique"):
        MalmquistDEA().fit_panel(
            (period, period),
            period_labels=("same", "same"),
        )


def test_adjacent_method_is_not_other_productivity_families() -> None:
    assert "four-distance" in (MalmquistDEA.__doc__ or "").lower()
    module_doc = __import__(
        "frontierguard.productivity",
        fromlist=["__doc__"],
    ).__doc__
    assert module_doc is not None
    assert "global Malmquist" in module_doc
    assert "Malmquist--Luenberger" in module_doc
    assert "dynamic/network DEA" in module_doc
