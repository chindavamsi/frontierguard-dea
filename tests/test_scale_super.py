from __future__ import annotations

import json

import numpy as np
import pytest

from frontierguard._data import DEAData
from frontierguard._enums import SolverStatus
from frontierguard._errors import ModelValidationError, SolverError
from frontierguard.scale import ScaleEfficiencyDEA, ScaleRegime
from frontierguard.super_efficiency import RadialSuperEfficiencyDEA


@pytest.fixture
def scale_oracle() -> DEAData:
    return DEAData(
        inputs=[[1.0], [2.0], [4.0]],
        outputs=[[1.0], [3.0], [4.0]],
        dmu_names=("A", "B", "C"),
        input_names=("x",),
        output_names=("y",),
    )


@pytest.mark.parametrize("orientation", ["input", "output"])
def test_scale_efficiency_and_nirs_regime_oracle(
    scale_oracle: DEAData,
    orientation: str,
) -> None:
    result = ScaleEfficiencyDEA(orientation=orientation).fit(scale_oracle)

    assert result.crs_efficiency == pytest.approx([2 / 3, 1.0, 2 / 3])
    assert result.vrs_efficiency == pytest.approx([1.0, 1.0, 1.0])
    assert result.nirs_efficiency == pytest.approx([2 / 3, 1.0, 1.0])
    assert result.scale_efficiency == pytest.approx([2 / 3, 1.0, 2 / 3])
    assert result.scale_regime == (
        ScaleRegime.IRS,
        ScaleRegime.CRS,
        ScaleRegime.DRS,
    )
    assert result.to_frame()["scale_regime"].tolist() == ["irs", "crs", "drs"]
    assert np.max(result.audit_residual) <= result.audit_tolerance


def test_scale_result_is_immutable_and_strict_json(scale_oracle: DEAData) -> None:
    result = ScaleEfficiencyDEA().fit(scale_oracle)

    with pytest.raises(ValueError, match="read-only"):
        result.scale_efficiency[0] = 0.0
    exported = result.to_dict()
    json.dumps(exported, allow_nan=False)
    assert exported["audit_residual"] == pytest.approx(result.audit_residual)
    assert exported["audit_tolerance"] == result.audit_tolerance
    with pytest.raises(ValueError, match="read-only"):
        result.audit_residual[0] = 0.0
    assert result.crs_result.returns_to_scale.value == "crs"
    assert result.vrs_result.returns_to_scale.value == "vrs"
    assert result.nirs_result.returns_to_scale.value == "nirs"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"orientation": "sideways"},
        {"tolerance": 0.0},
        {"classification_tolerance": float("nan")},
    ],
)
def test_scale_configuration_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ModelValidationError):
        ScaleEfficiencyDEA(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("orientation", ["input", "output"])
def test_crs_super_efficiency_preserves_scores_above_one(
    scale_oracle: DEAData,
    orientation: str,
) -> None:
    result = RadialSuperEfficiencyDEA(
        orientation=orientation,
        rts="crs",
    ).fit(scale_oracle)

    assert result.super_efficiency == pytest.approx([2 / 3, 1.5, 2 / 3])
    if orientation == "input":
        assert result.radial_factor == pytest.approx([2 / 3, 1.5, 2 / 3])
        assert result.input_targets[1] == pytest.approx([3.0])
        assert result.output_targets[1] == pytest.approx([3.0])
    else:
        assert result.radial_factor == pytest.approx([1.5, 2 / 3, 1.5])
        assert result.input_targets[1] == pytest.approx([2.0])
        assert result.output_targets[1] == pytest.approx([2.0])
    assert "B" not in result.peers("B")
    assert np.all(result.feasible)
    with pytest.raises(ValueError, match="read-only"):
        result.super_efficiency[0] = 0.0
    json.dumps(result.to_dict(), allow_nan=False)


def test_vrs_infeasibility_can_be_recorded_as_strict_json() -> None:
    data = DEAData(
        inputs=[[1.0], [2.0]],
        outputs=[[1.0], [2.0]],
        dmu_names=("A", "B"),
    )
    result = RadialSuperEfficiencyDEA(
        rts="vrs",
        on_infeasible="record",
    ).fit(data)
    exported = result.to_dict()

    assert result.super_efficiency[0] == pytest.approx(2.0)
    assert np.isnan(result.super_efficiency[1])
    assert result.statuses == (SolverStatus.OPTIMAL, SolverStatus.INFEASIBLE)
    assert result.feasible.tolist() == [True, False]
    assert exported["super_efficiency"] == [pytest.approx(2.0), None]
    assert exported["lambdas"][1] == [None, None]
    json.dumps(exported, allow_nan=False)
    with pytest.raises(SolverError, match="has no peer set"):
        result.peers("B")


def test_vrs_infeasibility_can_fail_fast() -> None:
    data = DEAData(
        inputs=[[1.0], [2.0]],
        outputs=[[1.0], [2.0]],
        dmu_names=("A", "B"),
    )

    with pytest.raises(SolverError, match="on_infeasible='record'"):
        RadialSuperEfficiencyDEA(rts="vrs").fit(data)


@pytest.mark.parametrize("orientation", ["input", "output"])
@pytest.mark.parametrize("rts", ["crs", "vrs", "nirs", "ndrs"])
def test_all_radial_rts_modes_have_explicit_per_dmu_status(
    scale_oracle: DEAData,
    orientation: str,
    rts: str,
) -> None:
    result = RadialSuperEfficiencyDEA(
        orientation=orientation,
        rts=rts,
        on_infeasible="record",
    ).fit(scale_oracle)

    assert len(result.statuses) == scale_oracle.n_dmus
    assert all(
        status in (SolverStatus.OPTIMAL, SolverStatus.INFEASIBLE)
        for status in result.statuses
    )
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"on_infeasible": "ignore"},
        {"tolerance": 0.0},
        {"rts": "magic"},
    ],
)
def test_super_efficiency_configuration_validation(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ModelValidationError):
        RadialSuperEfficiencyDEA(**kwargs)  # type: ignore[arg-type]


def test_super_efficiency_requires_a_nonempty_leave_one_out_set() -> None:
    data = DEAData(inputs=[[1.0]], outputs=[[1.0]], dmu_names=("only",))

    with pytest.raises(ModelValidationError, match="at least two"):
        RadialSuperEfficiencyDEA().fit(data)
