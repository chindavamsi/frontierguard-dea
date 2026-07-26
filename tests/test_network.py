from __future__ import annotations

import json

import numpy as np
import pytest

import frontierguard as fg


def _two_stage_data() -> fg.SerialNetworkData:
    return fg.SerialNetworkData(
        inputs=[[1.0], [2.0], [2.0]],
        links=([[1.0], [1.0], [2.0]],),
        outputs=[[1.0], [1.0], [1.5]],
        dmu_names=("A", "B", "C"),
        input_names=("resources",),
        link_names=(("service",),),
        output_names=("outcome",),
        stage_names=("operations", "delivery"),
    )


def test_two_stage_relational_model_matches_closed_form() -> None:
    data = _two_stage_data()
    result = fg.SerialNetworkDEA().fit(data)

    assert result.overall_efficiency == pytest.approx([1.0, 0.5, 0.75])
    np.testing.assert_allclose(
        result.stage_efficiency,
        [[1.0, 1.0], [0.5, 1.0], [1.0, 0.75]],
    )
    assert result.closure_error == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert np.max(result.audit_residual) <= result.audit_tolerance
    assert result.to_frame().columns.tolist() == [
        "overall_efficiency",
        "stage:operations",
        "stage:delivery",
        "closure_error",
        "status",
    ]
    json.dumps(result.to_dict(), allow_nan=False)


def test_published_style_internal_mismatch_oracle() -> None:
    """Relational DEA detects inefficiency hidden by a black-box CCR model."""
    network = fg.SerialNetworkData(
        inputs=[[1.0], [2.0]],
        links=([[1.0], [4.0]],),
        outputs=[[1.0], [2.0]],
        dmu_names=("A", "B"),
        stage_names=("upstream", "downstream"),
    )
    relational = fg.SerialNetworkDEA().fit(network)
    black_box = fg.RadialDEA(rts="crs", slacks="none").fit(
        fg.DEAData(
            inputs=network.inputs,
            outputs=network.outputs,
            dmu_names=network.dmu_names,
        )
    )

    assert relational.overall_efficiency == pytest.approx([0.5, 0.5])
    np.testing.assert_allclose(
        relational.stage_efficiency,
        [[0.5, 1.0], [1.0, 0.5]],
    )
    assert black_box.technical_efficiency == pytest.approx([1.0, 1.0])


def test_weight_floor_is_opt_in_model_variant() -> None:
    canonical = fg.SerialNetworkDEA()
    restricted = fg.SerialNetworkDEA(weight_floor=1e-8)

    assert canonical.weight_floor == 0.0
    assert restricted.weight_floor == pytest.approx(1e-8)


def test_general_three_stage_network_closes() -> None:
    data = fg.SerialNetworkData(
        inputs=[[1.0], [2.0], [3.0]],
        links=(
            [[1.0], [1.5], [2.0]],
            [[1.0], [1.2], [1.8]],
        ),
        outputs=[[1.0], [1.0], [1.5]],
        dmu_names=("A", "B", "C"),
        stage_names=("procure", "transform", "deliver"),
    )
    result = fg.SerialNetworkDEA().fit(data)

    assert result.stage_efficiency.shape == (3, 3)
    assert result.overall_efficiency == pytest.approx(
        np.prod(result.stage_efficiency, axis=1),
        rel=1e-9,
        abs=1e-11,
    )
    assert np.all(result.stage_efficiency <= 1.0 + 1e-8)


def test_network_scores_are_unit_invariant() -> None:
    original = _two_stage_data()
    rescaled = fg.SerialNetworkData(
        inputs=np.asarray(original.inputs) * 1000,
        links=(np.asarray(original.links[0]) * 0.01,),
        outputs=np.asarray(original.outputs) * 17,
        dmu_names=original.dmu_names,
        stage_names=original.stage_names,
    )

    first = fg.SerialNetworkDEA().fit(original)
    second = fg.SerialNetworkDEA().fit(rescaled)

    assert first.overall_efficiency == pytest.approx(
        second.overall_efficiency,
        rel=1e-9,
        abs=1e-11,
    )
    assert first.stage_efficiency == pytest.approx(
        second.stage_efficiency,
        rel=1e-9,
        abs=1e-11,
    )


def test_network_validation() -> None:
    with pytest.raises(fg.DataValidationError, match="at least one"):
        fg.SerialNetworkData(inputs=[[1.0]], links=(), outputs=[[1.0]])
    with pytest.raises(fg.DataValidationError, match="same DMUs"):
        fg.SerialNetworkData(
            inputs=[[1.0], [2.0]],
            links=([[1.0]],),
            outputs=[[1.0], [2.0]],
        )
    with pytest.raises(fg.ModelValidationError, match="weight_floor"):
        fg.SerialNetworkDEA(weight_floor=-1.0)
