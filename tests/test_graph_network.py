from __future__ import annotations

import json

import numpy as np
import pytest

import frontierguard as fg


def _serial_graph() -> fg.NetworkData:
    return fg.NetworkData(
        stages=(
            fg.NetworkStage("upstream", inputs=[[1.0], [2.0]]),
            fg.NetworkStage("downstream", outputs=[[1.0], [2.0]]),
        ),
        links=(
            fg.NetworkLink(
                "cases",
                "upstream",
                "downstream",
                [[1.0], [4.0]],
            ),
        ),
        dmu_names=("A", "B"),
    )


def test_graph_engine_matches_serial_internal_mismatch_oracle() -> None:
    result = fg.RelationalNetworkDEA().fit(_serial_graph())

    assert result.overall_efficiency == pytest.approx([0.5, 0.5])
    np.testing.assert_allclose(
        result.stage_efficiency,
        [[0.5, 1.0], [1.0, 0.5]],
    )
    assert result.closure_error == pytest.approx([0.0, 0.0], abs=1e-11)
    assert np.max(result.audit_residual) <= result.audit_tolerance
    json.dumps(result.to_dict(), allow_nan=False)


def test_branched_graph_gap_decomposition_and_unit_invariance() -> None:
    stages = (
        fg.NetworkStage("source", inputs=[[1.0], [2.0], [3.0]]),
        fg.NetworkStage("left", outputs=[[1.0], [1.0], [1.4]]),
        fg.NetworkStage("right", outputs=[[1.0], [2.0], [1.0]]),
    )
    links = (
        fg.NetworkLink("left_flow", "source", "left", [[1.0], [2.0], [2.0]]),
        fg.NetworkLink("right_flow", "source", "right", [[1.0], [1.0], [2.0]]),
    )
    original = fg.NetworkData(stages, links, dmu_names=("A", "B", "C"))
    scaled = fg.NetworkData(
        (
            fg.NetworkStage("source", inputs=np.asarray(stages[0].inputs) * 100),
            fg.NetworkStage("left", outputs=np.asarray(stages[1].outputs) * 0.25),
            fg.NetworkStage("right", outputs=np.asarray(stages[2].outputs) * 17),
        ),
        (
            fg.NetworkLink(
                "left_flow",
                "source",
                "left",
                np.asarray(links[0].values) * 4,
            ),
            fg.NetworkLink(
                "right_flow",
                "source",
                "right",
                np.asarray(links[1].values) * 0.1,
            ),
        ),
        dmu_names=("A", "B", "C"),
    )

    first = fg.RelationalNetworkDEA().fit(original)
    second = fg.RelationalNetworkDEA().fit(scaled)
    black_box = fg.RadialDEA(rts="crs", slacks="none").fit(
        fg.DEAData(
            inputs=stages[0].inputs,
            outputs=np.column_stack((stages[1].outputs, stages[2].outputs)),
            dmu_names=original.dmu_names,
        )
    )

    assert first.overall_efficiency == pytest.approx(
        second.overall_efficiency,
        rel=1e-8,
        abs=1e-10,
    )
    assert 1.0 - first.overall_efficiency == pytest.approx(
        first.stage_gap.sum(axis=1),
        abs=1e-10,
    )
    assert np.all(first.stage_gap >= -1e-10)
    assert np.all(
        first.overall_efficiency
        <= black_box.technical_efficiency + first.audit_tolerance
    )
    assert any(
        value is None for row in first.to_dict()["stage_efficiency"] for value in row
    )
    json.dumps(first.to_dict(), allow_nan=False)


def test_graph_validation_rejects_cycles_and_incomplete_stages() -> None:
    a = fg.NetworkStage("a", inputs=[[1.0]], outputs=[[1.0]])
    b = fg.NetworkStage("b", inputs=[[1.0]], outputs=[[1.0]])
    with pytest.raises(fg.DataValidationError, match="acyclic"):
        fg.NetworkData(
            (a, b),
            (
                fg.NetworkLink("ab", "a", "b", [[1.0]]),
                fg.NetworkLink("ba", "b", "a", [[1.0]]),
            ),
        )
    with pytest.raises(fg.DataValidationError, match="no external input"):
        fg.NetworkData(
            (
                fg.NetworkStage("source", outputs=[[1.0]]),
                fg.NetworkStage("sink", outputs=[[1.0]]),
            ),
            (fg.NetworkLink("flow", "source", "sink", [[1.0]]),),
        )


def test_graph_input_contract_and_weight_floor() -> None:
    data = _serial_graph()
    source = np.array([[1.0], [2.0]])
    stage = fg.NetworkStage("copy", inputs=source, outputs=[[1.0], [1.0]])
    source[0, 0] = 99.0

    assert stage.inputs is not None
    assert stage.inputs[0, 0] == pytest.approx(1.0)
    assert data.topological_order == ("upstream", "downstream")
    assert fg.RelationalNetworkDEA().weight_floor == 0.0
    with pytest.raises(fg.ModelValidationError, match="weight_floor"):
        fg.RelationalNetworkDEA(weight_floor=-1.0)


def test_graph_fingerprint_includes_link_topology() -> None:
    stages = (
        fg.NetworkStage("a", inputs=[[1.0]], outputs=[[1.0]]),
        fg.NetworkStage("b", inputs=[[1.0]], outputs=[[1.0]]),
    )
    forward = fg.NetworkData(
        stages,
        (fg.NetworkLink("flow", "a", "b", [[1.0]]),),
    )
    reverse = fg.NetworkData(
        stages,
        (fg.NetworkLink("flow", "b", "a", [[1.0]]),),
    )

    assert forward.fingerprint != reverse.fingerprint
