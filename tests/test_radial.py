from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import frontierguard as fg


@pytest.fixture
def simple_data() -> fg.DEAData:
    return fg.DEAData(
        inputs=[[1.0], [2.0], [3.0]],
        outputs=[[1.0], [1.0], [2.0]],
        dmu_names=("A", "B", "C"),
        input_names=("staff",),
        output_names=("cases",),
    )


def test_crs_input_scores_match_hand_calculation(simple_data: fg.DEAData) -> None:
    result = fg.RadialDEA(
        orientation="input",
        rts="crs",
        slacks="none",
    ).fit(simple_data)

    assert result.technical_efficiency == pytest.approx([1.0, 0.5, 2 / 3])
    assert result.radial_factor == pytest.approx([1.0, 0.5, 2 / 3])
    assert np.max(result.audit_residual) <= result.audit_tolerance
    assert result.peers("B") == {"A": pytest.approx(1.0)}
    assert result.targets("B") == {
        "inputs": {"staff": pytest.approx(1.0)},
        "outputs": {"cases": pytest.approx(1.0)},
    }


def test_classical_ccr_bcc_oracle() -> None:
    data = fg.DEAData(
        inputs=[[1.0], [2.0], [4.0]],
        outputs=[[1.0], [3.0], [4.0]],
        dmu_names=("A", "B", "C"),
    )

    crs = fg.RadialDEA(rts="crs", slacks="none").fit(data)
    vrs = fg.RadialDEA(rts="vrs", slacks="none").fit(data)

    assert crs.technical_efficiency == pytest.approx([2 / 3, 1.0, 2 / 3])
    assert vrs.technical_efficiency == pytest.approx([1.0, 1.0, 1.0])


def test_cross_language_five_dmu_radial_oracle() -> None:
    """Published pyDEA/Julia fixture fixes score and reciprocal conventions."""
    data = fg.DEAData(
        inputs=[[2, 5], [2, 4], [6, 6], [3, 2], [6, 2]],
        outputs=[[1], [2], [3], [1], [2]],
        dmu_names=("A", "B", "C", "D", "E"),
    )

    input_result = fg.RadialDEA(
        orientation="input",
        rts="crs",
        slacks="none",
    ).fit(data)
    output_result = fg.RadialDEA(
        orientation="output",
        rts="crs",
        slacks="none",
    ).fit(data)
    vrs_result = fg.RadialDEA(
        orientation="input",
        rts="vrs",
        slacks="none",
    ).fit(data)

    expected = [0.5, 1.0, 5 / 6, 5 / 7, 1.0]
    assert input_result.technical_efficiency == pytest.approx(expected)
    assert output_result.radial_factor == pytest.approx([2.0, 1.0, 1.2, 1.4, 1.0])
    assert output_result.technical_efficiency == pytest.approx(expected)
    assert vrs_result.technical_efficiency == pytest.approx(np.ones(5))


def test_vrs_orientation_conventions_are_explicit(simple_data: fg.DEAData) -> None:
    input_result = fg.RadialDEA(
        orientation="input",
        rts="vrs",
    ).fit(simple_data)
    output_result = fg.RadialDEA(
        orientation="output",
        rts="vrs",
    ).fit(simple_data)

    assert input_result.technical_efficiency == pytest.approx([1.0, 0.5, 1.0])
    assert output_result.radial_factor == pytest.approx([1.0, 1.5, 1.0])
    assert output_result.technical_efficiency == pytest.approx([1.0, 2 / 3, 1.0])
    assert np.allclose(input_result.lambdas.sum(axis=1), 1.0)
    assert np.allclose(output_result.lambdas.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    ("rts", "comparison"),
    [
        ("nirs", "at_most"),
        ("ndrs", "at_least"),
    ],
)
def test_returns_to_scale_intensity_contract(
    simple_data: fg.DEAData,
    rts: str,
    comparison: str,
) -> None:
    result = fg.RadialDEA(rts=rts).fit(simple_data)
    sums = result.lambdas.sum(axis=1)

    if comparison == "at_most":
        assert np.all(sums <= 1.0 + 1e-8)
    else:
        assert np.all(sums >= 1.0 - 1e-8)


def test_phase_two_targets_are_feasible() -> None:
    data = fg.DEAData(
        inputs=[[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [3.0, 4.0]],
        outputs=[[1.0], [1.0], [1.0], [1.0]],
        dmu_names=("A", "B", "C", "D"),
    )
    result = fg.RadialDEA(
        orientation="input",
        rts="vrs",
        slacks="normalized",
    ).fit(data)

    reconstructed_inputs = result.lambdas @ np.asarray(data.inputs)
    reconstructed_outputs = result.lambdas @ np.asarray(data.outputs)
    assert result.input_targets == pytest.approx(reconstructed_inputs)
    assert result.output_targets == pytest.approx(reconstructed_outputs)
    assert np.all(
        result.input_targets
        <= result.radial_factor[:, None] * np.asarray(data.inputs) + 1e-8
    )
    assert np.all(result.output_targets >= np.asarray(data.outputs) - 1e-8)
    assert np.all(result.input_slacks >= 0)
    assert np.all(result.output_slacks >= 0)


def test_separate_reference_technology_and_leave_one_out(
    simple_data: fg.DEAData,
) -> None:
    external = fg.DEAData(
        inputs=[[1.0]],
        outputs=[[1.0]],
        dmu_names=("benchmark",),
        input_names=("staff",),
        output_names=("cases",),
    )
    external_result = fg.RadialDEA(rts="crs", slacks="none").fit(
        simple_data,
        reference=external,
    )
    super_result = fg.RadialDEA(rts="crs", slacks="none").fit(
        simple_data,
        exclude_self=True,
    )

    assert external_result.technical_efficiency == pytest.approx([1.0, 0.5, 2 / 3])
    assert super_result.technical_efficiency[0] == pytest.approx(1.5)
    assert "A" not in super_result.peers("A")


def test_from_frame_and_strict_serialization() -> None:
    frame = pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 1.0, 2.0],
        }
    )
    data = fg.DEAData.from_frame(
        frame,
        inputs=["x"],
        outputs=["y"],
        dmu="name",
    )
    result = fg.dea(data, orientation="input", rts="crs")
    exported = result.to_dict()

    assert data.fingerprint == result.data_fingerprint
    assert result.to_frame().index.tolist() == ["A", "B", "C"]
    json.dumps(exported, allow_nan=False)
    assert exported["audit_tolerance"] == pytest.approx(result.audit_tolerance)


def test_input_arrays_are_copied_and_read_only() -> None:
    source = np.array([[1.0], [2.0]])
    data = fg.DEAData(source, [[1.0], [1.0]])
    source[0, 0] = 99.0

    assert data.inputs[0, 0] == pytest.approx(1.0)  # type: ignore[index]
    with pytest.raises(ValueError, match="read-only"):
        data.inputs[0, 0] = 2.0  # type: ignore[index]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"orientation": "sideways"},
        {"rts": "magic"},
        {"slacks": "sometimes"},
        {"tolerance": 0.0},
    ],
)
def test_invalid_model_configuration_fails(kwargs: dict[str, object]) -> None:
    with pytest.raises(fg.ModelValidationError):
        fg.RadialDEA(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("inputs", "outputs", "message"),
    [
        ([[1.0], [2.0]], [[1.0]], "same number"),
        ([[1.0], [np.nan]], [[1.0], [2.0]], "non-finite"),
        ([[1.0], [-1.0]], [[1.0], [2.0]], "non-negative"),
        ([[0.0], [0.0]], [[1.0], [2.0]], "positive"),
    ],
)
def test_invalid_data_fail(
    inputs: list[list[float]],
    outputs: list[list[float]],
    message: str,
) -> None:
    with pytest.raises(fg.DataValidationError, match=message):
        fg.DEAData(inputs=inputs, outputs=outputs)
