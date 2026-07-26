from __future__ import annotations

import json

import numpy as np
import pytest

from frontierguard._data import DEAData
from frontierguard._errors import ModelValidationError
from frontierguard._solver import HighsSolver, ScipyHighsSolver
from frontierguard.additive import WeightedAdditiveDEA
from frontierguard.directional import DirectionalDEA


@pytest.fixture
def dominance_data() -> DEAData:
    return DEAData(
        inputs=[[1.0], [2.0]],
        outputs=[[2.0], [1.0]],
        dmu_names=("A", "B"),
        input_names=("cost",),
        output_names=("service",),
    )


def test_weighted_additive_vrs_oracle_and_projection(
    dominance_data: DEAData,
) -> None:
    result = WeightedAdditiveDEA(rts="vrs").fit(dominance_data)

    assert result.inefficiency == pytest.approx([0.0, 1.0])
    assert result.input_weights == pytest.approx([0.5])
    assert result.output_weights == pytest.approx([0.5])
    assert result.lambdas[1] == pytest.approx([1.0, 0.0])
    assert result.input_slacks[1] == pytest.approx([1.0])
    assert result.output_slacks[1] == pytest.approx([1.0])
    assert result.input_targets[1] == pytest.approx([1.0])
    assert result.output_targets[1] == pytest.approx([2.0])
    assert np.max(result.audit_residual) <= result.audit_tolerance

    reconstructed_inputs = result.lambdas @ dominance_data.inputs
    reconstructed_outputs = result.lambdas @ dominance_data.outputs
    assert result.input_targets == pytest.approx(reconstructed_inputs)
    assert result.output_targets == pytest.approx(reconstructed_outputs)
    assert result.input_targets + result.input_slacks == pytest.approx(
        dominance_data.inputs
    )
    assert result.output_targets - result.output_slacks == pytest.approx(
        dominance_data.outputs
    )


def test_weighted_additive_raw_weights_are_used_without_rescaling(
    dominance_data: DEAData,
) -> None:
    result = WeightedAdditiveDEA(
        rts="vrs",
        weighting="raw",
        input_weights=[2.0],
        output_weights=[3.0],
    ).fit(dominance_data)

    assert result.inefficiency == pytest.approx([0.0, 5.0])
    assert result.input_weights == pytest.approx([2.0])
    assert result.output_weights == pytest.approx([3.0])
    assert result.to_frame().loc["B", "additive_inefficiency"] == pytest.approx(5.0)
    json.dumps(result.to_dict(), allow_nan=False)


def test_unit_normalized_additive_is_invariant_to_column_units(
    dominance_data: DEAData,
) -> None:
    baseline = WeightedAdditiveDEA(rts="vrs").fit(dominance_data)
    rescaled = DEAData(
        inputs=np.asarray(dominance_data.inputs) * 10.0,
        outputs=np.asarray(dominance_data.outputs) * 0.1,
        dmu_names=dominance_data.dmu_names,
        input_names=dominance_data.input_names,
        output_names=dominance_data.output_names,
    )

    transformed = WeightedAdditiveDEA(rts="vrs").fit(rescaled)

    assert transformed.inefficiency == pytest.approx(baseline.inefficiency)
    assert transformed.input_slacks == pytest.approx(baseline.input_slacks * 10.0)
    assert transformed.output_slacks == pytest.approx(baseline.output_slacks * 0.1)


@pytest.mark.parametrize("rts", ["crs", "vrs", "nirs", "ndrs"])
def test_weighted_additive_respects_declared_rts(
    dominance_data: DEAData,
    rts: str,
) -> None:
    result = WeightedAdditiveDEA(rts=rts).fit(dominance_data)
    intensity_sum = result.lambdas.sum(axis=1)

    if rts == "vrs":
        assert intensity_sum == pytest.approx([1.0, 1.0])
    elif rts == "nirs":
        assert np.all(intensity_sum <= 1.0 + 1e-8)
    elif rts == "ndrs":
        assert np.all(intensity_sum >= 1.0 - 1e-8)
    assert np.all(result.input_slacks >= 0.0)
    assert np.all(result.output_slacks >= 0.0)
    assert np.max(result.audit_residual) <= result.audit_tolerance


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weighting": "raw"},
        {
            "weighting": "unit_normalized",
            "input_weights": [1.0],
        },
        {
            "weighting": "raw",
            "input_weights": [-1.0],
            "output_weights": [1.0],
        },
        {
            "weighting": "raw",
            "input_weights": [0.0],
            "output_weights": [0.0],
        },
        {
            "weighting": "raw",
            "input_weights": [1.0, 2.0],
            "output_weights": [1.0],
        },
        {"weighting": "unknown"},
    ],
)
def test_weighted_additive_rejects_ambiguous_or_invalid_weights(
    dominance_data: DEAData,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ModelValidationError):
        WeightedAdditiveDEA(**kwargs).fit(dominance_data)  # type: ignore[arg-type]


@pytest.mark.parametrize("rts", ["crs", "vrs", "nirs", "ndrs"])
def test_directional_distance_oracle_under_each_rts(
    dominance_data: DEAData,
    rts: str,
) -> None:
    result = DirectionalDEA(rts=rts).fit(
        dominance_data,
        input_direction=[1.0],
        output_direction=[1.0],
    )

    assert result.beta == pytest.approx([0.0, 1.0])
    assert result.input_targets[1] == pytest.approx([1.0])
    assert result.output_targets[1] == pytest.approx([2.0])
    assert np.max(result.audit_residual) <= result.audit_tolerance
    intensity_sum = result.lambdas.sum(axis=1)
    if rts == "vrs":
        assert intensity_sum == pytest.approx([1.0, 1.0])
    elif rts == "nirs":
        assert np.all(intensity_sum <= 1.0 + 1e-8)
    elif rts == "ndrs":
        assert np.all(intensity_sum >= 1.0 - 1e-8)


def test_directional_targets_and_slacks_reconstruct_feasibility() -> None:
    data = DEAData(
        inputs=[[1.0, 4.0], [2.0, 3.0], [4.0, 1.0], [4.0, 4.0]],
        outputs=[[2.0], [2.0], [2.0], [1.0]],
        dmu_names=("A", "B", "C", "D"),
    )
    input_direction = np.asarray([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0], [1.0, 1.0]])
    output_direction = np.asarray([[1.0], [1.0], [1.0], [0.5]])
    result = DirectionalDEA(rts="vrs").fit(
        data,
        input_direction=input_direction,
        output_direction=output_direction,
    )

    directional_inputs = data.inputs - result.beta[:, None] * result.input_directions
    directional_outputs = data.outputs + result.beta[:, None] * result.output_directions
    reconstructed_inputs = result.lambdas @ data.inputs
    reconstructed_outputs = result.lambdas @ data.outputs
    assert result.input_targets == pytest.approx(reconstructed_inputs)
    assert result.output_targets == pytest.approx(reconstructed_outputs)
    assert result.input_targets + result.input_slacks == pytest.approx(
        directional_inputs
    )
    assert result.output_targets - result.output_slacks == pytest.approx(
        directional_outputs
    )
    assert np.all(result.input_slacks >= 0.0)
    assert np.all(result.output_slacks >= 0.0)


def test_directional_beta_is_free_for_off_technology_external_observation() -> None:
    evaluated = DEAData(
        inputs=[[0.5]],
        outputs=[[1.0]],
        dmu_names=("evaluated",),
        input_names=("resource",),
        output_names=("service",),
    )
    reference = DEAData(
        inputs=[[1.0]],
        outputs=[[1.0]],
        dmu_names=("reference",),
        input_names=("resource",),
        output_names=("service",),
    )

    for solver in (ScipyHighsSolver(), HighsSolver()):
        result = DirectionalDEA(rts="vrs", solver=solver).fit(
            evaluated,
            reference=reference,
            input_direction=[1.0],
            output_direction=[0.0],
        )

        assert result.beta == pytest.approx([-0.5])
        assert result.lambdas == pytest.approx(np.asarray([[1.0]]))
        assert result.input_targets == pytest.approx(np.asarray([[1.0]]))
        assert result.output_targets == pytest.approx(np.asarray([[1.0]]))
        assert result.input_slacks == pytest.approx(np.asarray([[0.0]]))
        assert result.output_slacks == pytest.approx(np.asarray([[0.0]]))
        assert np.max(result.audit_residual) <= result.audit_tolerance
        json.dumps(result.to_dict(), allow_nan=False)


def test_directional_vector_broadcast_and_unit_covariance(
    dominance_data: DEAData,
) -> None:
    vector = DirectionalDEA(rts="vrs").fit(
        dominance_data,
        input_direction=[1.0],
        output_direction=[1.0],
    )
    matrix = DirectionalDEA(rts="vrs").fit(
        dominance_data,
        input_direction=[[1.0], [1.0]],
        output_direction=[[1.0], [1.0]],
    )
    scaled_data = DEAData(
        inputs=np.asarray(dominance_data.inputs) * 10.0,
        outputs=np.asarray(dominance_data.outputs) * 0.1,
        dmu_names=dominance_data.dmu_names,
        input_names=dominance_data.input_names,
        output_names=dominance_data.output_names,
    )
    scaled = DirectionalDEA(rts="vrs").fit(
        scaled_data,
        input_direction=[10.0],
        output_direction=[0.1],
    )

    assert matrix.beta == pytest.approx(vector.beta)
    assert scaled.beta == pytest.approx(vector.beta)
    assert matrix.input_directions.shape == (2, 1)
    assert matrix.output_directions.shape == (2, 1)
    json.dumps(vector.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("input_direction", "output_direction"),
    [
        ([1.0, 2.0], [1.0]),
        ([1.0], [1.0, 2.0]),
        ([-1.0], [1.0]),
        ([1.0], [np.nan]),
        ([0.0], [0.0]),
        ([[1.0]], [[1.0], [1.0]]),
        ([[1.0], [1.0]], [[1.0]]),
        ([[1.0], [0.0]], [[1.0], [0.0]]),
    ],
)
def test_directional_rejects_invalid_direction_contracts(
    dominance_data: DEAData,
    input_direction: list[float] | list[list[float]],
    output_direction: list[float] | list[list[float]],
) -> None:
    with pytest.raises(ModelValidationError):
        DirectionalDEA().fit(
            dominance_data,
            input_direction=input_direction,
            output_direction=output_direction,
        )


def test_new_result_arrays_are_copied_and_read_only(
    dominance_data: DEAData,
) -> None:
    additive = WeightedAdditiveDEA().fit(dominance_data)
    directional = DirectionalDEA().fit(
        dominance_data,
        input_direction=[1.0],
        output_direction=[1.0],
    )

    with pytest.raises(ValueError, match="read-only"):
        additive.inefficiency[0] = 10.0
    with pytest.raises(ValueError, match="read-only"):
        directional.beta[0] = 10.0


def test_models_validate_configuration_and_reference_names(
    dominance_data: DEAData,
) -> None:
    mismatched = DEAData(
        inputs=[[1.0]],
        outputs=[[2.0]],
        input_names=("other_input",),
        output_names=("service",),
    )

    with pytest.raises(ModelValidationError, match="input names"):
        WeightedAdditiveDEA().fit(dominance_data, reference=mismatched)
    with pytest.raises(ModelValidationError, match="input names"):
        DirectionalDEA().fit(
            dominance_data,
            reference=mismatched,
            input_direction=[1.0],
            output_direction=[1.0],
        )
    with pytest.raises(ModelValidationError):
        WeightedAdditiveDEA(rts="invalid")
    with pytest.raises(ModelValidationError):
        DirectionalDEA(tolerance=0.0)
