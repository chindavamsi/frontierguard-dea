from __future__ import annotations

import json

import numpy as np
import pytest

import frontierguard as fg


def test_published_eleven_dmu_fdh_input_oracle() -> None:
    result = fg.FreeDisposalHullDEA().fit(fg.load_eleven_dmu())

    assert result.technical_efficiency == pytest.approx(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            8 / 9,
            1.0,
            1.0,
            1.0,
            1.0,
            25 / 42,
            1.0,
        ],
        rel=1e-8,
        abs=1e-9,
    )
    assert np.max(result.audit_residual) <= result.audit_tolerance
    json.dumps(result.to_dict(), allow_nan=False)


def test_fdh_input_output_reciprocal_conventions_and_peers() -> None:
    data = fg.DEAData(
        inputs=[[1.0], [2.0], [3.0]],
        outputs=[[1.0], [1.0], [2.0]],
        dmu_names=("A", "B", "C"),
    )

    input_result = fg.FreeDisposalHullDEA(orientation="input").fit(data)
    output_result = fg.FreeDisposalHullDEA(orientation="output").fit(data)

    assert input_result.technical_efficiency == pytest.approx([1.0, 0.5, 1.0])
    assert output_result.radial_factor == pytest.approx([1.0, 1.0, 1.0])
    assert output_result.technical_efficiency == pytest.approx([1.0, 1.0, 1.0])
    assert input_result.peers("B") == {"A": 1.0}
    assert input_result.lambdas.sum(axis=1) == pytest.approx(np.ones(3))


def test_fdh_is_unit_and_row_order_invariant() -> None:
    data = fg.load_eleven_dmu()
    permutation = np.asarray([3, 1, 10, 4, 8, 0, 6, 2, 9, 5, 7])
    transformed = fg.DEAData(
        inputs=data.inputs[permutation] * [100.0, 0.01],
        outputs=data.outputs[permutation] * 7.0,
        dmu_names=tuple(data.dmu_names[index] for index in permutation),
        input_names=data.input_names,
        output_names=data.output_names,
    )

    first = fg.FreeDisposalHullDEA().fit(data)
    second = fg.FreeDisposalHullDEA().fit(transformed)
    restored = np.empty(data.n_dmus)
    restored[permutation] = second.technical_efficiency

    assert restored == pytest.approx(first.technical_efficiency)


def test_fdh_zero_components_and_leave_one_out_are_explicit() -> None:
    data = fg.DEAData(
        inputs=[[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]],
        outputs=[[1.0], [1.0], [1.0]],
        dmu_names=("A", "B", "C"),
    )

    ordinary = fg.FreeDisposalHullDEA().fit(data)
    super_result = fg.FreeDisposalHullDEA().fit(data, exclude_self=True)

    assert ordinary.technical_efficiency == pytest.approx([1.0, 0.5, 1.0])
    assert super_result.technical_efficiency[0] == pytest.approx(2.0)
    assert super_result.peers("A") == {"B": 1.0}


def test_fdh_validates_configuration_and_reference_contract() -> None:
    data = fg.DEAData([[1.0]], [[1.0]], input_names=("x",))
    mismatched = fg.DEAData([[1.0]], [[1.0]], input_names=("other",))

    with pytest.raises(fg.ModelValidationError, match="input names"):
        fg.FreeDisposalHullDEA().fit(data, reference=mismatched)
    with pytest.raises(fg.ModelValidationError):
        fg.FreeDisposalHullDEA(orientation="graph")
    with pytest.raises(fg.ModelValidationError):
        fg.FreeDisposalHullDEA(tolerance=0.0)
    with pytest.raises(fg.ModelValidationError, match="empty reference"):
        fg.FreeDisposalHullDEA().fit(data, exclude_self=True)
