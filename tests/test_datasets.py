from __future__ import annotations

import pytest

import frontierguard as fg


def test_packaged_classical_datasets_match_documented_shapes() -> None:
    five = fg.load_five_dmu()
    eleven = fg.load_eleven_dmu()

    assert five.inputs.shape == (5, 2)
    assert five.outputs.shape == (5, 1)
    assert eleven.inputs.shape == (11, 2)
    assert eleven.outputs.shape == (11, 1)
    assert fg.RadialDEA(rts="crs", slacks="none").fit(
        five
    ).technical_efficiency == pytest.approx([0.5, 1.0, 5 / 6, 5 / 7, 1.0])


def test_packaged_network_fixture_exposes_internal_inefficiency() -> None:
    result = fg.RelationalNetworkDEA().fit(fg.load_internal_mismatch_network())

    assert result.overall_efficiency == pytest.approx([0.5, 0.5])
