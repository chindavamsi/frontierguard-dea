from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as strategies

import frontierguard as fg


@given(
    seed=strategies.integers(min_value=0, max_value=2**32 - 1),
    n_dmus=strategies.integers(min_value=4, max_value=10),
    n_inputs=strategies.integers(min_value=1, max_value=3),
    n_outputs=strategies.integers(min_value=1, max_value=3),
)
@settings(max_examples=30, deadline=None)
def test_random_vrs_input_solutions_satisfy_primal_constraints(
    seed: int,
    n_dmus: int,
    n_inputs: int,
    n_outputs: int,
) -> None:
    rng = np.random.default_rng(seed)
    inputs = rng.lognormal(size=(n_dmus, n_inputs))
    outputs = rng.lognormal(size=(n_dmus, n_outputs))
    data = fg.DEAData(inputs, outputs)
    result = fg.RadialDEA(
        orientation="input",
        rts="vrs",
        slacks="none",
    ).fit(data)

    projected_inputs = result.lambdas @ inputs
    projected_outputs = result.lambdas @ outputs
    assert np.all(projected_inputs <= result.radial_factor[:, None] * inputs + 1e-7)
    assert np.all(projected_outputs >= outputs - 1e-7)
    assert result.lambdas.sum(axis=1) == pytest.approx(np.ones(n_dmus), abs=1e-7)
    assert np.all(result.technical_efficiency <= 1.0 + 1e-7)
    assert np.all(result.technical_efficiency > 0)


@given(seed=strategies.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=20, deadline=None)
def test_positive_unit_changes_preserve_radial_scores(seed: int) -> None:
    rng = np.random.default_rng(seed)
    inputs = rng.lognormal(size=(7, 2))
    outputs = rng.lognormal(size=(7, 2))
    input_scale = rng.lognormal(size=2)
    output_scale = rng.lognormal(size=2)

    original = fg.RadialDEA(rts="crs").fit(fg.DEAData(inputs, outputs))
    rescaled = fg.RadialDEA(rts="crs").fit(
        fg.DEAData(inputs * input_scale, outputs * output_scale)
    )

    assert original.technical_efficiency == pytest.approx(
        rescaled.technical_efficiency,
        rel=1e-8,
        abs=1e-10,
    )
