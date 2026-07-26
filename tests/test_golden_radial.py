from __future__ import annotations

import pytest

import frontierguard as fg


@pytest.fixture
def published_eleven_dmu_data() -> fg.DEAData:
    return fg.DEAData(
        inputs=[
            [5, 13],
            [16, 12],
            [16, 26],
            [17, 15],
            [18, 14],
            [23, 6],
            [25, 10],
            [27, 22],
            [37, 14],
            [42, 25],
            [5, 17],
        ],
        outputs=[[12], [14], [25], [26], [8], [9], [27], [30], [31], [26], [12]],
    )


def test_published_eleven_dmu_ccr_bcc_scores(
    published_eleven_dmu_data: fg.DEAData,
) -> None:
    crs_input = fg.RadialDEA(
        orientation="input",
        rts="crs",
        slacks="none",
    ).fit(published_eleven_dmu_data)
    vrs_input = fg.RadialDEA(
        orientation="input",
        rts="vrs",
        slacks="none",
    ).fit(published_eleven_dmu_data)
    crs_output = fg.RadialDEA(
        orientation="output",
        rts="crs",
        slacks="none",
    ).fit(published_eleven_dmu_data)

    assert crs_input.technical_efficiency == pytest.approx(
        [
            1.0,
            0.6222896791,
            0.8198562444,
            1.0,
            0.3103709311,
            0.5555555556,
            1.0,
            0.7576690896,
            0.8201058201,
            0.4905660377,
            1.0,
        ],
        rel=1e-8,
        abs=1e-9,
    )
    assert vrs_input.technical_efficiency == pytest.approx(
        [
            1.0,
            0.8699861687,
            1.0,
            1.0,
            0.7116402116,
            1.0,
            1.0,
            1.0,
            1.0,
            0.4931209269,
            1.0,
        ],
        rel=1e-8,
        abs=1e-9,
    )
    assert crs_output.radial_factor == pytest.approx(
        [
            1.0,
            1.606968641,
            1.219726027,
            1.0,
            3.221951220,
            1.8,
            1.0,
            1.319837398,
            1.219354839,
            2.038461538,
            1.0,
        ],
        rel=1e-8,
        abs=1e-9,
    )
