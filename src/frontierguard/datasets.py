"""Small canonical datasets for tutorials and mathematical regression tests."""

from __future__ import annotations

from frontierguard._data import DEAData
from frontierguard.graph_network import NetworkData, NetworkLink, NetworkStage


def load_five_dmu() -> DEAData:
    """Return the five-DMU radial DEA fixture used by multiple implementations."""
    return DEAData(
        inputs=[[2, 5], [2, 4], [6, 6], [3, 2], [6, 2]],
        outputs=[[1], [2], [3], [1], [2]],
        dmu_names=("A", "B", "C", "D", "E"),
        input_names=("x1", "x2"),
        output_names=("y1",),
    )


def load_eleven_dmu() -> DEAData:
    """Return a published eleven-DMU CCR/BCC/FDH comparison fixture."""
    return DEAData(
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
        dmu_names=tuple(f"DMU{position}" for position in range(1, 12)),
        input_names=("x1", "x2"),
        output_names=("y1",),
    )


def load_internal_mismatch_network() -> NetworkData:
    """Return a two-stage fixture where black-box CCR hides internal inefficiency."""
    return NetworkData(
        stages=(
            NetworkStage(
                "upstream",
                inputs=[[1.0], [2.0]],
                input_names=("resource",),
            ),
            NetworkStage(
                "downstream",
                outputs=[[1.0], [2.0]],
                output_names=("outcome",),
            ),
        ),
        links=(
            NetworkLink(
                "intermediate",
                "upstream",
                "downstream",
                [[1.0], [4.0]],
                factor_names=("processed",),
            ),
        ),
        dmu_names=("A", "B"),
    )
