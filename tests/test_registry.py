from __future__ import annotations

import json

import pytest

import frontierguard as fg


def test_registry_exposes_explicit_scientific_contracts() -> None:
    radial = fg.get_method("radial.envelopment")

    assert radial.efficient_value == 1.0
    assert radial.orientations == ("input", "output")
    assert radial.returns_to_scale == ("crs", "vrs", "nirs", "ndrs")
    json.dumps(radial.to_dict(), allow_nan=False)


def test_registry_filters_and_unknown_ids() -> None:
    methods = fg.list_methods(
        stability="provisional",
        problem_class="linear_program",
    )

    assert [method.method_id for method in methods] == [
        "additive.weighted",
        "directional.distance",
        "economic.cost",
        "economic.revenue",
        "multiplier.crs.cross",
        "network.serial.relational",
        "productivity.malmquist.adjacent",
        "radial.envelopment",
        "radial.scale_efficiency",
        "radial.super_efficiency",
        "sbm.nonoriented",
    ]
    assert (
        fg.get_method("network.dag.relational").stability
        is fg.MethodStability.EXPERIMENTAL
    )
    with pytest.raises(fg.ModelValidationError, match="Unknown method_id"):
        fg.get_method("imaginary")
    with pytest.raises(fg.ModelValidationError, match="Filter"):
        fg.list_methods(stability="finished")
