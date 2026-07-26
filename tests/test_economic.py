from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import pytest

from frontierguard._data import DEAData
from frontierguard._enums import SolverStatus
from frontierguard._errors import (
    DataValidationError,
    ModelValidationError,
    SolverError,
)
from frontierguard._solver import LinearProgram, LinearSolution
from frontierguard.economic import (
    CostEfficiencyDEA,
    CostEfficiencyResult,
    RevenueEfficiencyDEA,
    RevenueEfficiencyResult,
)


@pytest.fixture
def published_cost_data() -> DEAData:
    """Cooper et al.-style three-DMU cost-efficiency oracle."""
    return DEAData(
        inputs=[[3.0, 2.0], [1.0, 3.0], [4.0, 6.0]],
        outputs=[[3.0], [5.0], [6.0]],
        dmu_names=("A", "B", "C"),
        input_names=("labor", "capital"),
        output_names=("service",),
    )


@pytest.fixture
def revenue_data() -> DEAData:
    """A price-specialization oracle with an analytical revenue frontier."""
    return DEAData(
        inputs=[[1.0], [1.0], [1.0]],
        outputs=[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
        dmu_names=("A", "B", "C"),
        input_names=("resource",),
        output_names=("type_1", "type_2"),
    )


def test_published_cost_oracle_and_decomposition(
    published_cost_data: DEAData,
) -> None:
    result = CostEfficiencyDEA([4.0, 2.0], rts="crs").fit(published_cost_data)

    assert isinstance(result, CostEfficiencyResult)
    assert result.observed_cost == pytest.approx([16.0, 10.0, 28.0])
    assert result.minimum_cost == pytest.approx([6.0, 10.0, 12.0])
    assert result.cost_efficiency == pytest.approx([0.375, 1.0, 3 / 7])
    assert result.technical_efficiency == pytest.approx([0.9, 1.0, 0.6])
    assert result.allocative_efficiency == pytest.approx([5 / 12, 1.0, 5 / 7])
    assert result.cost_efficiency == pytest.approx(
        result.technical_efficiency * result.allocative_efficiency
    )
    assert result.economic_efficiency is result.cost_efficiency
    assert result.score_convention == "minimum_cost / observed_cost"
    assert result.peers("A") == {"B": pytest.approx(0.6)}


def test_analytical_revenue_oracle_and_decomposition(
    revenue_data: DEAData,
) -> None:
    result = RevenueEfficiencyDEA([2.0, 1.0], rts="crs").fit(revenue_data)

    assert isinstance(result, RevenueEfficiencyResult)
    assert result.observed_revenue == pytest.approx([2.0, 2.0, 3.0])
    assert result.maximum_revenue == pytest.approx([3.0, 3.0, 3.0])
    assert result.revenue_efficiency == pytest.approx([2 / 3, 2 / 3, 1.0])
    assert result.technical_efficiency == pytest.approx([1.0, 1.0, 1.0])
    assert result.allocative_efficiency == pytest.approx([2 / 3, 2 / 3, 1.0])
    assert result.revenue_efficiency == pytest.approx(
        result.technical_efficiency * result.allocative_efficiency
    )
    assert result.economic_efficiency is result.revenue_efficiency
    assert result.score_convention == "observed_revenue / maximum_revenue"
    assert result.peers("A") == {"C": pytest.approx(1.0)}


def test_factor_price_vector_broadcast_and_per_dmu_prices(
    published_cost_data: DEAData,
    revenue_data: DEAData,
) -> None:
    common = CostEfficiencyDEA([4.0, 2.0], rts="crs").fit(published_cost_data)
    varying_cost = CostEfficiencyDEA(
        [[4.0, 2.0], [8.0, 1.0], [1.0, 3.0]],
        rts="crs",
    ).fit(published_cost_data)
    varying_revenue = RevenueEfficiencyDEA(
        [[2.0, 1.0], [1.0, 2.0], [3.0, 4.0]],
        rts="crs",
    ).fit(revenue_data)

    assert common.price_scope == "common"
    assert common.input_prices == pytest.approx(
        np.asarray([[4.0, 2.0], [4.0, 2.0], [4.0, 2.0]])
    )
    assert varying_cost.price_scope == "per_dmu"
    assert varying_cost.observed_cost == pytest.approx([16.0, 11.0, 22.0])
    assert varying_revenue.price_scope == "per_dmu"
    assert varying_revenue.observed_revenue == pytest.approx([2.0, 4.0, 7.0])


@pytest.mark.parametrize("rts", ["crs", "vrs", "nirs", "ndrs"])
def test_all_returns_to_scale_contracts(
    rts: str,
    published_cost_data: DEAData,
    revenue_data: DEAData,
) -> None:
    cost = CostEfficiencyDEA([4.0, 2.0], rts=rts).fit(published_cost_data)
    revenue = RevenueEfficiencyDEA([2.0, 1.0], rts=rts).fit(revenue_data)

    for result in (cost, revenue):
        sums = result.lambdas.sum(axis=1)
        assert result.returns_to_scale.value == rts
        if rts == "vrs":
            assert sums == pytest.approx(np.ones(len(sums)))
        elif rts == "nirs":
            assert np.all(sums <= 1.0 + 1e-8)
        elif rts == "ndrs":
            assert np.all(sums >= 1.0 - 1e-8)
        assert result.statuses == (SolverStatus.OPTIMAL,) * 3
        assert result.technical_statuses == (SolverStatus.OPTIMAL,) * 3
        assert np.max(result.audit_residual) <= result.audit_tolerance


def test_cost_targets_are_price_optimal_and_output_feasible(
    published_cost_data: DEAData,
) -> None:
    prices = np.asarray([4.0, 2.0])
    result = CostEfficiencyDEA(prices, rts="crs").fit(published_cost_data)

    assert result.input_targets == pytest.approx(
        result.lambdas @ published_cost_data.inputs
    )
    assert result.output_targets == pytest.approx(
        result.lambdas @ published_cost_data.outputs
    )
    assert np.all(result.output_targets >= published_cost_data.outputs - 1e-8)
    assert result.minimum_cost == pytest.approx(result.input_targets @ prices)
    assert result.targets("A") == {
        "inputs": {
            "labor": pytest.approx(0.6),
            "capital": pytest.approx(1.8),
        },
        "outputs": {"service": pytest.approx(3.0)},
    }


def test_revenue_targets_are_resource_feasible_and_price_optimal(
    revenue_data: DEAData,
) -> None:
    prices = np.asarray([2.0, 1.0])
    result = RevenueEfficiencyDEA(prices, rts="crs").fit(revenue_data)

    assert result.input_targets == pytest.approx(result.lambdas @ revenue_data.inputs)
    assert result.output_targets == pytest.approx(result.lambdas @ revenue_data.outputs)
    assert np.all(result.input_targets <= revenue_data.inputs + 1e-8)
    assert result.maximum_revenue == pytest.approx(result.output_targets @ prices)
    assert result.targets("A") == {
        "inputs": {"resource": pytest.approx(1.0)},
        "outputs": {
            "type_1": pytest.approx(1.0),
            "type_2": pytest.approx(1.0),
        },
    }


def test_separate_reference_technology_is_recorded() -> None:
    evaluated = DEAData(
        inputs=[[3.0, 2.0]],
        outputs=[[3.0]],
        dmu_names=("evaluated",),
        input_names=("labor", "capital"),
        output_names=("service",),
    )
    cost_reference = DEAData(
        inputs=[[1.0, 3.0]],
        outputs=[[5.0]],
        dmu_names=("benchmark",),
        input_names=("labor", "capital"),
        output_names=("service",),
    )
    cost = CostEfficiencyDEA([4.0, 2.0], rts="crs").fit(
        evaluated,
        reference=cost_reference,
    )

    revenue_evaluated = DEAData(
        inputs=[[2.0]],
        outputs=[[1.0]],
        dmu_names=("evaluated",),
        input_names=("resource",),
        output_names=("service",),
    )
    revenue_reference = DEAData(
        inputs=[[1.0]],
        outputs=[[2.0]],
        dmu_names=("benchmark",),
        input_names=("resource",),
        output_names=("service",),
    )
    revenue = RevenueEfficiencyDEA(3.0, rts="crs").fit(
        revenue_evaluated,
        reference=revenue_reference,
    )

    assert cost.reference_names == ("benchmark",)
    assert cost.minimum_cost == pytest.approx([6.0])
    assert cost.reference_fingerprint == cost_reference.fingerprint
    assert revenue.reference_names == ("benchmark",)
    assert revenue.maximum_revenue == pytest.approx([12.0])
    assert revenue.revenue_efficiency == pytest.approx([0.25])
    assert revenue.reference_fingerprint == revenue_reference.fingerprint


def test_cost_currency_and_physical_unit_invariance(
    published_cost_data: DEAData,
) -> None:
    baseline = CostEfficiencyDEA([4.0, 2.0], rts="crs").fit(published_cost_data)
    currency = CostEfficiencyDEA([400.0, 200.0], rts="crs").fit(published_cost_data)
    rescaled_data = DEAData(
        inputs=np.asarray(published_cost_data.inputs) * [10.0, 1.0],
        outputs=published_cost_data.outputs,
        dmu_names=published_cost_data.dmu_names,
        input_names=published_cost_data.input_names,
        output_names=published_cost_data.output_names,
    )
    units = CostEfficiencyDEA([0.4, 2.0], rts="crs").fit(rescaled_data)

    assert currency.cost_efficiency == pytest.approx(baseline.cost_efficiency)
    assert currency.technical_efficiency == pytest.approx(baseline.technical_efficiency)
    assert currency.allocative_efficiency == pytest.approx(
        baseline.allocative_efficiency
    )
    assert units.cost_efficiency == pytest.approx(baseline.cost_efficiency)
    assert units.technical_efficiency == pytest.approx(baseline.technical_efficiency)
    assert units.lambdas == pytest.approx(baseline.lambdas)
    assert units.input_targets[:, 0] == pytest.approx(
        10.0 * baseline.input_targets[:, 0]
    )
    assert units.input_targets[:, 1] == pytest.approx(baseline.input_targets[:, 1])


def test_revenue_currency_and_physical_unit_invariance(
    revenue_data: DEAData,
) -> None:
    baseline = RevenueEfficiencyDEA([2.0, 1.0], rts="crs").fit(revenue_data)
    currency = RevenueEfficiencyDEA([200.0, 100.0], rts="crs").fit(revenue_data)
    rescaled_data = DEAData(
        inputs=revenue_data.inputs,
        outputs=np.asarray(revenue_data.outputs) * [10.0, 1.0],
        dmu_names=revenue_data.dmu_names,
        input_names=revenue_data.input_names,
        output_names=revenue_data.output_names,
    )
    units = RevenueEfficiencyDEA([0.2, 1.0], rts="crs").fit(rescaled_data)

    assert currency.revenue_efficiency == pytest.approx(baseline.revenue_efficiency)
    assert currency.technical_efficiency == pytest.approx(baseline.technical_efficiency)
    assert currency.allocative_efficiency == pytest.approx(
        baseline.allocative_efficiency
    )
    assert units.revenue_efficiency == pytest.approx(baseline.revenue_efficiency)
    assert units.technical_efficiency == pytest.approx(baseline.technical_efficiency)
    assert units.lambdas == pytest.approx(baseline.lambdas)
    assert units.output_targets[:, 0] == pytest.approx(
        10.0 * baseline.output_targets[:, 0]
    )
    assert units.output_targets[:, 1] == pytest.approx(baseline.output_targets[:, 1])


def test_results_are_immutable_tidy_and_strict_json(
    published_cost_data: DEAData,
    revenue_data: DEAData,
) -> None:
    results = (
        CostEfficiencyDEA([4.0, 2.0], rts="crs").fit(published_cost_data),
        RevenueEfficiencyDEA([2.0, 1.0], rts="crs").fit(revenue_data),
    )

    for result in results:
        assert result.to_frame().index.tolist() == ["A", "B", "C"]
        exported = result.to_dict()
        json.dumps(exported, allow_nan=False)
        assert exported["data_fingerprint"] == result.data_fingerprint
        assert exported["price_fingerprint"] == result.price_fingerprint
        with pytest.raises(ValueError, match="read-only"):
            result.lambdas[0, 0] = 99.0
        with pytest.raises(DataValidationError, match="Unknown DMU"):
            result.targets("missing")


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: CostEfficiencyDEA([], rts="crs"), "exactly 2"),
        (lambda: CostEfficiencyDEA([4.0, -1.0], rts="crs"), "non-negative"),
        (lambda: CostEfficiencyDEA([4.0, np.nan], rts="crs"), "finite"),
        (lambda: CostEfficiencyDEA([0.0, 0.0], rts="crs"), "must be positive"),
        (
            lambda: CostEfficiencyDEA(
                [[4.0, 2.0], [4.0, 2.0]],
                rts="crs",
            ),
            "shape",
        ),
    ],
)
def test_invalid_cost_prices_fail(
    factory: Callable[[], CostEfficiencyDEA],
    message: str,
    published_cost_data: DEAData,
) -> None:
    with pytest.raises(ModelValidationError, match=message):
        factory().fit(published_cost_data)


@pytest.mark.parametrize(
    ("prices", "message"),
    [
        ([2.0], "exactly 2"),
        ([2.0, -1.0], "non-negative"),
        ([2.0, np.inf], "finite"),
        ([0.0, 0.0], "must be positive"),
        ([[2.0, 1.0]], "shape"),
    ],
)
def test_invalid_revenue_prices_fail(
    prices: object,
    message: str,
    revenue_data: DEAData,
) -> None:
    with pytest.raises(ModelValidationError, match=message):
        RevenueEfficiencyDEA(prices, rts="crs").fit(revenue_data)


def test_zero_observed_revenue_is_rejected_even_with_nonzero_prices(
    revenue_data: DEAData,
) -> None:
    prices = [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]]

    with pytest.raises(ModelValidationError, match="A, B"):
        RevenueEfficiencyDEA(prices, rts="crs").fit(revenue_data)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CostEfficiencyDEA([1.0], rts="magic"),
        lambda: RevenueEfficiencyDEA([1.0], rts="magic"),
        lambda: CostEfficiencyDEA([1.0], tolerance=0.0),
        lambda: RevenueEfficiencyDEA([1.0], tolerance=np.inf),
    ],
)
def test_invalid_model_configuration_fails(factory: Callable[[], object]) -> None:
    with pytest.raises(ModelValidationError):
        factory()


def test_reference_schema_must_match(published_cost_data: DEAData) -> None:
    reference = DEAData(
        inputs=[[1.0, 1.0]],
        outputs=[[1.0]],
        input_names=("different", "capital"),
        output_names=("service",),
    )

    with pytest.raises(ModelValidationError, match="input names"):
        CostEfficiencyDEA([4.0, 2.0]).fit(
            published_cost_data,
            reference=reference,
        )


class _InfeasibleSolver:
    name = "infeasible-test-solver"

    def solve(self, program: LinearProgram) -> LinearSolution:
        variables = np.full(np.asarray(program.objective).shape, np.nan)
        variables.setflags(write=False)
        return LinearSolution(
            status=SolverStatus.INFEASIBLE,
            variables=variables,
            objective=float("nan"),
            message="test infeasibility",
            raw_status=2,
        )


def test_nonoptimal_solver_status_is_reported_as_cost_failure(
    published_cost_data: DEAData,
) -> None:
    with pytest.raises(SolverError, match=r"infeasible.*test infeasibility"):
        CostEfficiencyDEA(
            [4.0, 2.0],
            rts="crs",
            solver=_InfeasibleSolver(),
        ).fit(published_cost_data)


def test_nonoptimal_solver_status_is_reported_as_revenue_failure(
    revenue_data: DEAData,
) -> None:
    with pytest.raises(SolverError, match=r"infeasible.*test infeasibility"):
        RevenueEfficiencyDEA(
            [2.0, 1.0],
            rts="crs",
            solver=_InfeasibleSolver(),
        ).fit(revenue_data)
