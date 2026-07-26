"""Conformance tests for FrontierGuard linear-program backends."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from frontierguard._enums import SolverStatus
from frontierguard._solver import (
    HighsSolver,
    LinearProgram,
    ScipyHighsSolver,
    _highs_status,
)


@pytest.fixture
def constrained_program() -> LinearProgram:
    """LP with inequality, equality, finite, and infinite bounds."""
    return LinearProgram(
        objective=[1.0, 2.0],
        inequalities=sparse.csr_matrix([[-1.0, -1.0]]),
        inequality_bounds=[-3.0],
        equalities=[[1.0, -1.0]],
        equality_bounds=[1.0],
        variable_bounds=[(0.0, 4.0), (0.0, np.inf)],
    )


def test_native_highs_matches_scipy(
    constrained_program: LinearProgram,
) -> None:
    native = HighsSolver().solve(constrained_program)
    fallback = ScipyHighsSolver().solve(constrained_program)

    assert native.status is SolverStatus.OPTIMAL
    assert fallback.status is SolverStatus.OPTIMAL
    assert native.variables == pytest.approx([2.0, 1.0])
    assert native.variables == pytest.approx(fallback.variables)
    assert native.objective == pytest.approx(4.0)
    assert native.objective == pytest.approx(fallback.objective)
    assert native.raw_status > 0
    assert "Optimal" in native.message


def test_native_highs_exposes_immutable_duals(
    constrained_program: LinearProgram,
) -> None:
    result = HighsSolver().solve(constrained_program)

    assert result.reduced_costs is not None
    assert result.inequality_duals is not None
    assert result.equality_duals is not None
    assert result.reduced_costs.shape == (2,)
    assert result.inequality_duals.shape == (1,)
    assert result.equality_duals.shape == (1,)
    assert not result.reduced_costs.flags.writeable
    assert not result.inequality_duals.flags.writeable
    assert not result.equality_duals.flags.writeable


@pytest.mark.parametrize(
    ("program", "expected"),
    [
        (
            LinearProgram(
                objective=[1.0],
                inequalities=[[1.0]],
                inequality_bounds=[0.0],
                variable_bounds=[(1.0, None)],
            ),
            SolverStatus.INFEASIBLE,
        ),
        (
            LinearProgram(
                objective=[-1.0],
                variable_bounds=[(0.0, None)],
            ),
            SolverStatus.UNBOUNDED,
        ),
    ],
)
def test_native_highs_normalizes_terminal_statuses(
    program: LinearProgram,
    expected: SolverStatus,
) -> None:
    result = HighsSolver().solve(program)

    assert result.status is expected
    assert np.isnan(result.objective)
    assert np.all(np.isnan(result.variables))


def test_native_highs_distinguishes_limit_statuses() -> None:
    import highspy

    assert (
        _highs_status(highspy.HighsModelStatus.kIterationLimit, highspy)
        is SolverStatus.ITERATION_LIMIT
    )
    assert (
        _highs_status(highspy.HighsModelStatus.kTimeLimit, highspy)
        is SolverStatus.TIME_LIMIT
    )
    assert (
        _highs_status(highspy.HighsModelStatus.kSolutionLimit, highspy)
        is SolverStatus.SOLUTION_LIMIT
    )


def test_native_highs_supports_free_variables_and_explicit_infinities() -> None:
    program = LinearProgram(
        objective=[1.0],
        inequalities=[[-1.0]],
        inequality_bounds=[2.0],
        variable_bounds=[(-np.inf, np.inf)],
    )

    result = HighsSolver().solve(program)

    assert result.status is SolverStatus.OPTIMAL
    assert result.variables == pytest.approx([-2.0])
    assert result.objective == pytest.approx(-2.0)


def test_native_highs_applies_limits_and_reproducibility_options(
    constrained_program: LinearProgram,
) -> None:
    limited = HighsSolver(
        presolve=False,
        parallel=False,
        time_limit=10.0,
        iteration_limit=0,
        random_seed=17,
    ).solve(constrained_program)

    assert limited.status is SolverStatus.ITERATION_LIMIT
    assert np.all(np.isnan(limited.variables))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"primal_feasibility_tolerance": 0.0},
        {"dual_feasibility_tolerance": np.nan},
        {"threads": 0},
        {"time_limit": np.inf},
        {"iteration_limit": -1},
    ],
)
def test_native_highs_rejects_invalid_configuration(
    kwargs: dict[str, float | int],
) -> None:
    with pytest.raises(ValueError, match=r"must|cannot"):
        HighsSolver(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "program",
    [
        LinearProgram(objective=[]),
        LinearProgram(objective=[np.nan]),
        LinearProgram(objective=[1.0], inequality_bounds=[1.0]),
        LinearProgram(objective=[1.0], inequalities=[[1.0]]),
        LinearProgram(
            objective=[1.0],
            inequalities=[[1.0, 2.0]],
            inequality_bounds=[1.0],
        ),
        LinearProgram(
            objective=[1.0],
            variable_bounds=[(0.0, 1.0), (0.0, 1.0)],
        ),
        LinearProgram(
            objective=[1.0],
            variable_bounds=[(2.0, 1.0)],
        ),
        LinearProgram(
            objective=[1.0],
            inequalities=[1.0],
            inequality_bounds=[1.0],
        ),
        LinearProgram(
            objective=[1.0],
            inequalities=[[np.nan]],
            inequality_bounds=[1.0],
        ),
        LinearProgram(
            objective=[1.0],
            inequalities=[[1.0]],
            inequality_bounds=[[1.0]],
        ),
        LinearProgram(
            objective=[1.0],
            inequalities=[[1.0]],
            inequality_bounds=[np.nan],
        ),
        LinearProgram(
            objective=[1.0],
            variable_bounds=[(np.nan, 1.0)],
        ),
        LinearProgram(
            objective=[1.0],
            variable_bounds=[(np.inf, None)],
        ),
        LinearProgram(
            objective=[1.0],
            variable_bounds=[(None, -np.inf)],
        ),
    ],
)
def test_native_highs_validates_program_shape(program: LinearProgram) -> None:
    with pytest.raises(
        ValueError,
        match=r"must|require|cannot|bounds|objective|variable|finite|NaN|invalid",
    ):
        HighsSolver().solve(program)


def test_native_highs_rejects_unknown_option() -> None:
    solver = HighsSolver(options={"not_a_highs_option": 1})

    with pytest.raises(ValueError, match="not_a_highs_option"):
        solver.solve(LinearProgram(objective=[1.0]))
