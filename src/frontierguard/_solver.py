"""Optimization backend contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.optimize import linprog

from frontierguard._enums import SolverStatus


@dataclass(frozen=True, slots=True)
class LinearProgram:
    """A minimization-form linear program."""

    objective: ArrayLike
    inequalities: ArrayLike | None = None
    inequality_bounds: ArrayLike | None = None
    equalities: ArrayLike | None = None
    equality_bounds: ArrayLike | None = None
    variable_bounds: Sequence[tuple[float | None, float | None]] | None = None


@dataclass(frozen=True, slots=True)
class LinearSolution:
    """Framework-neutral linear-program solution."""

    status: SolverStatus
    variables: NDArray[np.float64]
    objective: float
    message: str
    raw_status: int
    reduced_costs: NDArray[np.float64] | None = None
    inequality_duals: NDArray[np.float64] | None = None
    equality_duals: NDArray[np.float64] | None = None


class LinearSolver(Protocol):
    """Protocol implemented by linear optimization backends."""

    name: str

    def solve(self, program: LinearProgram) -> LinearSolution:
        """Solve one linear program."""


class ScipyHighsSolver:
    """SciPy adapter for the open-source HiGHS linear optimizer."""

    name = "scipy-highs"

    def __init__(self, *, presolve: bool = True) -> None:
        self._presolve = presolve

    def solve(self, program: LinearProgram) -> LinearSolution:
        objective = np.asarray(program.objective, dtype=float)
        inequalities = program.inequalities
        if inequalities is not None:
            inequalities = (
                sparse.csr_matrix(inequalities, dtype=float)
                if sparse.issparse(inequalities)
                else np.asarray(inequalities, dtype=float)
            )
        equalities = program.equalities
        if equalities is not None:
            equalities = (
                sparse.csr_matrix(equalities, dtype=float)
                if sparse.issparse(equalities)
                else np.asarray(equalities, dtype=float)
            )
        result = linprog(
            objective,
            A_ub=inequalities,
            b_ub=(
                None
                if program.inequality_bounds is None
                else np.asarray(program.inequality_bounds, dtype=float)
            ),
            A_eq=equalities,
            b_eq=(
                None
                if program.equality_bounds is None
                else np.asarray(program.equality_bounds, dtype=float)
            ),
            bounds=program.variable_bounds,
            method="highs",
            options={"presolve": self._presolve},
        )
        status = {
            0: SolverStatus.OPTIMAL,
            1: SolverStatus.ITERATION_LIMIT,
            2: SolverStatus.INFEASIBLE,
            3: SolverStatus.UNBOUNDED,
        }.get(int(result.status), SolverStatus.ERROR)
        variables = (
            np.full(objective.shape, np.nan, dtype=float)
            if result.x is None
            else np.asarray(result.x, dtype=float)
        )
        variables.setflags(write=False)
        return LinearSolution(
            status=status,
            variables=variables,
            objective=float(result.fun) if result.fun is not None else float("nan"),
            message=str(result.message),
            raw_status=int(result.status),
        )


class HighsSolver:
    """Direct adapter for the optional :mod:`highspy` package.

    The adapter is intentionally stateless: each call constructs an independent
    HiGHS instance, applies deterministic defaults, and passes one complete LP.
    Install ``frontierguard-dea[highs]`` to use it.

    Parameters
    ----------
    presolve:
        Enable HiGHS presolve.
    primal_feasibility_tolerance, dual_feasibility_tolerance:
        Feasibility tolerances passed directly to HiGHS.
    threads:
        Solver threads. The default of one avoids run-to-run scheduling
        variation.
    random_seed:
        HiGHS random seed.
    parallel:
        Enable parallel solver strategies. Disabled by default for
        reproducibility.
    time_limit:
        Optional wall-clock limit in seconds.
    iteration_limit:
        Optional limit applied to both simplex and interior-point iterations.
    options:
        Additional HiGHS options. These are applied last and therefore may
        deliberately override the reproducible defaults.
    """

    name = "highs"

    def __init__(
        self,
        *,
        presolve: bool = True,
        primal_feasibility_tolerance: float = 1e-7,
        dual_feasibility_tolerance: float = 1e-7,
        threads: int = 1,
        random_seed: int = 0,
        parallel: bool = False,
        time_limit: float | None = None,
        iteration_limit: int | None = None,
        options: dict[str, bool | int | float | str] | None = None,
    ) -> None:
        if (
            not np.isfinite(primal_feasibility_tolerance)
            or primal_feasibility_tolerance <= 0
        ):
            raise ValueError("primal_feasibility_tolerance must be positive.")
        if (
            not np.isfinite(dual_feasibility_tolerance)
            or dual_feasibility_tolerance <= 0
        ):
            raise ValueError("dual_feasibility_tolerance must be positive.")
        if threads < 1:
            raise ValueError("threads must be at least one.")
        if time_limit is not None and (not np.isfinite(time_limit) or time_limit <= 0):
            raise ValueError("time_limit must be positive and finite.")
        if iteration_limit is not None and iteration_limit < 0:
            raise ValueError("iteration_limit cannot be negative.")

        self._presolve = presolve
        self._primal_feasibility_tolerance = primal_feasibility_tolerance
        self._dual_feasibility_tolerance = dual_feasibility_tolerance
        self._threads = threads
        self._random_seed = random_seed
        self._parallel = parallel
        self._time_limit = time_limit
        self._iteration_limit = iteration_limit
        self._options = {} if options is None else dict(options)

    def solve(self, program: LinearProgram) -> LinearSolution:
        """Solve one LP through the native HiGHS Python bindings."""
        try:
            import highspy
        except ImportError as exc:  # pragma: no cover - exercised without the extra
            raise ImportError(
                "HighsSolver requires the optional 'highs' extra; install "
                "'frontierguard-dea[highs]'."
            ) from exc

        (
            objective,
            matrix,
            row_lower,
            row_upper,
            variable_lower,
            variable_upper,
            n_inequalities,
        ) = _normalize_program(program, infinity=float(highspy.kHighsInf))

        model = highspy.HighsLp()
        model.num_col_ = objective.size
        model.num_row_ = matrix.shape[0]
        model.col_cost_ = objective
        model.col_lower_ = variable_lower
        model.col_upper_ = variable_upper
        model.row_lower_ = row_lower
        model.row_upper_ = row_upper
        model.sense_ = highspy.ObjSense.kMinimize
        model.offset_ = 0.0
        model.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        model.a_matrix_.start_ = matrix.indptr
        model.a_matrix_.index_ = matrix.indices
        model.a_matrix_.value_ = matrix.data

        solver = highspy.Highs()  # type: ignore[no-untyped-call]
        self._configure(solver, highspy)
        pass_status = solver.passModel(model)
        if pass_status == highspy.HighsStatus.kError:
            return _failed_solution(
                objective.size,
                message="HiGHS rejected the linear program.",
                raw_status=int(highspy.HighsModelStatus.kModelError),
            )

        run_status = solver.run()
        model_status = solver.getModelStatus()
        status = _highs_status(model_status, highspy)
        message = solver.modelStatusToString(model_status)
        if run_status == highspy.HighsStatus.kError:
            status = SolverStatus.ERROR
            message = f"HiGHS solve error: {message}"

        solution = solver.getSolution()
        has_optimal_solution = status is SolverStatus.OPTIMAL and solution.value_valid
        variables = _solution_vector(
            solution.col_value if has_optimal_solution else None,
            size=objective.size,
        )
        objective_value = (
            float(np.dot(objective, variables))
            if has_optimal_solution
            else float("nan")
        )

        reduced_costs: NDArray[np.float64] | None = None
        inequality_duals: NDArray[np.float64] | None = None
        equality_duals: NDArray[np.float64] | None = None
        if status is SolverStatus.OPTIMAL and solution.dual_valid:
            reduced_costs = _solution_vector(
                solution.col_dual,
                size=objective.size,
            )
            row_duals = _solution_vector(
                solution.row_dual,
                size=matrix.shape[0],
            )
            inequality_duals = _readonly(row_duals[:n_inequalities])
            equality_duals = _readonly(row_duals[n_inequalities:])

        return LinearSolution(
            status=status,
            variables=variables,
            objective=objective_value,
            message=message,
            raw_status=int(model_status),
            reduced_costs=reduced_costs,
            inequality_duals=inequality_duals,
            equality_duals=equality_duals,
        )

    def _configure(self, solver: object, highspy: object) -> None:
        option_values: dict[str, bool | int | float | str] = {
            "output_flag": False,
            "log_to_console": False,
            "presolve": "on" if self._presolve else "off",
            "parallel": "on" if self._parallel else "off",
            "threads": self._threads,
            "random_seed": self._random_seed,
            "primal_feasibility_tolerance": self._primal_feasibility_tolerance,
            "dual_feasibility_tolerance": self._dual_feasibility_tolerance,
        }
        if self._time_limit is not None:
            option_values["time_limit"] = self._time_limit
        if self._iteration_limit is not None:
            option_values["simplex_iteration_limit"] = self._iteration_limit
            option_values["ipm_iteration_limit"] = self._iteration_limit
        option_values.update(self._options)

        for name, value in option_values.items():
            option_status = solver.setOptionValue(name, value)  # type: ignore[attr-defined]
            if option_status == highspy.HighsStatus.kError:  # type: ignore[attr-defined]
                raise ValueError(f"HiGHS rejected option {name!r}={value!r}.")


def _normalize_program(
    program: LinearProgram,
    *,
    infinity: float,
) -> tuple[
    NDArray[np.float64],
    sparse.csr_matrix,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    int,
]:
    objective = np.asarray(program.objective, dtype=float)
    if objective.ndim != 1 or objective.size == 0:
        raise ValueError("objective must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(objective)):
        raise ValueError("objective must contain only finite values.")

    inequalities, inequality_bounds = _constraint_block(
        program.inequalities,
        program.inequality_bounds,
        columns=objective.size,
        label="inequalities",
    )
    equalities, equality_bounds = _constraint_block(
        program.equalities,
        program.equality_bounds,
        columns=objective.size,
        label="equalities",
    )
    matrix = sparse.vstack((inequalities, equalities), format="csr")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()

    n_inequalities = inequalities.shape[0]
    row_lower = np.concatenate(
        (
            np.full(n_inequalities, -infinity, dtype=float),
            equality_bounds,
        )
    )
    row_upper = np.concatenate((inequality_bounds, equality_bounds))
    row_lower = _replace_infinities(row_lower, infinity)
    row_upper = _replace_infinities(row_upper, infinity)
    variable_lower, variable_upper = _variable_bounds(
        program.variable_bounds,
        variables=objective.size,
        infinity=infinity,
    )
    return (
        objective,
        matrix,
        row_lower,
        row_upper,
        variable_lower,
        variable_upper,
        n_inequalities,
    )


def _constraint_block(
    coefficients: ArrayLike | None,
    bounds: ArrayLike | None,
    *,
    columns: int,
    label: str,
) -> tuple[sparse.csr_matrix, NDArray[np.float64]]:
    if coefficients is None:
        if bounds is not None:
            raise ValueError(f"{label} bounds require a coefficient matrix.")
        return sparse.csr_matrix((0, columns), dtype=float), np.empty(0, dtype=float)
    if bounds is None:
        raise ValueError(f"{label} require matching bounds.")

    if sparse.issparse(coefficients):
        matrix = sparse.csr_matrix(coefficients, dtype=float)
    else:
        dense = np.asarray(coefficients, dtype=float)
        if dense.ndim != 2:
            raise ValueError(f"{label} must be a two-dimensional matrix.")
        matrix = sparse.csr_matrix(dense)
    if matrix.ndim != 2 or matrix.shape[1] != columns:
        raise ValueError(f"{label} must have {columns} columns.")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"{label} must contain only finite coefficients.")

    vector = np.asarray(bounds, dtype=float)
    if vector.ndim != 1 or vector.size != matrix.shape[0]:
        raise ValueError(f"{label} bounds must match the matrix row count.")
    if np.any(np.isnan(vector)):
        raise ValueError(f"{label} bounds cannot contain NaN.")
    return matrix, vector


def _variable_bounds(
    bounds: Sequence[tuple[float | None, float | None]] | None,
    *,
    variables: int,
    infinity: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if bounds is None:
        return (
            np.zeros(variables, dtype=float),
            np.full(variables, infinity, dtype=float),
        )
    if len(bounds) != variables:
        raise ValueError("variable_bounds must contain one pair per variable.")

    lower = np.empty(variables, dtype=float)
    upper = np.empty(variables, dtype=float)
    for index, (lower_value, upper_value) in enumerate(bounds):
        lower[index] = -infinity if lower_value is None else float(lower_value)
        upper[index] = infinity if upper_value is None else float(upper_value)
    lower = _replace_infinities(lower, infinity)
    upper = _replace_infinities(upper, infinity)
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ValueError("variable bounds cannot contain NaN.")
    if np.any(lower == infinity) or np.any(upper == -infinity):
        raise ValueError("variable bounds contain an invalid infinite endpoint.")
    if np.any(lower > upper):
        raise ValueError("each variable lower bound must not exceed its upper bound.")
    return lower, upper


def _replace_infinities(
    values: NDArray[np.float64],
    infinity: float,
) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=float).copy()
    result[np.isneginf(result)] = -infinity
    result[np.isposinf(result)] = infinity
    return result


def _highs_status(model_status: object, highspy: object) -> SolverStatus:
    if model_status == highspy.HighsModelStatus.kOptimal:  # type: ignore[attr-defined]
        return SolverStatus.OPTIMAL
    if model_status == highspy.HighsModelStatus.kInfeasible:  # type: ignore[attr-defined]
        return SolverStatus.INFEASIBLE
    if model_status == highspy.HighsModelStatus.kUnbounded:  # type: ignore[attr-defined]
        return SolverStatus.UNBOUNDED
    if model_status == highspy.HighsModelStatus.kIterationLimit:  # type: ignore[attr-defined]
        return SolverStatus.ITERATION_LIMIT
    if model_status == highspy.HighsModelStatus.kTimeLimit:  # type: ignore[attr-defined]
        return SolverStatus.TIME_LIMIT
    if model_status == highspy.HighsModelStatus.kSolutionLimit:  # type: ignore[attr-defined]
        return SolverStatus.SOLUTION_LIMIT
    return SolverStatus.ERROR


def _solution_vector(
    values: Sequence[float] | None,
    *,
    size: int,
) -> NDArray[np.float64]:
    if values is None or len(values) != size:
        return _readonly(np.full(size, np.nan, dtype=float))
    return _readonly(np.asarray(values, dtype=float))


def _readonly(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=float).copy()
    result.setflags(write=False)
    return result


def _failed_solution(
    variables: int,
    *,
    message: str,
    raw_status: int,
) -> LinearSolution:
    return LinearSolution(
        status=SolverStatus.ERROR,
        variables=_readonly(np.full(variables, np.nan, dtype=float)),
        objective=float("nan"),
        message=message,
        raw_status=raw_status,
    )
