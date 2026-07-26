"""Public exception hierarchy."""


class FrontierGuardError(Exception):
    """Base class for expected FrontierGuard failures."""


class DataValidationError(FrontierGuardError, ValueError):
    """Raised when DEA data violate the declared model contract."""


class ModelValidationError(FrontierGuardError, ValueError):
    """Raised when a model configuration is unsupported or inconsistent."""


class SolverError(FrontierGuardError, RuntimeError):
    """Raised when an optimization problem does not solve to optimality."""


class NumericalError(FrontierGuardError, ArithmeticError):
    """Raised when a numerical invariant cannot be verified."""
