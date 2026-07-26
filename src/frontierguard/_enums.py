"""Shared model conventions."""

from __future__ import annotations

from enum import Enum
from typing import TypeVar

from frontierguard._errors import ModelValidationError


class Orientation(str, Enum):
    """Radial adjustment direction."""

    INPUT = "input"
    OUTPUT = "output"


class ReturnsToScale(str, Enum):
    """Convex production-technology assumptions."""

    CRS = "crs"
    VRS = "vrs"
    NIRS = "nirs"
    NDRS = "ndrs"


class SlackStrategy(str, Enum):
    """Phase-two slack objective."""

    NONE = "none"
    MAX_SUM = "max_sum"
    NORMALIZED = "normalized"


class SolverStatus(str, Enum):
    """Framework-neutral optimization status."""

    OPTIMAL = "optimal"
    ITERATION_LIMIT = "iteration_limit"
    TIME_LIMIT = "time_limit"
    SOLUTION_LIMIT = "solution_limit"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ERROR = "error"


EnumT = TypeVar("EnumT", bound=Enum)


def coerce_enum(value: EnumT | str, enum_type: type[EnumT], *, label: str) -> EnumT:
    """Coerce a public string into a documented enum."""
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(repr(member.value) for member in enum_type)
        raise ModelValidationError(
            f"{label} must be one of {choices}; received {value!r}."
        ) from exc
