"""Auditable method metadata and compatibility discovery."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TypeVar

from frontierguard._errors import ModelValidationError

EnumT = TypeVar("EnumT", bound=Enum)


class MethodStability(str, Enum):
    """Scientific stability of one mathematical method contract."""

    STABLE = "stable"
    PROVISIONAL = "provisional"
    EXPERIMENTAL = "experimental"


class ProblemClass(str, Enum):
    """Optimization class required by a method."""

    LP = "linear_program"
    ENUMERATION = "finite_enumeration"
    MILP = "mixed_integer_linear_program"
    QP = "quadratic_program"
    CONIC = "conic_program"
    NLP = "nonlinear_program"
    STATISTICAL = "statistical_procedure"


@dataclass(frozen=True, slots=True)
class MethodDescriptor:
    """Machine-readable scientific contract for one implemented method."""

    method_id: str
    equation_version: str
    label: str
    stability: MethodStability
    problem_class: ProblemClass
    data_contract: str
    score_name: str
    efficient_value: float
    orientations: tuple[str, ...]
    returns_to_scale: tuple[str, ...]
    citations: tuple[str, ...]
    result_models: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible descriptor."""
        return {
            "method_id": self.method_id,
            "equation_version": self.equation_version,
            "label": self.label,
            "stability": self.stability.value,
            "problem_class": self.problem_class.value,
            "data_contract": self.data_contract,
            "score_name": self.score_name,
            "efficient_value": self.efficient_value,
            "orientations": list(self.orientations),
            "returns_to_scale": list(self.returns_to_scale),
            "citations": list(self.citations),
            "result_models": list(self.result_models),
            "limitations": list(self.limitations),
        }


_METHODS = MappingProxyType(
    {
        "additive.weighted": MethodDescriptor(
            method_id="additive.weighted",
            equation_version="1.0",
            label="Weighted additive DEA",
            stability=MethodStability.PROVISIONAL,
            problem_class=ProblemClass.LP,
            data_contract="DEAData",
            score_name="additive_inefficiency",
            efficient_value=0.0,
            orientations=("nonradial",),
            returns_to_scale=("crs", "vrs", "nirs", "ndrs"),
            citations=("Charnes et al. (1985)",),
            result_models=("weighted_additive",),
            limitations=(
                "Non-negative data only",
                "Raw weighted scores are unit-dependent",
            ),
        ),
        "directional.distance": MethodDescriptor(
            method_id="directional.distance",
            equation_version="1.0",
            label="Directional distance DEA",
            stability=MethodStability.PROVISIONAL,
            problem_class=ProblemClass.LP,
            data_contract="DEAData",
            score_name="directional_distance",
            efficient_value=0.0,
            orientations=("explicit_direction",),
            returns_to_scale=("crs", "vrs", "nirs", "ndrs"),
            citations=("Chambers, Chung & Färe (1996)",),
            result_models=("directional_distance",),
            limitations=(
                "Non-negative data and directions only",
                "Beta is signed and may be negative against an external technology",
            ),
        ),
        "economic.cost": MethodDescriptor(
            method_id="economic.cost",
            equation_version="1.0",
            label="Cost, technical, and allocative efficiency",
            stability=MethodStability.PROVISIONAL,
            problem_class=ProblemClass.LP,
            data_contract="DEAData plus non-negative input prices",
            score_name="cost_efficiency",
            efficient_value=1.0,
            orientations=("input_economic",),
            returns_to_scale=("crs", "vrs", "nirs", "ndrs"),
            citations=(
                "Farrell (1957)",
                "Färe, Grosskopf & Lovell (1985)",
            ),
            result_models=("cost_efficiency",),
            limitations=(
                "Observed input cost must be positive",
                "Prices are treated as observed, exogenous model inputs",
                "Profit efficiency is a separate model",
            ),
        ),
        "economic.revenue": MethodDescriptor(
            method_id="economic.revenue",
            equation_version="1.0",
            label="Revenue, technical, and allocative efficiency",
            stability=MethodStability.PROVISIONAL,
            problem_class=ProblemClass.LP,
            data_contract="DEAData plus non-negative output prices",
            score_name="revenue_efficiency",
            efficient_value=1.0,
            orientations=("output_economic",),
            returns_to_scale=("crs", "vrs", "nirs", "ndrs"),
            citations=(
                "Farrell (1957)",
                "Färe, Grosskopf & Lovell (1985)",
            ),
            result_models=("revenue_efficiency",),
            limitations=(
                "Observed output revenue must be positive",
                "Prices are treated as observed, exogenous model inputs",
                "Profit efficiency is a separate model",
            ),
        ),
        "fdh.radial": MethodDescriptor(
            method_id="fdh.radial",
            equation_version="1.0",
            label="Radial Free Disposal Hull",
            stability=MethodStability.PROVISIONAL,
            problem_class=ProblemClass.ENUMERATION,
            data_contract="DEAData",
            score_name="technical_efficiency",
            efficient_value=1.0,
            orientations=("input", "output"),
            returns_to_scale=("fdh",),
            citations=("Deprins, Simar & Tulkens (1984)",),
            result_models=("free_disposal_hull",),
            limitations=(
                "Exact peer enumeration",
                "FDH+ and partial-frontier variants are separate methods",
            ),
        ),
        "multiplier.crs.cross": MethodDescriptor(
            method_id="multiplier.crs.cross",
            equation_version="1.0",
            label="CRS multiplier and selected cross-efficiency",
            stability=MethodStability.PROVISIONAL,
            problem_class=ProblemClass.LP,
            data_contract="DEAData",
            score_name="technical_efficiency",
            efficient_value=1.0,
            orientations=("input", "output"),
            returns_to_scale=("crs",),
            citations=(
                "Charnes, Cooper & Rhodes (1978)",
                "Sexton, Silkman & Hogan (1986)",
            ),
            result_models=("crs_multiplier_cross_efficiency",),
            limitations=(
                "Cross scores use one solver-selected optimal multiplier system",
                "Aggressive and benevolent secondary objectives are not included",
                "Assurance-region bounds are assumptions in original data units",
            ),
        ),
        "radial.envelopment": MethodDescriptor(
            method_id="radial.envelopment",
            equation_version="1.0",
            label="Radial envelopment DEA",
            stability=MethodStability.PROVISIONAL,
            problem_class=ProblemClass.LP,
            data_contract="DEAData",
            score_name="technical_efficiency",
            efficient_value=1.0,
            orientations=("input", "output"),
            returns_to_scale=("crs", "vrs", "nirs", "ndrs"),
            citations=(
                "Charnes, Cooper & Rhodes (1978)",
                "Banker, Charnes & Cooper (1984)",
            ),
            result_models=("radial_envelopment",),
            limitations=("Non-negative data only",),
        ),
        "radial.scale_efficiency": MethodDescriptor(
            method_id="radial.scale_efficiency",
            equation_version="1.0",
            label="Scale efficiency and RTS diagnosis",
            stability=MethodStability.PROVISIONAL,
            problem_class=ProblemClass.LP,
            data_contract="DEAData",
            score_name="scale_efficiency",
            efficient_value=1.0,
            orientations=("input", "output"),
            returns_to_scale=("crs", "vrs", "nirs"),
            citations=("Färe, Grosskopf & Lovell (1985)",),
            result_models=("scale_efficiency",),
            limitations=("RTS classification uses the standard NIRS test",),
        ),
        "radial.super_efficiency": MethodDescriptor(
            method_id="radial.super_efficiency",
            equation_version="1.0",
            label="Radial leave-one-out super-efficiency",
            stability=MethodStability.PROVISIONAL,
            problem_class=ProblemClass.LP,
            data_contract="DEAData",
            score_name="super_efficiency",
            efficient_value=1.0,
            orientations=("input", "output"),
            returns_to_scale=("crs", "vrs", "nirs", "ndrs"),
            citations=("Andersen & Petersen (1993)",),
            result_models=("radial_super_efficiency",),
            limitations=(
                "VRS and restricted-RTS models can be infeasible",
                "Output score is reciprocal to the raw expansion factor",
            ),
        ),
        "sbm.nonoriented": MethodDescriptor(
            method_id="sbm.nonoriented",
            equation_version="1.0",
            label="Tone non-oriented slack-based measure",
            stability=MethodStability.PROVISIONAL,
            problem_class=ProblemClass.LP,
            data_contract="DEAData",
            score_name="sbm_efficiency",
            efficient_value=1.0,
            orientations=("nonoriented",),
            returns_to_scale=("crs", "vrs", "nirs", "ndrs"),
            citations=("Tone (2001)",),
            result_models=("slack_based_measure",),
            limitations=(
                "Evaluated inputs and outputs must be strictly positive",
                "Leave-one-out SBM is not super-SBM",
            ),
        ),
        "productivity.malmquist.adjacent": MethodDescriptor(
            method_id="productivity.malmquist.adjacent",
            equation_version="1.0",
            label="Adjacent-period four-distance Malmquist index",
            stability=MethodStability.PROVISIONAL,
            problem_class=ProblemClass.LP,
            data_contract="Two or more exactly aligned DEAData periods",
            score_name="malmquist_productivity_index",
            efficient_value=1.0,
            orientations=("input", "output"),
            returns_to_scale=("crs", "vrs"),
            citations=(
                "Caves, Christensen & Diewert (1982)",
                "Färe et al. (1994)",
            ),
            result_models=(
                "adjacent_malmquist",
                "adjacent_malmquist_panel",
            ),
            limitations=(
                "Adjacent-period technologies only",
                "VRS result is not the standard CRS scale-inclusive index",
                "Global and Malmquist-Luenberger indexes are separate methods",
            ),
        ),
        "network.serial.relational": MethodDescriptor(
            method_id="network.serial.relational",
            equation_version="1.0",
            label="Serial relational network DEA",
            stability=MethodStability.PROVISIONAL,
            problem_class=ProblemClass.LP,
            data_contract="SerialNetworkData",
            score_name="overall_efficiency",
            efficient_value=1.0,
            orientations=("multiplier_ratio",),
            returns_to_scale=("crs",),
            citations=("Kao & Hwang (2008)",),
            result_models=("serial_relational_network",),
            limitations=(
                "Serial topology",
                "External inputs at the first stage and outputs at the last stage",
                "Stage multiplier solutions can be non-unique",
            ),
        ),
        "network.dag.relational": MethodDescriptor(
            method_id="network.dag.relational",
            equation_version="1.0",
            label="Graph-defined relational network DEA",
            stability=MethodStability.EXPERIMENTAL,
            problem_class=ProblemClass.LP,
            data_contract="NetworkData",
            score_name="overall_efficiency",
            efficient_value=1.0,
            orientations=("multiplier_ratio",),
            returns_to_scale=("crs",),
            citations=("Kao (2014)",),
            result_models=("network_dag_relational",),
            limitations=(
                "Acyclic directed topology",
                "CRS relational technology",
                "Stage multiplier solutions can be non-unique",
            ),
        ),
    }
)


def get_method(method_id: str) -> MethodDescriptor:
    """Return one descriptor or fail with the available method IDs."""
    try:
        return _METHODS[method_id]
    except KeyError as exc:
        available = ", ".join(sorted(_METHODS))
        raise ModelValidationError(
            f"Unknown method_id {method_id!r}. Available methods: {available}."
        ) from exc


def list_methods(
    *,
    stability: MethodStability | str | None = None,
    problem_class: ProblemClass | str | None = None,
) -> tuple[MethodDescriptor, ...]:
    """List method descriptors with optional scientific/capability filters."""
    resolved_stability = (
        None if stability is None else _coerce_filter(stability, MethodStability)
    )
    resolved_problem = (
        None if problem_class is None else _coerce_filter(problem_class, ProblemClass)
    )
    methods: Iterable[MethodDescriptor] = _METHODS.values()
    if resolved_stability is not None:
        methods = (
            method for method in methods if method.stability is resolved_stability
        )
    if resolved_problem is not None:
        methods = (
            method for method in methods if method.problem_class is resolved_problem
        )
    return tuple(sorted(methods, key=lambda method: method.method_id))


def _coerce_filter(value: EnumT | str, enum_type: type[EnumT]) -> EnumT:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(repr(member.value) for member in enum_type)
        raise ModelValidationError(
            f"Filter must be one of {choices}; received {value!r}."
        ) from exc
