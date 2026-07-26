"""FrontierGuard DEA public API."""

from frontierguard._data import DEAData, SerialNetworkData
from frontierguard._enums import (
    Orientation,
    ReturnsToScale,
    SlackStrategy,
    SolverStatus,
)
from frontierguard._errors import (
    DataValidationError,
    FrontierGuardError,
    ModelValidationError,
    NumericalError,
    SolverError,
)
from frontierguard._results import RadialResult, SerialNetworkResult
from frontierguard._solver import (
    HighsSolver,
    LinearProgram,
    LinearSolution,
    ScipyHighsSolver,
)
from frontierguard._version import __version__
from frontierguard.additive import AdditiveResult, WeightedAdditiveDEA
from frontierguard.datasets import (
    load_eleven_dmu,
    load_five_dmu,
    load_internal_mismatch_network,
)
from frontierguard.directional import DirectionalDEA, DirectionalResult
from frontierguard.economic import (
    CostEfficiencyDEA,
    CostEfficiencyResult,
    RevenueEfficiencyDEA,
    RevenueEfficiencyResult,
)
from frontierguard.fdh import FDHResult, FreeDisposalHullDEA
from frontierguard.graph_network import (
    NetworkData,
    NetworkLink,
    NetworkResult,
    NetworkStage,
    RelationalNetworkDEA,
)
from frontierguard.multiplier import (
    AssuranceRegion,
    MultiplierDEA,
    MultiplierResult,
    WeightKind,
    multiplier_dea,
)
from frontierguard.network import SerialNetworkDEA
from frontierguard.productivity import (
    AdjacentMalmquistResult,
    MalmquistDEA,
    MalmquistPanelResult,
    malmquist,
    malmquist_panel,
)
from frontierguard.radial import RadialDEA, dea
from frontierguard.registry import (
    MethodDescriptor,
    MethodStability,
    ProblemClass,
    get_method,
    list_methods,
)
from frontierguard.reporting import write_evidence_bundle
from frontierguard.sbm import SBMDEA, SBMResult, sbm
from frontierguard.scale import (
    ScaleEfficiencyDEA,
    ScaleEfficiencyResult,
    ScaleRegime,
)
from frontierguard.super_efficiency import (
    RadialSuperEfficiencyDEA,
    SuperEfficiencyResult,
)

__all__ = [
    "SBMDEA",
    "AdditiveResult",
    "AdjacentMalmquistResult",
    "AssuranceRegion",
    "CostEfficiencyDEA",
    "CostEfficiencyResult",
    "DEAData",
    "DataValidationError",
    "DirectionalDEA",
    "DirectionalResult",
    "FDHResult",
    "FreeDisposalHullDEA",
    "FrontierGuardError",
    "HighsSolver",
    "LinearProgram",
    "LinearSolution",
    "MalmquistDEA",
    "MalmquistPanelResult",
    "MethodDescriptor",
    "MethodStability",
    "ModelValidationError",
    "MultiplierDEA",
    "MultiplierResult",
    "NetworkData",
    "NetworkLink",
    "NetworkResult",
    "NetworkStage",
    "NumericalError",
    "Orientation",
    "ProblemClass",
    "RadialDEA",
    "RadialResult",
    "RadialSuperEfficiencyDEA",
    "RelationalNetworkDEA",
    "ReturnsToScale",
    "RevenueEfficiencyDEA",
    "RevenueEfficiencyResult",
    "SBMResult",
    "ScaleEfficiencyDEA",
    "ScaleEfficiencyResult",
    "ScaleRegime",
    "ScipyHighsSolver",
    "SerialNetworkDEA",
    "SerialNetworkData",
    "SerialNetworkResult",
    "SlackStrategy",
    "SolverError",
    "SolverStatus",
    "SuperEfficiencyResult",
    "WeightKind",
    "WeightedAdditiveDEA",
    "__version__",
    "dea",
    "get_method",
    "list_methods",
    "load_eleven_dmu",
    "load_five_dmu",
    "load_internal_mismatch_network",
    "malmquist",
    "malmquist_panel",
    "multiplier_dea",
    "sbm",
    "write_evidence_bundle",
]
