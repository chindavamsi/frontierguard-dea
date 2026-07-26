"""Radial scale-efficiency and returns-to-scale diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from frontierguard._data import DEAData
from frontierguard._enums import Orientation, ReturnsToScale, SlackStrategy, coerce_enum
from frontierguard._errors import ModelValidationError, NumericalError
from frontierguard._results import RadialResult
from frontierguard._solver import LinearSolver, ScipyHighsSolver
from frontierguard.radial import RadialDEA


def _readonly(values: ArrayLike) -> NDArray[np.float64]:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _close(left: float, right: float, tolerance: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= tolerance * scale


class ScaleRegime(str, Enum):
    """Local returns-to-scale regime identified by the NIRS test."""

    CRS = "crs"
    IRS = "irs"
    DRS = "drs"


@dataclass(frozen=True, slots=True, init=False)
class ScaleEfficiencyResult:
    """CRS, VRS, and NIRS radial fits with scale diagnostics.

    ``scale_efficiency`` is the normalized technical-efficiency ratio
    ``TE_CRS / TE_VRS``. ``scale_regime`` applies the standard NIRS test:
    CRS and VRS equality implies CRS; otherwise NIRS and VRS equality implies
    DRS, with the remaining case classified as IRS.
    """

    dmu_names: tuple[str, ...]
    orientation: Orientation
    crs_efficiency: NDArray[np.float64]
    vrs_efficiency: NDArray[np.float64]
    nirs_efficiency: NDArray[np.float64]
    scale_efficiency: NDArray[np.float64]
    scale_regime: tuple[ScaleRegime, ...]
    classification_tolerance: float
    data_fingerprint: str
    reference_fingerprint: str
    solver: str
    audit_residual: NDArray[np.float64]
    audit_tolerance: float
    crs_result: RadialResult
    vrs_result: RadialResult
    nirs_result: RadialResult

    def __init__(
        self,
        *,
        orientation: Orientation,
        scale_efficiency: ArrayLike,
        scale_regime: tuple[ScaleRegime, ...],
        classification_tolerance: float,
        crs_result: RadialResult,
        vrs_result: RadialResult,
        nirs_result: RadialResult,
    ) -> None:
        """Copy component scores into an immutable diagnostic result."""
        object.__setattr__(self, "dmu_names", crs_result.dmu_names)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(
            self,
            "crs_efficiency",
            _readonly(crs_result.technical_efficiency),
        )
        object.__setattr__(
            self,
            "vrs_efficiency",
            _readonly(vrs_result.technical_efficiency),
        )
        object.__setattr__(
            self,
            "nirs_efficiency",
            _readonly(nirs_result.technical_efficiency),
        )
        object.__setattr__(
            self,
            "scale_efficiency",
            _readonly(scale_efficiency),
        )
        object.__setattr__(self, "scale_regime", tuple(scale_regime))
        object.__setattr__(
            self,
            "classification_tolerance",
            float(classification_tolerance),
        )
        object.__setattr__(
            self,
            "data_fingerprint",
            crs_result.data_fingerprint,
        )
        object.__setattr__(
            self,
            "reference_fingerprint",
            crs_result.reference_fingerprint,
        )
        object.__setattr__(self, "solver", crs_result.solver)
        component_audits = np.maximum.reduce(
            (
                crs_result.audit_residual,
                vrs_result.audit_residual,
                nirs_result.audit_residual,
            )
        )
        object.__setattr__(self, "audit_residual", _readonly(component_audits))
        object.__setattr__(
            self,
            "audit_tolerance",
            max(
                crs_result.audit_tolerance,
                vrs_result.audit_tolerance,
                nirs_result.audit_tolerance,
            ),
        )
        object.__setattr__(self, "crs_result", crs_result)
        object.__setattr__(self, "vrs_result", vrs_result)
        object.__setattr__(self, "nirs_result", nirs_result)

    def to_frame(self) -> pd.DataFrame:
        """Return component scores, scale efficiency, and regime by DMU."""
        return pd.DataFrame(
            {
                "dmu": self.dmu_names,
                "crs_efficiency": self.crs_efficiency,
                "vrs_efficiency": self.vrs_efficiency,
                "nirs_efficiency": self.nirs_efficiency,
                "scale_efficiency": self.scale_efficiency,
                "scale_regime": [regime.value for regime in self.scale_regime],
                "audit_residual": self.audit_residual,
            }
        ).set_index("dmu")

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible diagnostic result."""
        return {
            "schema_version": "0.1.0",
            "model": "scale_efficiency",
            "orientation": self.orientation.value,
            "solver": self.solver,
            "classification_tolerance": self.classification_tolerance,
            "data_fingerprint": self.data_fingerprint,
            "reference_fingerprint": self.reference_fingerprint,
            "audit_residual": self.audit_residual.tolist(),
            "audit_tolerance": self.audit_tolerance,
            "dmu_names": list(self.dmu_names),
            "crs_efficiency": self.crs_efficiency.tolist(),
            "vrs_efficiency": self.vrs_efficiency.tolist(),
            "nirs_efficiency": self.nirs_efficiency.tolist(),
            "scale_efficiency": self.scale_efficiency.tolist(),
            "scale_regime": [regime.value for regime in self.scale_regime],
            "component_statuses": {
                "crs": [status.value for status in self.crs_result.statuses],
                "vrs": [status.value for status in self.vrs_result.statuses],
                "nirs": [status.value for status in self.nirs_result.statuses],
            },
        }


class ScaleEfficiencyDEA:
    """Compare CRS, VRS, and NIRS radial models and diagnose scale.

    The three component models use phase-one radial scores. The implementation
    works with FrontierGuard's common technical-efficiency convention, so the
    formula is identical for input and output orientation.
    """

    def __init__(
        self,
        *,
        orientation: Orientation | str = Orientation.INPUT,
        solver: LinearSolver | None = None,
        tolerance: float = 1e-9,
        classification_tolerance: float = 1e-7,
    ) -> None:
        self.orientation = coerce_enum(
            orientation,
            Orientation,
            label="orientation",
        )
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ModelValidationError("tolerance must be a positive finite number.")
        if not np.isfinite(classification_tolerance) or classification_tolerance <= 0:
            raise ModelValidationError(
                "classification_tolerance must be a positive finite number."
            )
        self.tolerance = float(tolerance)
        self.classification_tolerance = float(classification_tolerance)
        self.solver = ScipyHighsSolver() if solver is None else solver

    def fit(
        self,
        data: DEAData,
        *,
        reference: DEAData | None = None,
    ) -> ScaleEfficiencyResult:
        """Fit the component technologies and classify each evaluated DMU."""
        component_results: dict[ReturnsToScale, RadialResult] = {}
        for rts in (
            ReturnsToScale.CRS,
            ReturnsToScale.VRS,
            ReturnsToScale.NIRS,
        ):
            component_results[rts] = RadialDEA(
                orientation=self.orientation,
                rts=rts,
                slacks=SlackStrategy.NONE,
                solver=self.solver,
                tolerance=self.tolerance,
            ).fit(data, reference=reference)

        crs_result = component_results[ReturnsToScale.CRS]
        vrs_result = component_results[ReturnsToScale.VRS]
        nirs_result = component_results[ReturnsToScale.NIRS]
        if not (crs_result.dmu_names == vrs_result.dmu_names == nirs_result.dmu_names):
            raise NumericalError(
                "Scale-efficiency component DMU order is inconsistent."
            )

        crs = np.asarray(crs_result.technical_efficiency)
        vrs = np.asarray(vrs_result.technical_efficiency)
        nirs = np.asarray(nirs_result.technical_efficiency)
        if (vrs <= 0).any():
            raise NumericalError("VRS technical efficiencies must be positive.")

        raw_scale_efficiency = crs / vrs
        tolerance = self.classification_tolerance
        if (raw_scale_efficiency <= 0).any() or (
            raw_scale_efficiency > 1.0 + tolerance
        ).any():
            raise NumericalError(
                "CRS/VRS scale-efficiency ratios violate the interval (0, 1]."
            )
        scale_efficiency = np.minimum(raw_scale_efficiency, 1.0)
        scale_efficiency[
            np.abs(scale_efficiency - 1.0) <= self.classification_tolerance
        ] = 1.0

        regimes: list[ScaleRegime] = []
        for crs_score, vrs_score, nirs_score in zip(
            crs,
            vrs,
            nirs,
            strict=True,
        ):
            if _close(float(crs_score), float(vrs_score), tolerance):
                regimes.append(ScaleRegime.CRS)
            elif _close(float(nirs_score), float(vrs_score), tolerance):
                regimes.append(ScaleRegime.DRS)
            else:
                regimes.append(ScaleRegime.IRS)

        return ScaleEfficiencyResult(
            orientation=self.orientation,
            scale_efficiency=scale_efficiency,
            scale_regime=tuple(regimes),
            classification_tolerance=self.classification_tolerance,
            crs_result=crs_result,
            vrs_result=vrs_result,
            nirs_result=nirs_result,
        )
