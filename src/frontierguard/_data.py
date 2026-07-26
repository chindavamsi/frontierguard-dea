"""Validated immutable data containers."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from frontierguard._errors import DataValidationError


def _matrix(values: ArrayLike, *, label: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{label} must be a numeric matrix.") from exc
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise DataValidationError(f"{label} must be two-dimensional.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise DataValidationError(f"{label} must contain at least one row and column.")
    if not np.isfinite(array).all():
        raise DataValidationError(f"{label} contain missing or non-finite values.")
    if (array < 0).any():
        raise DataValidationError(
            f"{label} must be non-negative for the supported radial technology."
        )
    if np.any(array.sum(axis=1) <= 0):
        raise DataValidationError(
            f"Every row of {label} must contain a positive value."
        )
    if np.any(array.max(axis=0) <= 0):
        raise DataValidationError(
            f"Every column of {label} must contain a positive value."
        )
    result = np.array(array, dtype=float, copy=True, order="C")
    result.setflags(write=False)
    return result


def _names(
    values: Sequence[object] | None,
    *,
    count: int,
    prefix: str,
    label: str,
) -> tuple[str, ...]:
    names = (
        tuple(f"{prefix}{index + 1}" for index in range(count))
        if values is None
        else tuple(str(value) for value in values)
    )
    if len(names) != count:
        raise DataValidationError(f"{label} must contain exactly {count} names.")
    if any(not name.strip() for name in names):
        raise DataValidationError(f"{label} cannot contain blank names.")
    if len(set(names)) != len(names):
        raise DataValidationError(f"{label} must be unique.")
    return names


def _fingerprint(parts: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(str(part.shape).encode())
            digest.update(part.dtype.str.encode())
            digest.update(part.tobytes(order="C"))
        else:
            digest.update(repr(part).encode())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class DEAData:
    """Inputs and outputs for a single production technology."""

    inputs: NDArray[np.float64]
    outputs: NDArray[np.float64]
    dmu_names: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]

    def __init__(
        self,
        inputs: ArrayLike,
        outputs: ArrayLike,
        dmu_names: Sequence[object] | None = None,
        input_names: Sequence[object] | None = None,
        output_names: Sequence[object] | None = None,
    ) -> None:
        """Validate flexible array-like inputs and store canonical immutable values."""
        resolved_inputs = _matrix(inputs, label="inputs")
        resolved_outputs = _matrix(outputs, label="outputs")
        if resolved_inputs.shape[0] != resolved_outputs.shape[0]:
            raise DataValidationError(
                "inputs and outputs must contain the same number of DMUs."
            )
        object.__setattr__(self, "inputs", resolved_inputs)
        object.__setattr__(self, "outputs", resolved_outputs)
        object.__setattr__(
            self,
            "dmu_names",
            _names(
                dmu_names,
                count=resolved_inputs.shape[0],
                prefix="DMU",
                label="dmu_names",
            ),
        )
        object.__setattr__(
            self,
            "input_names",
            _names(
                input_names,
                count=resolved_inputs.shape[1],
                prefix="x",
                label="input_names",
            ),
        )
        object.__setattr__(
            self,
            "output_names",
            _names(
                output_names,
                count=resolved_outputs.shape[1],
                prefix="y",
                label="output_names",
            ),
        )

    @property
    def n_dmus(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def n_inputs(self) -> int:
        return int(self.inputs.shape[1])

    @property
    def n_outputs(self) -> int:
        return int(self.outputs.shape[1])

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            [
                self.inputs,
                self.outputs,
                self.dmu_names,
                self.input_names,
                self.output_names,
            ]
        )

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        inputs: Sequence[str],
        outputs: Sequence[str],
        dmu: str | None = None,
    ) -> DEAData:
        """Create a validated DEA dataset from named DataFrame columns."""
        if not isinstance(frame, pd.DataFrame):
            raise DataValidationError("frame must be a pandas DataFrame.")
        if not inputs or not outputs:
            raise DataValidationError("inputs and outputs cannot be empty.")
        requested = [*inputs, *outputs]
        missing = [column for column in requested if column not in frame.columns]
        if missing:
            raise DataValidationError(
                "The DataFrame is missing columns: " + ", ".join(missing)
            )
        if len(set(requested)) != len(requested):
            raise DataValidationError(
                "Input and output column selections must not overlap or repeat."
            )
        if dmu is not None and dmu not in frame.columns:
            raise DataValidationError(f"DMU column {dmu!r} does not exist.")
        dmu_names: Sequence[object] = (
            frame.index.tolist() if dmu is None else frame[dmu].tolist()
        )
        return cls(
            inputs=frame.loc[:, list(inputs)].to_numpy(),
            outputs=frame.loc[:, list(outputs)].to_numpy(),
            dmu_names=dmu_names,
            input_names=inputs,
            output_names=outputs,
        )


@dataclass(frozen=True, slots=True, init=False)
class SerialNetworkData:
    """A serial production network with one or more intermediate-link matrices."""

    inputs: NDArray[np.float64]
    links: tuple[NDArray[np.float64], ...]
    outputs: NDArray[np.float64]
    dmu_names: tuple[str, ...]
    input_names: tuple[str, ...]
    link_names: tuple[tuple[str, ...], ...]
    output_names: tuple[str, ...]
    stage_names: tuple[str, ...]

    def __init__(
        self,
        inputs: ArrayLike,
        links: Sequence[ArrayLike],
        outputs: ArrayLike,
        dmu_names: Sequence[object] | None = None,
        input_names: Sequence[object] | None = None,
        link_names: Sequence[Sequence[object]] | None = None,
        output_names: Sequence[object] | None = None,
        stage_names: Sequence[object] | None = None,
    ) -> None:
        """Validate a flexible serial-network specification into immutable arrays."""
        resolved_inputs = _matrix(inputs, label="network inputs")
        resolved_outputs = _matrix(outputs, label="network outputs")
        resolved_links = tuple(
            _matrix(link, label=f"link {index + 1}") for index, link in enumerate(links)
        )
        if not resolved_links:
            raise DataValidationError(
                "A serial network requires at least one intermediate link."
            )
        n_dmus = resolved_inputs.shape[0]
        if resolved_outputs.shape[0] != n_dmus or any(
            link.shape[0] != n_dmus for link in resolved_links
        ):
            raise DataValidationError(
                "Inputs, every link, and outputs must contain the same DMUs."
            )
        if link_names is not None and len(link_names) != len(resolved_links):
            raise DataValidationError(
                "link_names must contain one name sequence per intermediate link."
            )
        resolved_link_names = tuple(
            _names(
                None if link_names is None else link_names[index],
                count=link.shape[1],
                prefix=f"z{index + 1}_",
                label=f"link_names[{index}]",
            )
            for index, link in enumerate(resolved_links)
        )
        object.__setattr__(self, "inputs", resolved_inputs)
        object.__setattr__(self, "links", resolved_links)
        object.__setattr__(self, "outputs", resolved_outputs)
        object.__setattr__(
            self,
            "dmu_names",
            _names(
                dmu_names,
                count=n_dmus,
                prefix="DMU",
                label="dmu_names",
            ),
        )
        object.__setattr__(
            self,
            "input_names",
            _names(
                input_names,
                count=resolved_inputs.shape[1],
                prefix="x",
                label="input_names",
            ),
        )
        object.__setattr__(self, "link_names", resolved_link_names)
        object.__setattr__(
            self,
            "output_names",
            _names(
                output_names,
                count=resolved_outputs.shape[1],
                prefix="y",
                label="output_names",
            ),
        )
        object.__setattr__(
            self,
            "stage_names",
            _names(
                stage_names,
                count=len(resolved_links) + 1,
                prefix="stage_",
                label="stage_names",
            ),
        )

    @property
    def n_dmus(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def n_stages(self) -> int:
        return len(self.links) + 1

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            [
                self.inputs,
                *self.links,
                self.outputs,
                self.dmu_names,
                self.input_names,
                self.link_names,
                self.output_names,
                self.stage_names,
            ]
        )
