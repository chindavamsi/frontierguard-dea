from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import frontierguard as fg


def test_evidence_bundle_is_strict_complete_and_checksummed(tmp_path: Path) -> None:
    result = fg.RadialDEA(rts="crs").fit(fg.load_five_dmu())
    target = fg.write_evidence_bundle(
        result,
        tmp_path / "evidence",
        method_id="radial.envelopment",
    )

    assert sorted(path.name for path in target.iterdir()) == [
        "METHOD_CARD.md",
        "manifest.json",
        "result.json",
        "scores.csv",
    ]
    evidence = json.loads((target / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    card = (target / "METHOD_CARD.md").read_text(encoding="utf-8")
    assert evidence["model"] == "radial_envelopment"
    assert evidence["method_id"] == "radial.envelopment"
    assert manifest["method_id"] == "radial.envelopment"
    assert "causality" in card
    for name, record in manifest["files"].items():
        payload = (target / name).read_bytes()
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()
        assert record["bytes"] == len(payload)


def test_evidence_bundle_refuses_implicit_overwrite(tmp_path: Path) -> None:
    result = fg.RadialDEA(rts="crs").fit(fg.load_five_dmu())
    target = fg.write_evidence_bundle(
        result,
        tmp_path / "evidence",
        method_id="radial.envelopment",
    )
    unrelated = target / "research-notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        fg.write_evidence_bundle(
            result,
            target,
            method_id="radial.envelopment",
        )
    fg.write_evidence_bundle(
        result,
        target,
        method_id="radial.envelopment",
        overwrite=True,
    )
    assert unrelated.read_text(encoding="utf-8") == "keep me"


def test_evidence_bundle_requires_registered_method(tmp_path: Path) -> None:
    result = fg.RadialDEA(rts="crs").fit(fg.load_five_dmu())
    target = tmp_path / "evidence"

    with pytest.raises(fg.ModelValidationError, match="Unknown method_id"):
        fg.write_evidence_bundle(
            result,
            target,
            method_id="unknown",
        )
    assert not target.exists()


def test_evidence_bundle_rejects_mislabeled_result_before_writing(
    tmp_path: Path,
) -> None:
    result = fg.sbm(fg.load_five_dmu(), rts="crs")
    target = tmp_path / "evidence"

    with pytest.raises(fg.ModelValidationError, match="incompatible"):
        fg.write_evidence_bundle(
            result,
            target,
            method_id="radial.envelopment",
        )
    assert not target.exists()


def test_multiplier_bundle_reports_nested_audit_maximum(tmp_path: Path) -> None:
    result = fg.multiplier_dea(fg.load_five_dmu())
    target = fg.write_evidence_bundle(
        result,
        tmp_path / "evidence",
        method_id="multiplier.crs.cross",
    )
    card = (target / "METHOD_CARD.md").read_text(encoding="utf-8")

    assert f"{result.max_audit_residual:.6e}" in card
    assert "not reported" not in card.split("## Citations", maxsplit=1)[0]
