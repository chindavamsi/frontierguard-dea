from __future__ import annotations

import json

import frontierguard
from frontierguard.cli import main


def test_version_command(capsys: object) -> None:
    assert main(["version"]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]

    assert output.strip() == frontierguard.__version__


def test_methods_command_is_filterable_and_machine_readable(capsys: object) -> None:
    assert (
        main(
            [
                "methods",
                "--stability",
                "experimental",
                "--problem-class",
                "linear_program",
                "--json",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    methods = json.loads(output)

    assert [method["method_id"] for method in methods] == ["network.dag.relational"]


def test_doctor_reports_runtime_without_secrets(capsys: object) -> None:
    assert main(["doctor", "--json"]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    diagnostics = json.loads(output)

    assert diagnostics["frontierguard"] == frontierguard.__version__
    assert diagnostics["dependencies"]["numpy"]
    assert "python" in diagnostics
