import json
from pathlib import Path

import pytest

from echo_spin.cli import main


def test_register_cli_reports_resolved_configuration(tmp_path, capsys):
    """Expose the loaded geometry and explicit units through valid JSON output."""
    configuration = {
        "positions": [[0, 0, 0]],
        "nv_axes": [[0, 0, 1]],
        "transverse_references": [[1, 0, 0]],
        "field_at_reference": [0, 0, 0.01],
    }
    path = tmp_path / "register.json"
    path.write_text(json.dumps(configuration), encoding="utf-8")

    assert main([str(path)]) == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert captured.err == ""
    assert report["number_of_nv"] == 1
    assert report["units"]["positions"] == "m"
    assert report["units"]["magnetic_fields"] == "T"
    assert report["positions"] == [[0.0, 0.0, 0.0]]
    assert report["magnetic_fields"] == [[0.0, 0.0, 0.01]]


def test_register_cli_reports_missing_file(tmp_path, capsys):
    """Return a command-line error without emitting a partial register report."""
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path / "missing.json")])

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err


def test_scenario_cli_reports_complete_handoff(capsys):
    """Expose the resolved register, task, provenance and modelling limits."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/scenario.json"
    )
    original = json.loads(path.read_text(encoding="utf-8"))

    assert main([str(path), "--scenario"]) == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert captured.err == ""
    assert report["number_of_nv"] == 3
    assert report["specification"] == original
    assert report["specification"]["task"]["target_pair"] == [0, 1]
    assert report["units"]["gamma_e"] == "rad/us/T"
    assert report["scope"]["experimental_feasibility"] == "Not assessed."
    assert report["scope"]["individual_addressability"] == (
        "Assumed, not established."
    )
    assert report["software"]["python"]
    assert "echo-spin" in report["software"]["packages"]
    assert "git_revision" in report["software"]
    assert "git_dirty" in report["software"]
    assert report["validation"]["geometry_and_frames"] == "Passed."
    assert report["validation"]["hardware_feasibility"] == "Not assessed."
    assert report["validation"]["physical_model_validity"] == (
        "Not established by input validation."
    )


def test_scenario_cli_rejects_invalid_task_without_partial_output(
    tmp_path, capsys
):
    """Reject an invalid scenario before emitting a handoff report."""
    source = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/scenario.json"
    )
    specification = json.loads(source.read_text(encoding="utf-8"))
    specification["task"]["target_pair"] = [0, 3]

    path = tmp_path / "invalid_scenario.json"
    path.write_text(json.dumps(specification), encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main([str(path), "--scenario"])

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "target_pair" in captured.err


def test_scenario_cli_distinguishes_validated_and_inapplicable_records(capsys):
    """Report section status without claiming propagation or physical validity."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/scenario.json"
    )

    assert main([str(path), "--scenario"]) == 0
    report = json.loads(capsys.readouterr().out)

    checks = report["environment_validation"]
    assert checks["noise"] == {
        "status": "specified",
        "parameter_checks": "Passed for the supported schema.",
    }
    assert checks["preparation"]["status"] == "not_applicable"
    assert checks["preparation"]["parameter_checks"].startswith("Not performed")
    assert report["constraint_checks"]["minimum_separation"] == "Passed."
    assert report["validation"]["physical_model_validity"] == (
        "Not established by input validation."
    )
