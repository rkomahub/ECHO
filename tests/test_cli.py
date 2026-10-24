import json

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
