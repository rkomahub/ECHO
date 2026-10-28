import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from echo_spin.characterisation import longitudinal_characterisation_report
from echo_spin.comparisons import scan_pair_separations, scan_pulse_durations
from echo_spin.protocols import ideal_protocol_report
from echo_spin.representation import longitudinal_model_report
from echo_spin.scenario import longitudinal_scenario_from_dict


def test_spacing_scan_preserves_failures_and_recovers_cubic_gate_time():
    """Retain rejected points while verifying ideal scaling and input preservation."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/scenario.json"
    )
    specification = json.loads(path.read_text(encoding="utf-8"))
    original = deepcopy(specification)

    rejected, reference, expanded = scan_pair_separations(
        specification,
        [4e-9, 15e-9, 30e-9],
    )

    assert specification == original
    assert rejected["status"] == "rejected"
    assert "minimum_separation" in rejected["reason"]
    assert reference["status"] == expanded["status"] == "evaluated"

    np.testing.assert_allclose(
        expanded["target_coupling_rad_per_us"],
        reference["target_coupling_rad_per_us"] / 8,
    )
    np.testing.assert_allclose(
        expanded["duration_us"],
        8 * reference["duration_us"],
    )

    for record in (reference, expanded):
        np.testing.assert_allclose(
            record["protocols"]["hadamard_selective"]["average_gate_fidelity"],
            1.0,
            atol=1e-12,
            rtol=0.0,
        )

    restored = json.loads(json.dumps(
        [rejected, reference, expanded],
        allow_nan=False,
    ))
    assert restored[0]["status"] == "rejected"
    assert restored[1]["protocols"]["hadamard_selective"]["resources"][
        "individual_pulse_count"
    ] == 20

    for name, baseline in reference["protocols"].items():
        scaled = expanded["protocols"][name]

        np.testing.assert_allclose(
            scaled["average_gate_fidelity"],
            baseline["average_gate_fidelity"],
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            scaled["phases"]["local_generator_phases_rad"],
            baseline["phases"]["local_generator_phases_rad"],
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            scaled["phases"]["pair_generator_phases_rad"],
            baseline["phases"]["pair_generator_phases_rad"],
            atol=1e-12,
            rtol=0.0,
        )
        assert scaled["resources"]["individual_pulse_count"] == (
            baseline["resources"]["individual_pulse_count"]
        )
        assert scaled["resources"]["distinct_pulse_time_count"] == (
            baseline["resources"]["distinct_pulse_time_count"]
        )


def test_pulse_scan_saves_both_shapes_and_rejected_windows():
    """Preserve scan inputs, resolution diagnostics and infeasible durations."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/scenario.json"
    )
    scenario = longitudinal_scenario_from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
    source = ideal_protocol_report(
        longitudinal_model_report(
            longitudinal_characterisation_report(scenario)
        )
    )
    original = deepcopy(source)
    saved = json.loads(json.dumps(
        scan_pulse_durations(
            source, [0.02, source["duration_us"]], steps=8
        ),
        allow_nan=False,
    ))

    assert source == original
    evaluated = saved["records"][:2]
    rejected = saved["records"][2:]

    assert {record["pulse_shape"] for record in evaluated} == {
        "square", "sine"
    }
    assert all(record["status"] == "evaluated" for record in evaluated)
    assert all(record["status"] == "rejected" for record in rejected)
    assert all(record["reason"] for record in rejected)

    sine = next(
        record for record in evaluated
        if record["pulse_shape"] == "sine"
    )
    assert sine["integration_steps"] == 16
    for result in sine["protocols"].values():
        assert result["resolution_change"] >= 0
        assert result["unitarity_error"] < 1e-10
        assert result["fidelity_resolution_change"] >= 0
