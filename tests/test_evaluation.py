import json
from pathlib import Path

import numpy as np

from echo_spin.characterisation import longitudinal_characterisation_report
from echo_spin.evaluation import evaluate_ideal_implementation
from echo_spin.implementation import ideal_implementation_report
from echo_spin.protocols import ideal_protocol_report
from echo_spin.representation import longitudinal_model_report
from echo_spin.scenario import longitudinal_scenario_from_dict


def test_selective_operation_matches_whole_register_target():
    """Protect the spectator while exposing unwanted evolution in the baselines."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/scenario.json"
    )
    scenario = longitudinal_scenario_from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
    model = longitudinal_model_report(
        longitudinal_characterisation_report(scenario)
    )
    implementation = ideal_implementation_report(
        ideal_protocol_report(model)
    )

    result = evaluate_ideal_implementation(implementation)
    selective = result["protocols"]["hadamard_selective"]

    np.testing.assert_allclose(
        selective["average_gate_fidelity"], 1.0, atol=1e-12, rtol=0
    )
    assert selective["phase_aligned_frobenius_error"] < 1e-10

    for protocol in result["protocols"].values():
        assert protocol["unitarity_error"] < 1e-10

    assert result["protocols"]["free"]["average_gate_fidelity"] < 0.99
    assert (
        result["protocols"]["closed_global_hahn"]["average_gate_fidelity"]
        < 0.99
    )
    assert result["assessment"]["physical_approximation_validated"] is False
