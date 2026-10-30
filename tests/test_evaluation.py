import json
from pathlib import Path

import numpy as np

from echo_spin.characterisation import longitudinal_characterisation_report
from echo_spin.evaluation import (
    evaluate_finite_protocols,
    evaluate_ideal_implementation,
    finite_evaluation_report,
    ideal_evaluation_report,
)
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
        assert protocol["phase_prediction_error"] < 1e-10

    assert result["protocols"]["free"]["average_gate_fidelity"] < 0.99
    assert (
        result["protocols"]["closed_global_hahn"]["average_gate_fidelity"]
        < 0.99
    )
    assert result["assessment"]["physical_approximation_validated"] is False

    selective_phases = selective["phases"]
    np.testing.assert_allclose(
        selective_phases["local_generator_phases_rad"],
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        selective_phases["residual_pair_generator_phases_rad"],
        0.0,
        atol=1e-12,
    )

    hahn = result["protocols"]["closed_global_hahn"]
    np.testing.assert_allclose(
        hahn["phases"]["local_generator_phases_rad"],
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        hahn["phases"]["residual_pair_generator_phases_rad"][1][2],
        np.pi / 4,
        atol=1e-12,
    )
    assert hahn["resources"]["individual_pulse_count"] == 6
    assert hahn["resources"]["distinct_pulse_time_count"] == 2

    saved = json.loads(json.dumps(
        ideal_evaluation_report(implementation),
        allow_nan=False,
    ))
    stored = saved["protocols"]["hadamard_selective"]
    assert stored["phase_prediction_error"] < 1e-10

    restored_propagator = (
        np.asarray(stored["propagator"]["real"])
        + 1j * np.asarray(stored["propagator"]["imag"])
    )

    np.testing.assert_allclose(
        restored_propagator,
        selective["propagator"].full(),
        atol=1e-12,
        rtol=0.0,
    )
    assert saved["dimension"] == 8
    assert stored["resources"] == selective["resources"]
    assert saved["assessment"]["physical_approximation_validated"] is False


def test_finite_evaluation_recovers_short_pulse_limit():
    """Recover ideal fidelities with short square and sine pulses."""
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
    ideal = evaluate_ideal_implementation(
        ideal_implementation_report(source)
    )

    for shape in ("square", "sine"):
        finite = evaluate_finite_protocols(
            source,
            pulse_duration_us=1e-6,
            shape=shape,
            steps=256,
        )

        for name, result in finite["protocols"].items():
            reference = ideal["protocols"][name]
            np.testing.assert_allclose(
                result["average_gate_fidelity"],
                reference["average_gate_fidelity"],
                atol=1e-5,
                rtol=0.0,
            )
            assert result["unitarity_error"] < 1e-10
            assert result["resources"] == reference["resources"]
            assert "phase_prediction_error" not in result

        np.testing.assert_allclose(
            finite["protocols"]["free"]["propagator"].full(),
            ideal["protocols"]["free"]["propagator"].full(),
            atol=1e-12,
            rtol=0.0,
        )

        saved = json.loads(json.dumps(
        finite_evaluation_report(
            source,
            pulse_duration_us=0.02,
            shape="square",
        ),
        allow_nan=False,
    ))

    assert saved["pulse_shape"] == "square"
    assert saved["integration_steps"] is None
    assert saved["source_protocol_report"] == source

    for result in saved["protocols"].values():
        matrix = (
            np.asarray(result["propagator"]["real"])
            + 1j * np.asarray(result["propagator"]["imag"])
        )
        np.testing.assert_allclose(
            np.einsum(
                "ki,kj->ij", matrix.conj(), matrix, optimize=False
            ),
            np.eye(saved["dimension"]),
            atol=1e-10,
            rtol=0.0,
        )


def test_zero_drive_recovers_free_evolution_for_every_schedule():
    """Recover free evolution when a common amplitude error disables the drive."""
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

    for shape in ("square", "sine"):
        saved = json.loads(json.dumps(
            finite_evaluation_report(
                source,
                pulse_duration_us=0.02,
                shape=shape,
                steps=8,
                amplitude_error=-1.0,
            ),
            allow_nan=False,
        ))
        assert saved["amplitude_error"] == -1.0
        assert saved["peak_rabi_frequency_rad_per_us"] == 0.0

        def matrix(result):
            """Reconstruct the saved complex propagator."""
            return (
                np.asarray(result["propagator"]["real"])
                + 1j * np.asarray(result["propagator"]["imag"])
            )

        reference = matrix(saved["protocols"]["free"])
        for result in saved["protocols"].values():
            np.testing.assert_allclose(
                matrix(result), reference, atol=1e-10, rtol=0.0
            )


def test_exact_amplitude_calibration_recovers_finite_pulse_baseline():
    """Recover calibrated register operations with an exact gain estimate."""
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
    baseline = evaluate_finite_protocols(
        source, pulse_duration_us=0.02
    )
    corrected = evaluate_finite_protocols(
        source,
        pulse_duration_us=0.02,
        amplitude_error=0.02,
        amplitude_error_estimate=0.02,
    )

    assert corrected["amplitude_error_estimate"] == 0.02
    for name, result in corrected["protocols"].items():
        reference = baseline["protocols"][name]
        np.testing.assert_allclose(
            result["propagator"].full(),
            reference["propagator"].full(),
            atol=1e-12,
            rtol=0.0,
        )
        assert result["resources"] == reference["resources"]
