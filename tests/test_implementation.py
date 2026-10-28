import json
from pathlib import Path

import numpy as np
import pytest
from qutip import qeye

from echo_spin.characterisation import longitudinal_characterisation_report
from echo_spin.dynamics.sequences import finite_pulse_sequence_propagator
from echo_spin.implementation import (
    ideal_implementation_report,
    implement_finite_protocols,
    implement_ideal_protocols,
    implement_square_protocols,
)
from echo_spin.protocols import ideal_protocol_report
from echo_spin.representation import longitudinal_model_report
from echo_spin.scenario import longitudinal_scenario_from_dict


def test_ideal_implementation_groups_pulses_and_preserves_closing_events():
    """Compile simultaneous Hahn pulses without dropping the terminal event."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/scenario.json"
    )
    scenario = longitudinal_scenario_from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
    model_report = longitudinal_model_report(
        longitudinal_characterisation_report(scenario)
    )
    result = implement_ideal_protocols(ideal_protocol_report(model_report))
    duration = result["duration_us"]

    free = result["protocols"]["free"]
    assert free["pulse_propagators"] == ()
    np.testing.assert_allclose(free["free_intervals_us"], [duration])

    hahn = result["protocols"]["closed_global_hahn"]
    assert hahn["addressed_sites"] == [[0, 1, 2], [0, 1, 2]]
    np.testing.assert_allclose(
        hahn["free_intervals_us"],
        [duration / 2, duration / 2, 0.0],
    )

    operators = result["model"].pauli_operators
    expected = (
        (-1j)**3
        * operators["X"][0]
        * operators["X"][1]
        * operators["X"][2]
    )
    assert (hahn["pulse_propagators"][0] - expected).norm() < 1e-12

    for protocol in result["protocols"].values():
        assert len(protocol["free_intervals_us"]) == (
            len(protocol["pulse_propagators"]) + 1
        )
        np.testing.assert_allclose(
            sum(protocol["free_intervals_us"]), duration
        )
        for pulse in protocol["pulse_propagators"]:
            assert (
                pulse.dag() * pulse - qeye([2, 2, 2])
            ).norm() < 1e-12

    source = ideal_protocol_report(model_report)
    saved = json.loads(json.dumps(
        ideal_implementation_report(source),
        allow_nan=False,
    ))
    timeline = saved["protocols"]["closed_global_hahn"]

    assert timeline["event_operator_count"] == 2
    assert timeline["addressed_sites"] == [[0, 1, 2], [0, 1, 2]]
    assert all(timeline["checks"].values())
    assert timeline["free_intervals_us"][-1] == 0.0

    reconstructed = implement_ideal_protocols(
        saved["source_protocol_report"]
    )

    for name, original in result["protocols"].items():
        restored = reconstructed["protocols"][name]
        stored = saved["protocols"][name]

        np.testing.assert_allclose(
            restored["event_times_us"],
            stored["event_times_us"],
        )
        np.testing.assert_allclose(
            restored["free_intervals_us"],
            stored["free_intervals_us"],
        )
        assert restored["addressed_sites"] == stored["addressed_sites"]

        assert len(restored["pulse_propagators"]) == len(
            original["pulse_propagators"]
        )
        for actual, expected in zip(
            restored["pulse_propagators"],
            original["pulse_propagators"],
        ):
            assert (actual - expected).norm() < 1e-12

    source["target_diagonal"]["real"][0] += 0.1
    with pytest.raises(ValueError, match="Saved target"):
        implement_ideal_protocols(source)


def test_square_controls_preserve_duration_and_approach_ideal_limit():
    """Preserve elapsed time, recover ideal control and reject overlapping windows."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/scenario.json"
    )
    scenario = longitudinal_scenario_from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
    protocol_report = ideal_protocol_report(
        longitudinal_model_report(
            longitudinal_characterisation_report(scenario)
        )
    )
    ideal = implement_ideal_protocols(protocol_report)
    finite = implement_square_protocols(
        protocol_report,
        pulse_duration_us=1e-6,
    )

    for name, control in finite["protocols"].items():
        count = len(control["pulse_propagators"])
        np.testing.assert_allclose(
            control["free_intervals_us"].sum()
            + count * finite["pulse_duration_us"],
            finite["duration_us"],
        )
        assert np.all(control["pulse_start_times_us"] >= 0)

        actual = finite_pulse_sequence_propagator(
            finite["model"].hamiltonian,
            control["free_intervals_us"],
            control["pulse_propagators"],
        )
        reference = ideal["protocols"][name]
        expected = finite_pulse_sequence_propagator(
            ideal["model"].hamiltonian,
            reference["free_intervals_us"],
            reference["pulse_propagators"],
        )
        error = np.linalg.norm(
            actual.full() - expected.full(), ord="fro"
        ) / np.sqrt(ideal["model"].system.dimension)
        assert error < 2e-5

    with pytest.raises(ValueError, match="overlap"):
        implement_square_protocols(
            protocol_report,
            pulse_duration_us=ideal["duration_us"] / 4,
        )


def test_sine_controls_converge_with_time_resolution():
    """Check sine-pulse normalization and convergence of register evolution."""
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

    duration = 0.02
    propagators = []

    for steps in (64, 128, 256):
        result = implement_finite_protocols(
            source,
            pulse_duration_us=duration,
            shape="sine",
            steps=steps,
        )
        np.testing.assert_allclose(
            result["peak_rabi_frequency_rad_per_us"],
            np.pi**2 / (2 * duration),
        )

        control = result["protocols"]["hadamard_selective"]
        np.testing.assert_allclose(
            control["free_intervals_us"].sum()
            + len(control["pulse_propagators"]) * duration,
            result["duration_us"],
        )
        propagators.append(
            finite_pulse_sequence_propagator(
                result["model"].hamiltonian,
                control["free_intervals_us"],
                control["pulse_propagators"],
            ).full()
        )

    dimension = propagators[0].shape[0]
    coarse = np.linalg.norm(
        propagators[1] - propagators[0], ord="fro"
    ) / np.sqrt(dimension)
    fine = np.linalg.norm(
        propagators[2] - propagators[1], ord="fro"
    ) / np.sqrt(dimension)

    assert fine < coarse
    assert fine < 2e-3
