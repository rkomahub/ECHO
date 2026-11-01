import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from qutip import Qobj, basis, tensor

from echo_spin.characterisation import (
    characterise_local_electrons,
    longitudinal_characterisation_report,
)
from echo_spin.dynamics.propagators import hermitian_propagator
from echo_spin.dynamics.sequences import (
    hermitian_pulse_propagator,
    periodic_square_propagator,
)
from echo_spin.electronic_reference import (
    build_shared_drive_rwa,
    compile_shared_selective_events,
    electronic_magnetic_drive_operator,
    electronic_population_diagnostics,
    local_square_pulse_populations,
    propagate_shared_events,
    shared_multitone_hamiltonian,
    shared_pulse_gate_diagnostics,
    shared_pulse_populations,
    shared_square_pulse_populations,
)
from echo_spin.protocols import ideal_protocol_report
from echo_spin.representation import longitudinal_model_report
from echo_spin.scenario import longitudinal_scenario_from_dict


def test_electronic_populations_conserve_probability_and_resolve_exchange():
    """Resolve logical exchange while accounting separately for leakage."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/ideal/scenario.json"
    )
    scenario = longitudinal_scenario_from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
    result = electronic_population_diagnostics(scenario, [0.0, 1.0])
    initial, evolved = result["records"]

    np.testing.assert_allclose(
        initial["mean_stay_probability"], 1.0, atol=1e-12
    )
    for record in result["records"]:
        total = sum(
            record[key] for key in (
                "mean_stay_probability",
                "mean_exchange_probability",
                "mean_other_logical_transfer_probability",
                "mean_leakage_probability",
            )
        )
        np.testing.assert_allclose(total, 1.0, atol=1e-8, rtol=0.0)

    assert evolved["mean_exchange_probability"] > 1e-6
    assert (
        evolved["mean_leakage_probability"]
        < evolved["mean_exchange_probability"]
    )

    np.testing.assert_allclose(
        initial["longitudinal_comparison_fidelity"],
        1.0,
        atol=1e-12,
        rtol=0.0,
    )
    assert evolved["longitudinal_comparison_fidelity"] < 1 - 1e-6


def test_uniform_microwave_field_couples_to_every_aligned_site():
    """Recover equal physical transition couplings under a uniform shared field."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/ideal/scenario.json"
    )
    scenario = longitudinal_scenario_from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
    amplitude = 1e-4
    fields = np.tile([amplitude, 0.0, 0.0], (3, 1))
    drive = electronic_magnetic_drive_operator(scenario, fields)
    initial = tensor([basis(3, 1)] * 3)

    assert drive.isherm
    for site in range(3):
        states = [basis(3, 1)] * 3
        states[site] = basis(3, 2)
        final = tensor(states)
        expected = (
            scenario.specification["sites"][site]["gamma_e"]
            * amplitude / np.sqrt(2)
        )
        np.testing.assert_allclose(
            drive.matrix_element(final.dag(), initial),
            expected,
            atol=1e-12,
        )


def test_drive_transitions_resolve_a_source_free_gradient():
    """Resolve gradient-shifted lines with the physical microwave matrix elements."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/ideal/scenario.json"
    )
    specification = json.loads(path.read_text(encoding="utf-8"))
    fields = np.tile([1e-4, 0.0, 0.0], (3, 1))
    uniform = characterise_local_electrons(
        longitudinal_scenario_from_dict(specification),
        field_profile_tesla=fields,
    )
    frequencies = [
        site["transitions"][0]["frequency_rad_per_us"]
        for site in uniform
    ]
    np.testing.assert_allclose(frequencies, frequencies[0])

    candidate = deepcopy(specification)
    gradient = 1e5
    candidate["register"]["gradient"] = [
        [0, 0, gradient], [0, 0, 0], [gradient, 0, 0]
    ]
    shifted = characterise_local_electrons(
        longitudinal_scenario_from_dict(candidate),
        field_profile_tesla=fields,
    )
    frequencies = [
        site["transitions"][0]["frequency_rad_per_us"]
        for site in shifted
    ]
    gamma = specification["sites"][0]["gamma_e"]
    np.testing.assert_allclose(
        np.diff(frequencies), -gamma * gradient * 15e-9
    )
    for site in shifted:
        np.testing.assert_allclose(
            site["transitions"][0]["strength"],
            (gamma * 1e-4)**2 / 2,
        )

    transition = uniform[0]["transitions"][0]
    carrier = transition["frequency_rad_per_us"]
    duration = np.pi / np.sqrt(transition["strength"])

    unresolved = local_square_pulse_populations(
        uniform, carrier, duration
    )
    resolved = local_square_pulse_populations(
        shifted, carrier, duration
    )

    assert all(
        site["eigenstate_populations"][1] > 0.99
        for site in unresolved
    )
    assert resolved[0]["eigenstate_populations"][1] > 0.99
    assert all(
        sum(site["eigenstate_populations"][1:]) < 0.01
        for site in resolved[1:]
    )


def test_shared_rwa_retains_exchange_and_matches_isolated_pulses():
    """Retain exchange while recovering the established isolated-site drive response."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/ideal/scenario.json"
    )
    scenario = longitudinal_scenario_from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
    fields = np.tile([1e-4, 0.0, 0.0], (3, 1))
    spectra = characterise_local_electrons(
        scenario, field_profile_tesla=fields
    )
    transition = spectra[0]["transitions"][0]
    carrier = transition["frequency_rad_per_us"]
    duration = np.pi / np.sqrt(transition["strength"])
    model = build_shared_drive_rwa(scenario, fields, carrier)

    interaction = model["dipolar_hamiltonian"]
    number = model["excitation_operator"]
    assert interaction.isherm
    assert (interaction * number - number * interaction).norm() < 1e-10

    initial_exchange = tensor(basis(3, 1), basis(3, 0), basis(3, 0))
    final_exchange = tensor(basis(3, 0), basis(3, 1), basis(3, 0))
    assert abs(interaction.matrix_element(
        final_exchange.dag(), initial_exchange
    )) > 1e-6

    # Disable interactions only for comparison with the isolated-site result.
    initial = tensor([basis(3, 0)] * 3)
    state = hermitian_propagator(
        model["local_rotating_hamiltonian"] + model["drive_operator"],
        duration,
    ) * initial
    expected = local_square_pulse_populations(spectra, carrier, duration)

    for site in range(3):
        np.testing.assert_allclose(
            state.ptrace(site).diag(),
            expected[site]["eigenstate_populations"],
            atol=1e-9,
            rtol=0.0,
        )

    interacting = shared_square_pulse_populations(
        scenario, fields, carrier, duration
    )
    np.testing.assert_allclose(
        interacting["total_probability"], 1.0, atol=1e-10
    )
    for site in interacting["local_populations"]:
        np.testing.assert_allclose(
            sum(site["eigenstate_populations"]), 1.0, atol=1e-10
        )

    constant_envelope = shared_pulse_populations(
        scenario,
        fields,
        carrier,
        duration,
        envelope=lambda time: 1.0,
        steps=4,
    )
    for actual, expected_site in zip(
        constant_envelope["local_populations"],
        interacting["local_populations"],
    ):
        np.testing.assert_allclose(
            actual["eigenstate_populations"],
            expected_site["eigenstate_populations"],
            atol=1e-10,
            rtol=0.0,
        )
    np.testing.assert_allclose(
        constant_envelope["target_population_probability"],
        interacting["target_population_probability"],
        atol=1e-10,
        rtol=0.0,
    )

    zero_time = shared_square_pulse_populations(
        scenario, fields, carrier, 0.0
    )
    assert zero_time["leakage_probability"] < 1e-20
    for site in zero_time["local_populations"]:
        np.testing.assert_allclose(
            site["eigenstate_populations"], [1, 0, 0], atol=1e-12
        )

    zero_gate = shared_pulse_gate_diagnostics(
        scenario, fields, carrier, 0.0
    )
    np.testing.assert_allclose(
        zero_gate["average_gate_fidelity"], 1 / 9, atol=1e-12
    )
    assert zero_gate["average_leakage"] < 1e-20

    # Two half-amplitude tones reproduce the original single tone.
    tones = [
        {"carrier_rad_per_us": carrier, "envelope": 0.5},
        {"carrier_rad_per_us": carrier, "envelope": 0.5},
    ]
    actual = shared_multitone_hamiltonian(0.3, model, tones)
    expected = model["drift_hamiltonian"] + model["drive_operator"]
    assert (actual - expected).norm("fro") < 1e-10

    # A carrier offset accumulates phase in the common rotating frame.
    shifted = shared_multitone_hamiltonian(
        np.pi / 2,
        model,
        [{"carrier_rad_per_us": carrier + 1.0}],
    )
    quadrature = build_shared_drive_rwa(
        scenario, fields, carrier, phase=np.pi / 2
    )
    expected = (
        quadrature["drift_hamiltonian"] + quadrature["drive_operator"]
    )
    assert (shifted - expected).norm("fro") < 1e-10

    protocol = ideal_protocol_report(
        longitudinal_model_report(
            longitudinal_characterisation_report(scenario)
        )
    )

    free = compile_shared_selective_events(
        protocol, model, duration, protocol_name="free"
    )
    assert free["events"] == []

    hahn = compile_shared_selective_events(
        protocol, model, duration, protocol_name="closed_global_hahn"
    )
    np.testing.assert_allclose(
        [event["end_us"] for event in hahn["events"]],
        [hahn["duration_us"] / 2, hahn["duration_us"]],
    )
    assert all(len(event["tones"]) == 3 for event in hahn["events"])

    events = compile_shared_selective_events(
        protocol, model, duration
    )

    assert len(events["events"]) == 8
    assert sum(
        len(event["tones"]) for event in events["events"]
    ) == 20
    assert events["events"][-1]["end_us"] == events["duration_us"]

    gamma = scenario.specification["sites"][0]["gamma_e"]
    expected_scale = np.pi / (gamma * 1e-4 / np.sqrt(2) * duration)
    for event in events["events"]:
        for tone in event["tones"]:
            np.testing.assert_allclose(
                tone["envelope_scale"], expected_scale
            )

    disabled = deepcopy(events)
    for event in disabled["events"]:
        for tone in event["tones"]:
            tone["envelope_scale"] = 0.0

    actual = propagate_shared_events(model, disabled, steps=8)
    expected = hermitian_propagator(
        model["drift_hamiltonian"], disabled["duration_us"]
    )
    np.testing.assert_allclose(
        actual.full(), expected.full(), atol=1e-7, rtol=0.0
    )


def test_periodic_propagation_matches_direct_time_ordering():
    """Recover direct propagation including a fractional final carrier period."""
    free = Qobj([[1, 0], [0, -1]])
    drive = Qobj([[0, 0.2], [0.2, 0]])
    carrier = 10.0
    duration = 3.25 * 2 * np.pi / carrier

    repeated = periodic_square_propagator(
        free, drive, carrier, duration, steps_per_period=64
    )
    direct = hermitian_pulse_propagator(
        free,
        drive,
        duration,
        lambda time: np.cos(carrier * time),
        steps=208,
    )
    np.testing.assert_allclose(
        repeated.full(), direct.full(), atol=1e-10, rtol=0.0
    )
