import json
from pathlib import Path

import numpy as np
from qutip import Qobj, basis, qeye, tensor

from echo_spin.core.operators import spin_operators
from echo_spin.dynamics.propagators import hermitian_propagator
from echo_spin.nitrogen_reference import (
    build_nitrogen_drive_rwa,
    build_nitrogen_reference,
    calibrate_local_nitrogen_square_pulse,
    electron_nitrogen_preparation,
    joint_exponential_action,
    nitrogen_electronic_gate_metrics,
    nitrogen_local_spectra,
    nitrogen_logical_input_columns,
    nitrogen_magnetic_drive_operator,
    nitrogen_product_preparation,
    propagate_nitrogen_events,
    slow_operator_in_nitrogen_rwa,
)
from echo_spin.scenario import longitudinal_scenario_from_dict


def test_nitrogen_reference_preserves_order_and_signed_local_terms():
    """Recover the nuclear Zeeman and hyperfine splitting in interleaved order."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/ideal/scenario.json"
    )
    scenario = longitudinal_scenario_from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
    records = {
        "sites": [
            {
                "isotope": "14N",
                "spin": 1.0,
                "gamma_n": -20.0,
                "Q": -1.7,
                "A": np.diag([0.1, 0.2, 0.3]).tolist(),
                "frame": "nv_local",
                "source": "Synthetic values for subsystem-order verification.",
            }
            for _ in range(3)
        ]
    }
    reference = build_nitrogen_reference(scenario, records)
    hamiltonian = reference["hamiltonian"]

    assert hamiltonian.shape == (729, 729)
    assert hamiltonian.isherm
    assert reference["subsystem_order"] == [
        "e0", "n0", "e1", "n1", "e2", "n2"
    ]

    states = [basis(3, 1)] * 6
    states[0] = basis(3, 2)  # electron 0: ms=-1
    states[1] = basis(3, 0)  # nucleus 0: mI=+1
    plus = tensor(states)
    states[1] = basis(3, 2)  # nucleus 0: mI=-1
    minus = tensor(states)

    difference = (
        hamiltonian.matrix_element(plus.dag(), plus)
        - hamiltonian.matrix_element(minus.dag(), minus)
    )
    omega_n = -20.0 * scenario.register.magnetic_fields[0, 2]
    np.testing.assert_allclose(
        difference, 2 * (omega_n - 0.3), atol=1e-9
    )

    amplitude = 1e-4
    drive = nitrogen_magnetic_drive_operator(
        scenario, records, np.tile([amplitude, 0, 0], (3, 1))
    )
    initial = tensor([basis(3, 1)] * 6)

    for subsystem, gamma in (
        (0, scenario.specification["sites"][0]["gamma_e"]),
        (1, records["sites"][0]["gamma_n"]),
    ):
        states = [basis(3, 1)] * 6
        states[subsystem] = basis(3, 2)
        final = tensor(states)
        np.testing.assert_allclose(
            drive.matrix_element(final.dag(), initial),
            gamma * amplitude / np.sqrt(2),
            atol=1e-12,
        )

    spectra = nitrogen_local_spectra(reference, drive)
    centre = (
        scenario.specification["sites"][0]["D"]
        - scenario.specification["sites"][0]["gamma_e"]
        * scenario.register.magnetic_fields[0, 2]
    )
    lines = [
        line for line in spectra[0]["transitions"]
        if abs(line["frequency_rad_per_us"] - centre) < 1.0
    ]
    strongest = max(line["strength"] for line in lines)
    dominant = sorted(
        line["frequency_rad_per_us"]
        for line in lines if line["strength"] > strongest / 10
    )
    np.testing.assert_allclose(
        dominant,
        centre + np.array([-0.3, 0.0, 0.3]),
        atol=1e-4,
        rtol=0.0,
    )

    calibrated = calibrate_local_nitrogen_square_pulse(reference, drive)
    assert calibrated["initial_state_overlap"] > 0.99
    assert calibrated["target_state_overlap"] > 0.99
    np.testing.assert_allclose(
        calibrated["carrier_rad_per_us"],
        centre - records["sites"][0]["A"][2][2],
        atol=1e-4,
        rtol=0.0,
    )
    propagator = calibrated["propagator"]
    assert (
        propagator.dag() * propagator - qeye([3, 3])
    ).norm("fro") < 1e-10

    model = build_nitrogen_drive_rwa(
        reference, drive, calibrated["carrier_rad_per_us"]
    )
    drift = model["drift_hamiltonian"]
    number = model["excitation_operator"]

    assert drift.shape == (729, 729)
    assert drift.isherm
    assert model["drive_operator"].isherm
    assert model["nuclear_reduction"] is False
    assert (
        drift * number - number * drift
    ).norm("fro") < 1e-8

    transform = model["basis_to_lab"]
    assert (
        transform.dag() * transform - qeye([3] * 6)
    ).norm("fro") < 1e-10


def test_nuclear_preparations_preserve_correlated_electronic_inputs():
    """Preserve both marginal states when interleaving a correlated preparation."""
    electron = (
        tensor([basis(3, 1)] * 3)
        + tensor([basis(3, 2)] * 3)
    ).unit().proj()

    polarised = nitrogen_product_preparation([1, 1, 1])
    unpolarised = nitrogen_product_preparation([None, None, None])

    for nuclear in (polarised, unpolarised):
        joint = electron_nitrogen_preparation(electron, nuclear)
        assert (
            joint.ptrace([0, 2, 4]) - electron
        ).norm("fro") < 1e-12
        assert (
            joint.ptrace([1, 3, 5]) - nuclear
        ).norm("fro") < 1e-12

    np.testing.assert_allclose((polarised * polarised).tr(), 1.0)
    np.testing.assert_allclose((unpolarised * unpolarised).tr(), 1 / 27)


def test_joint_propagation_retains_nuclear_memory():
    """Recover coherence that is lost when the nucleus is artificially reset."""
    _, _, sz = spin_operators(1)
    hamiltonian = tensor(sz, sz)
    electron = (basis(3, 1) + basis(3, 2)).unit()
    nucleus = (basis(3, 0) + basis(3, 2)).unit()
    initial = tensor(electron, nucleus)
    interval = np.pi / 2

    first = joint_exponential_action(
        hamiltonian, initial.full(), interval
    )
    second = joint_exponential_action(
        hamiltonian, first, interval
    )
    continuous = Qobj(second, dims=initial.dims)
    expected = hermitian_propagator(
        hamiltonian, 2 * interval
    ) * initial
    np.testing.assert_allclose(
        continuous.full(), expected.full(), atol=1e-11, rtol=0.0
    )

    electron_after_first = Qobj(
        first, dims=initial.dims
    ).ptrace(0)
    reset = tensor(electron_after_first, nucleus.proj())
    propagator = hermitian_propagator(hamiltonian, interval)
    reset_final = (
        propagator * reset * propagator.dag()
    ).ptrace(0)
    retained_final = continuous.ptrace(0)

    np.testing.assert_allclose(
        (retained_final * retained_final).tr(), 1.0, atol=1e-11
    )
    np.testing.assert_allclose(
        (reset_final * reset_final).tr(), 0.5, atol=1e-11
    )

    model = {
        "drift_hamiltonian": hamiltonian,
        "raising_operator": 0 * hamiltonian,
        "carrier_rad_per_us": 10.0,
    }
    schedule = {
        "duration_us": 2 * interval,
        "events": [
            {"start_us": 0.2, "end_us": 0.4, "tones": []}
        ],
    }
    through_events = propagate_nitrogen_events(
        model, schedule, initial.full(), steps=4
    )
    np.testing.assert_allclose(
        through_events, second, atol=1e-10, rtol=0.0
    )


def test_nuclear_channel_metrics_match_an_analytical_dephasing_channel():
    """Recover the exact polarised and mixed-nucleus electronic fidelities."""
    _, _, sz = spin_operators(1)
    hamiltonian = tensor(sz, sz)
    target = Qobj([[1, 0], [0, -1]])

    for projection, expected in ((1, 1.0), (None, 7 / 9)):
        inputs = nitrogen_logical_input_columns(
            nitrogen_product_preparation([projection])
        )
        evolved = joint_exponential_action(
            hamiltonian, inputs["columns"], np.pi
        )
        result = nitrogen_electronic_gate_metrics(
            evolved, inputs, target
        )
        np.testing.assert_allclose(
            result["average_gate_fidelity"], expected, atol=1e-11
        )
        np.testing.assert_allclose(
            result["total_probability"], 1.0, atol=1e-11
        )
        assert result["average_leakage"] < 1e-20


def test_joint_noise_drift_applies_during_free_and_pulse_windows():
    """Match constant noisy joint evolution across free and driven intervals."""
    sx, _, sz = spin_operators(1)
    drift = tensor(sz, sz)
    perturbation = 0.2 * tensor(sx, qeye(3))
    noisy_drift = drift + perturbation

    raising = tensor(
        basis(3, 0) * basis(3, 1).dag(), qeye(3)
    )
    drive = 0.3 * (raising + raising.dag())
    model = {
        "drift_hamiltonian": drift,
        "raising_operator": raising,
        "carrier_rad_per_us": 10.0,
    }
    schedule = {
        "duration_us": 0.8,
        "events": [{
            "start_us": 0.2,
            "end_us": 0.5,
            "tones": [{
                "carrier_rad_per_us": 10.0,
                "phase_rad": 0.0,
                "envelope_scale": 0.6,
            }],
        }],
    }
    initial = tensor(
        (basis(3, 1) + basis(3, 2)).unit(),
        (basis(3, 0) + basis(3, 2)).unit(),
    ).full()

    expected = joint_exponential_action(noisy_drift, initial, 0.2)
    expected = joint_exponential_action(
        noisy_drift + drive, expected, 0.3
    )
    expected = joint_exponential_action(noisy_drift, expected, 0.3)

    actual = propagate_nitrogen_events(
        model,
        schedule,
        initial,
        shape="square",
        steps=3,
        drift_at_time=lambda time: noisy_drift,
        free_step_us=0.07,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def test_slow_joint_noise_retains_nuclear_but_removes_electronic_transitions():
    """Retain intra-manifold nuclear coupling while discarding electronic flips."""
    sx, _, sz = spin_operators(1)
    model = {
        "basis_to_lab": qeye([3, 3]),
        "excitation_operator": tensor(
            Qobj(np.diag([1, 0, 1])), qeye(3)
        ),
    }
    retained = tensor(sz, qeye(3)) + 0.1 * tensor(qeye(3), sx)
    perturbation = retained + tensor(sx, qeye(3))

    actual = slow_operator_in_nitrogen_rwa(model, perturbation)
    np.testing.assert_allclose(
        actual.full(), retained.full(), atol=1e-12, rtol=0
    )
