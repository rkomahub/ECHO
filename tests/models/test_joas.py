import numpy as np

import pytest
from qutip import qeye

from echo_spin.control.rotations import two_qubit_rotation
from echo_spin.dynamics.sequences import (
    toggling_hamiltonians,
    toggling_propagator,
)
from echo_spin.models.joas import (
    driven_hamiltonian,
    free_hamiltonian,
    reduced_free_hamiltonian,
    two_pulse_gate_durations
)
from echo_spin.gates.gates import sqrt_zz_gate


def test_free_hamiltonian_sums_nv_terms_and_interaction():
    """The free Hamiltonian should sum all NV and interaction terms."""
    h_nv_1 = 1.0 * qeye(3)
    h_nv_2 = 2.0 * qeye(3)
    interaction = 0.5 * qeye(3)

    h_free = free_hamiltonian(
        nv_hamiltonians=[h_nv_1, h_nv_2],
        interaction=interaction,
    )

    expected = 3.5 * qeye(3)

    assert np.allclose(
        h_free.full(),
        expected.full(),
    )


def test_free_hamiltonian_requires_nv_system():
    """The free Hamiltonian requires at least one NV contribution."""
    import pytest

    with pytest.raises(ValueError):
        free_hamiltonian(
            nv_hamiltonians=[],
            interaction=qeye(3),
        )


def test_driven_hamiltonian_adds_microwave_term():
    """The driven Hamiltonian should equal H_free + H_mw at each time."""
    free = 2.0 * qeye(2)
    control = qeye(2)

    drives = [
        {
            "omega": 1.0,
            "phase": 0.0,
            "omega_x": 1.0,
            "omega_y": 0.0,
        }
    ]

    hamiltonian = driven_hamiltonian(
        time=0.0,
        free=free,
        control=control,
        drives=drives,
    )

    expected = (
        2.0 + np.sqrt(2.0)
    ) * qeye(2)

    assert np.allclose(
        hamiltonian.full(),
        expected.full(),
    )


def test_joas_two_pulse_toggling_hamiltonians():
    """The toggling Hamiltonians for the Joas two-pulse sequence. Periods 1 -- 5"""
    delta_1 = 0.31
    delta_2 = -0.17
    g = 0.08

    free = reduced_free_hamiltonian(
        delta_1=delta_1,
        delta_2=delta_2,
        coupling=g,
    )

    x1 = two_qubit_rotation(
    qubit=1,
    angle=np.pi,
    axis="x",
    )

    x2 = two_qubit_rotation(
        qubit=0,
        angle=np.pi,
        axis="x",
    )

    hamiltonians = toggling_hamiltonians(
        free_hamiltonian=free,
        pulses=[x1, x2, x1, x2],
    )

    expected = [
        np.diag([
            delta_1,
            0,
            delta_1 + delta_2 - g,
            delta_2,
        ]),
        np.diag([
            0,
            delta_1,
            delta_2,
            delta_1 + delta_2 - g,
        ]),
        np.diag([
            delta_2,
            delta_1 + delta_2 - g,
            0,
            delta_1,
        ]),
        np.diag([
            delta_1 + delta_2 - g,
            delta_2,
            delta_1,
            0,
        ]),
        np.diag([
            delta_1,
            0,
            delta_1 + delta_2 - g,
            delta_2,
        ]),
    ]

    assert len(hamiltonians) == 5

    for actual, target in zip(hamiltonians, expected):
        assert np.allclose(actual.full(), target)


def test_two_pulse_gate_durations():
    tau_1 = 0.8
    tau_2 = 0.2

    durations = two_pulse_gate_durations(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    expected = [
        0.4,
        0.2,
        0.6,
        0.2,
        0.2,
    ]

    assert np.allclose(durations, expected)


def test_two_pulse_gate_durations_rejects_large_tau_2():
    with pytest.raises(ValueError):
        two_pulse_gate_durations(
            tau_1=0.8,
            tau_2=0.5,
        )


def test_joas_two_pulse_gate_refocuses_detunings():
    """The Joas sequence refocuses detunings while retaining the ZZ phase."""
    delta_1 = 0.31
    delta_2 = -0.17
    g = 0.08

    tau_1 = 1.0
    tau_2 = 0.2

    free = reduced_free_hamiltonian(
        delta_1=delta_1,
        delta_2=delta_2,
        coupling=g,
    )

    # Joas reduced-basis convention:
    # NV1 -> ECHO qubit 1
    # NV2 -> ECHO qubit 0
    x1 = two_qubit_rotation(1, np.pi, "x")
    x2 = two_qubit_rotation(0, np.pi, "x")

    hamiltonians = toggling_hamiltonians(
        free_hamiltonian=free,
        pulses=[x1, x2, x1, x2],
    )

    durations = two_pulse_gate_durations(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    propagator = toggling_propagator(
        hamiltonians=hamiltonians,
        durations=durations,
    )

    # Remove the physically irrelevant global phase.
    matrix = propagator.full()
    matrix = matrix * np.exp(-1j * np.angle(matrix[0, 0]))

    expected_phase = 2 * g * tau_2

    expected = np.diag([
        1.0,
        np.exp(1j * expected_phase),
        np.exp(1j * expected_phase),
        1.0,
    ])

    assert np.allclose(matrix, expected)


def test_joas_two_pulse_gate_is_independent_of_detunings():
    """Different static detunings give the same gate up to global phase."""
    g = 0.08
    tau_1 = 1.0
    tau_2 = 0.2

    x1 = two_qubit_rotation(1, np.pi, "x")
    x2 = two_qubit_rotation(0, np.pi, "x")

    durations = two_pulse_gate_durations(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    def gate(delta_1, delta_2):
        free = reduced_free_hamiltonian(
            delta_1=delta_1,
            delta_2=delta_2,
            coupling=g,
        )

        hamiltonians = toggling_hamiltonians(
            free_hamiltonian=free,
            pulses=[x1, x2, x1, x2],
        )

        propagator = toggling_propagator(
            hamiltonians=hamiltonians,
            durations=durations,
        )

        matrix = propagator.full()

        # Remove global phase relative to the first diagonal element.
        return matrix * np.exp(-1j * np.angle(matrix[0, 0]))

    gate_a = gate(
        delta_1=0.31,
        delta_2=-0.17,
    )

    gate_b = gate(
        delta_1=2.7,
        delta_2=-1.4,
    )

    assert np.allclose(gate_a, gate_b)


def test_joas_two_pulse_sequence_generates_sqrt_zz():
    """The ideal Joas sequence generates sqrt(ZZ) up to global phase."""
    delta_1 = 0.31
    delta_2 = -0.17
    g = 0.08

    # For N_pi = 2, require 2 * g * tau_2 = pi / 2.
    tau_2 = np.pi / (4 * g)

    # tau_1 must satisfy tau_2 <= tau_1 / 2.
    tau_1 = 2.5 * tau_2

    free = reduced_free_hamiltonian(
        delta_1=delta_1,
        delta_2=delta_2,
        coupling=g,
    )

    # Joas reduced-basis convention:
    # NV1 -> ECHO qubit 1
    # NV2 -> ECHO qubit 0
    x1 = two_qubit_rotation(1, np.pi, "x")
    x2 = two_qubit_rotation(0, np.pi, "x")

    hamiltonians = toggling_hamiltonians(
        free_hamiltonian=free,
        pulses=[x1, x2, x1, x2],
    )

    durations = two_pulse_gate_durations(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    propagator = toggling_propagator(
        hamiltonians=hamiltonians,
        durations=durations,
    )

    # Remove global phase.
    matrix = propagator.full()
    matrix *= np.exp(-1j * np.angle(matrix[0, 0]))

    expected = sqrt_zz_gate().full()

    assert np.allclose(matrix, expected)


def test_sqrt_zz_gate_time_scales_inverse_with_coupling():
    """The sqrt(ZZ) interaction time scales as 1/g."""
    g_1 = 0.08
    g_2 = 2 * g_1

    tau_2_1 = np.pi / (4 * g_1)
    tau_2_2 = np.pi / (4 * g_2)

    assert np.isclose(tau_2_2, tau_2_1 / 2)