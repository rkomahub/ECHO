import numpy as np
import pytest
from qutip import qeye, sigmax, sigmaz, tensor

from echo_spin.control.rotations import single_qubit_rotation
from echo_spin.dynamics.sequences import (
    diagonal_x_pulse_propagator,
    finite_pulse_sequence_propagator,
)


def test_diagonal_pulses_without_control_match_free_evolution():
    """Recover exact free evolution when every pulse schedule is empty."""
    hamiltonian = 0.3 * tensor(sigmaz(), qeye(2))
    actual = diagonal_x_pulse_propagator(hamiltonian, [[], []], 1.7)
    expected = (-1j * hamiltonian * 1.7).expm()

    np.testing.assert_allclose(
        actual.full(), expected.full(), atol=1e-12, rtol=0.0
    )


def test_diagonal_pulses_match_explicit_sequence_with_boundary_pulses():
    """Preserve chronological lab-frame evolution including pulses at zero and T."""
    hamiltonian = (
        0.3 * tensor(sigmaz(), qeye(2))
        - 0.2 * tensor(qeye(2), sigmaz())
        + 0.4 * tensor(sigmaz(), sigmaz())
    )
    x_pi = single_qubit_rotation(np.pi, "x")
    pulse0 = tensor(x_pi, qeye(2))
    pulse1 = tensor(qeye(2), x_pi)

    actual = diagonal_x_pulse_propagator(
        hamiltonian, [[0.0, 0.3], [0.7, 1.0]], 1.0
    )
    expected = finite_pulse_sequence_propagator(
        free_hamiltonian=hamiltonian,
        durations=[0.0, 0.3, 0.4, 0.3, 0.0],
        pulse_propagators=[pulse0, pulse0, pulse1, pulse1],
    )

    np.testing.assert_allclose(
        actual.full(), expected.full(), atol=1e-12, rtol=0.0
    )


@pytest.mark.parametrize(
    "hamiltonian, schedules",
    [
        (sigmax(), [[]]),
        (sigmaz(), []),
        (sigmaz(), [[1.1]]),
        (sigmaz(), [[0.5, 0.5]]),
    ],
)
def test_diagonal_pulses_reject_incompatible_inputs(hamiltonian, schedules):
    """Reject transverse dynamics and invalid pulse schedules."""
    with pytest.raises(ValueError):
        diagonal_x_pulse_propagator(hamiltonian, schedules, 1.0)
