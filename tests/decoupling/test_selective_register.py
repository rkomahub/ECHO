import numpy as np
import pytest
from qutip import qdiags, qeye, sigmaz

from echo_spin.control.rotations import single_qubit_rotation
from echo_spin.core.operators import embed_operator
from echo_spin.core.system import SpinSystem
from echo_spin.decoupling.modulation import selective_pair_pulse_times
from echo_spin.models.longitudinal import longitudinal_register_hamiltonian


@pytest.mark.parametrize("number_of_qubits", [3, 4, 8])
@pytest.mark.parametrize("target_sign", [-1, 1])
def test_selective_gate_protects_complete_register(
    number_of_qubits, target_sign
):
    """Implement the selected pair gate on the full register using closed pulses."""
    system = SpinSystem([0.5] * number_of_qubits)
    target_pair = (0, number_of_qubits - 1)
    detunings = np.linspace(-0.7, 0.9, number_of_qubits)
    couplings = np.zeros((number_of_qubits, number_of_qubits))

    for i in range(number_of_qubits):
        for j in range(i + 1, number_of_qubits):
            value = (-1)**(i + j) * (0.03 + 0.01 * (i + j))
            couplings[i, j] = couplings[j, i] = value

    a, b = target_pair
    couplings[a, b] = couplings[b, a] = target_sign * 0.2
    duration = np.pi / (4 * abs(couplings[a, b]))

    hamiltonian = longitudinal_register_hamiltonian(detunings, couplings)
    schedules = selective_pair_pulse_times(
        number_of_qubits, target_pair, duration
    )
    event_times = np.unique(np.concatenate(schedules + [np.array([duration])]))

    x_pi = single_qubit_rotation(np.pi, "x")
    pulses = [
        embed_operator(x_pi, site, system)
        for site in range(number_of_qubits)
    ]

    actual = qeye(list(system.dimensions))
    previous_time = 0.0

    for time in event_times:
        # Exact diagonal free evolution avoids a general matrix exponential.
        free = qdiags(
            np.exp(-1j * hamiltonian.diag() * (time - previous_time)),
            0,
            dims=hamiltonian.dims,
        )
        actual = free * actual

        for site, pulse_times in enumerate(schedules):
            if np.any(pulse_times == time):
                actual = pulses[site] * actual

        previous_time = time

    za = embed_operator(sigmaz(), a, system)
    zb = embed_operator(sigmaz(), b, system)
    expected = qdiags(
        np.exp(-1j * couplings[a, b] * duration * (za * zb).diag()),
        0,
        dims=hamiltonian.dims,
    )

    # Compare lab-frame operators, allowing only a common global phase.
    relative_phase = np.angle(actual[0, 0] / expected[0, 0])
    corrected = np.exp(-1j * relative_phase) * actual.full()

    np.testing.assert_allclose(
        corrected, expected.full(), atol=1e-10, rtol=0.0
    )
