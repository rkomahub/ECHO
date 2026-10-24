import numpy as np
import pytest
from qutip import qdiags, qeye, sigmaz
from scipy.constants import hbar, mu_0, physical_constants

from echo_spin.control.rotations import single_qubit_rotation
from echo_spin.core.operators import embed_operator
from echo_spin.core.system import SpinSystem
from echo_spin.decoupling.modulation import selective_pair_pulse_times
from echo_spin.models.longitudinal import (
    longitudinal_qubit_parameters,
    longitudinal_register_hamiltonian,
)
from echo_spin.nv.frames import crystallographic_nv_axes
from echo_spin.nv.geometry import lattice_positions
from echo_spin.nv.interactions import register_dipolar_tensors
from echo_spin.nv.registers import NVRegisterConfiguration


@pytest.mark.parametrize("number_of_qubits", [3, 4, 8])
@pytest.mark.parametrize("target_sign", [-1, 1])
@pytest.mark.parametrize("model", ["synthetic", "geometry"])
def test_selective_gate_protects_complete_register(
    number_of_qubits, target_sign, model
):
    """Protect every spectator using synthetic or geometry-derived interactions."""
    system = SpinSystem([0.5] * number_of_qubits)
    constant = 0.0

    if model == "synthetic":
        target_pair = (0, number_of_qubits - 1)
        detunings = np.linspace(-0.7, 0.9, number_of_qubits)
        couplings = np.zeros((number_of_qubits, number_of_qubits))

        for i in range(number_of_qubits):
            for j in range(i + 1, number_of_qubits):
                value = (-1)**(i + j) * (0.03 + 0.01 * (i + j))
                couplings[i, j] = couplings[j, i] = value

        a, b = target_pair
        couplings[a, b] = couplings[b, a] = target_sign * 0.2

    else:
        # Preserve the target pair while adding spectators.
        target_pair = (0, 1)
        axis = crystallographic_nv_axes()[0]
        transverse = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)

        # Kzz is positive perpendicular to the axis and negative parallel to it.
        direction = transverse if target_sign == 1 else axis
        vectors = np.zeros((3, 3))
        vectors[0] = 15e-9 * direction
        positions = lattice_positions((number_of_qubits, 1, 1), vectors)

        register = NVRegisterConfiguration.from_geometry(
            positions=positions,
            nv_axes=np.tile(axis, (number_of_qubits, 1)),
            transverse_references=np.tile(transverse, (number_of_qubits, 1)),
            field_at_reference=0.01 * axis,
        )

        gamma_si = abs(physical_constants["electron gyromag. ratio"][0])
        prefactor = mu_0 * hbar * gamma_si**2 / (4 * np.pi) * 1e-6
        tensors = register_dipolar_tensors(
            register.positions, register.local_frames, prefactor
        )

        # Identical aligned NVs with a common resonant rotating frame.
        detunings, couplings, constant = longitudinal_qubit_parameters(
            np.zeros(number_of_qubits), tensors
        )

    a, b = target_pair
    assert np.sign(couplings[a, b]) == target_sign
    duration = np.pi / (4 * abs(couplings[a, b]))

    hamiltonian = longitudinal_register_hamiltonian(
        detunings, couplings, constant
    )

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
