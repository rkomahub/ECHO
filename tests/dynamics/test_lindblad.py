import numpy as np
import pytest
from qutip import basis, sigmax, sigmaz

from echo_spin.dynamics.lindblad import piecewise_lindblad_map, static_lindblad_map
from echo_spin.dynamics.propagators import static_propagator
from echo_spin.noise.dephasing import dephasing_channel
from echo_spin.noise.relaxation import (
    generalized_amplitude_damping_channel,
)


@pytest.mark.parametrize("equilibrium_excited", [0.0, 0.2, 0.5])
def test_lindblad_matches_relaxation_and_dephasing_channels(
    equilibrium_excited,
):
    """Verify Lindblad rates against analytical excitation and dephasing channels."""
    time = 0.7
    relaxation_time = 2.0
    pure_dephasing_time = 3.0

    ground = basis(2, 0)
    excited = basis(2, 1)

    state = (ground + (0.4 + 0.7j) * excited).unit()
    initial_density = 0.73 * state.proj()

    lowering = ground * excited.dag()

    collapse_operators = [
        np.sqrt(
            (1 - equilibrium_excited) / relaxation_time
        ) * lowering,
        np.sqrt(
            equilibrium_excited / relaxation_time
        ) * lowering.dag(),
        np.sqrt(
            1 / (2 * pure_dephasing_time)
        ) * sigmaz(),
    ]

    actual = static_lindblad_map(
        density_matrix=initial_density,
        hamiltonian=0 * sigmaz(),
        time=time,
        collapse_operators=collapse_operators,
    )

    relaxed = generalized_amplitude_damping_channel(
        density_matrix=initial_density,
        time=time,
        relaxation_time=relaxation_time,
        equilibrium_excited_population=equilibrium_excited,
    )
    expected = dephasing_channel(
        density_matrix=relaxed,
        time=time,
        coherence_time=pure_dephasing_time,
    )

    np.testing.assert_allclose(
        actual.full(),
        expected.full(),
        atol=1e-12,
        rtol=0.0,
    )
    assert actual.tr() == pytest.approx(0.73, abs=1e-12)
    assert np.min(actual.eigenenergies()) > -1e-12


def test_lindblad_without_collapse_recovers_unitary_evolution():
    """Verify that the collapse-free generator reproduces coherent evolution."""
    time = 0.7
    hamiltonian = 0.4 * sigmax() + 0.2 * sigmaz()
    initial_density = basis(2, 0).proj()

    actual = static_lindblad_map(
        density_matrix=initial_density,
        hamiltonian=hamiltonian,
        time=time,
        collapse_operators=[],
    )

    propagator = static_propagator(
        hamiltonian=hamiltonian,
        time=time,
    )
    expected = (
        propagator
        * initial_density
        * propagator.dag()
    )

    np.testing.assert_allclose(
        actual.full(),
        expected.full(),
        atol=1e-12,
        rtol=0.0,
    )


def test_driven_relaxation_matches_damped_rabi_solution():
    """Verify simultaneous resonant control and relaxation against damped Rabi dynamics."""
    rabi_frequency = 1.0
    decay_rate = 0.4
    total_time = 1.3

    ground = basis(2, 0)
    excited = basis(2, 1)

    hamiltonian = rabi_frequency * sigmax() / 2
    lowering = ground * excited.dag()

    actual = piecewise_lindblad_map(
        density_matrix=ground.proj(),
        hamiltonians=[hamiltonian] * 3,
        durations=[0.2, 0.4, 0.7],
        collapse_operators=[
            np.sqrt(decay_rate) * lowering
        ],
    )

    # Exact underdamped optical-Bloch solution, starting in the ground state.
    damped_frequency = np.sqrt(
        rabi_frequency**2 - decay_rate**2 / 16
    )
    stationary_z = (
        decay_rate**2
        / (decay_rate**2 + 2 * rabi_frequency**2)
    )

    expected_z = (
        stationary_z
        + (1 - stationary_z)
        * np.exp(-3 * decay_rate * total_time / 4)
        * (
            np.cos(damped_frequency * total_time)
            + 3 * decay_rate / (4 * damped_frequency)
            * np.sin(damped_frequency * total_time)
        )
    )

    expected_excited = (1 - expected_z) / 2
    actual_excited = float(np.real(
        excited.dag() * actual * excited
    ))

    assert actual_excited == pytest.approx(
        expected_excited,
        abs=1e-12,
    )
    assert actual.tr() == pytest.approx(1.0, abs=1e-12)

    # Applying all relaxation afterward gives a different population.
    relaxation_after_drive = (
        np.sin(rabi_frequency * total_time / 2) ** 2
        * np.exp(-decay_rate * total_time)
    )

    assert abs(
        actual_excited - relaxation_after_drive
    ) > 1e-3


def test_piecewise_lindblad_preserves_control_order():
    """Verify chronological ordering of noncommuting coherent control intervals."""
    initial_density = basis(2, 0).proj()
    first_hamiltonian = 0.7 * sigmax()
    second_hamiltonian = 0.4 * sigmaz()

    actual = piecewise_lindblad_map(
        density_matrix=initial_density,
        hamiltonians=[first_hamiltonian, second_hamiltonian],
        durations=[0.3, 0.5],
        collapse_operators=[],
    )

    first = static_propagator(first_hamiltonian, 0.3)
    second = static_propagator(second_hamiltonian, 0.5)
    propagator = second * first

    expected = (
        propagator * initial_density * propagator.dag()
    )

    np.testing.assert_allclose(
        actual.full(),
        expected.full(),
        atol=1e-12,
        rtol=0.0,
    )
