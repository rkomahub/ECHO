import numpy as np
import pytest
from qutip import basis, sigmax, sigmaz

from echo_spin.dynamics.lindblad import static_lindblad_map
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
