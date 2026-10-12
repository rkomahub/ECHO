import numpy as np
import pytest
from qutip import basis

from echo_spin.noise.relaxation import (
    amplitude_damping_channel,
    generalized_amplitude_damping_channel,
    relaxation_dephasing_channel,
)


@pytest.mark.parametrize("time", [0.0, 0.7, 20.0])
def test_amplitude_damping_excited_population(time):
    """Verify exponential excited-state decay and transfer into the ground state."""
    relaxation_time = 2.0

    actual = amplitude_damping_channel(
        density_matrix=basis(2, 1).proj(),
        time=time,
        relaxation_time=relaxation_time,
    )

    survival = np.exp(-time / relaxation_time)

    np.testing.assert_allclose(
        actual.full(),
        np.diag([1 - survival, survival]),
        atol=1e-12,
        rtol=0.0,
    )
    assert actual.tr() == pytest.approx(1.0, abs=1e-12)


def test_relaxation_and_dephasing_coherence_relation():
    """Verify the combined T1 and Tphi population and coherence decay."""
    time = 0.7
    relaxation_time = 2.0
    pure_dephasing_time = 3.0

    plus = (basis(2, 0) + basis(2, 1)).unit()

    actual = relaxation_dephasing_channel(
        density_matrix=plus.proj(),
        time=time,
        relaxation_time=relaxation_time,
        pure_dephasing_time=pure_dephasing_time,
    )

    survival = np.exp(-time / relaxation_time)
    t_two = 1 / (
        1 / (2 * relaxation_time)
        + 1 / pure_dephasing_time
    )
    coherence = np.exp(-time / t_two)

    expected = np.array(
        [
            [1 - survival / 2, coherence / 2],
            [coherence / 2, survival / 2],
        ],
        dtype=complex,
    )

    np.testing.assert_allclose(
        actual.full(),
        expected,
        atol=1e-12,
        rtol=0.0,
    )
    assert t_two <= 2 * relaxation_time
    assert np.min(actual.eigenenergies()) > -1e-12


def test_amplitude_damping_preserves_ground_state():
    """Verify that the zero-temperature relaxation ground state is stationary."""
    ground_density = basis(2, 0).proj()

    actual = amplitude_damping_channel(
        density_matrix=ground_density,
        time=4.0,
        relaxation_time=2.0,
    )

    np.testing.assert_allclose(
        actual.full(),
        ground_density.full(),
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize("initial_excited", [0.0, 1.0])
@pytest.mark.parametrize("equilibrium_excited", [0.0, 0.2, 0.5])
def test_generalized_damping_population_equilibration(
    initial_excited,
    equilibrium_excited,
):
    """Verify excitation and relaxation toward the specified equilibrium population."""
    time = 0.7
    relaxation_time = 2.0

    initial_density = (
        (1 - initial_excited) * basis(2, 0).proj()
        + initial_excited * basis(2, 1).proj()
    )

    actual = generalized_amplitude_damping_channel(
        density_matrix=initial_density,
        time=time,
        relaxation_time=relaxation_time,
        equilibrium_excited_population=equilibrium_excited,
    )

    expected_excited = (
        equilibrium_excited
        + (initial_excited - equilibrium_excited)
        * np.exp(-time / relaxation_time)
    )

    np.testing.assert_allclose(
        actual.full(),
        np.diag([1 - expected_excited, expected_excited]),
        atol=1e-12,
        rtol=0.0,
    )
    assert actual.tr() == pytest.approx(1.0, abs=1e-12)


def test_generalized_damping_stationary_state_and_coherence():
    """Verify equilibrium stationarity and the relaxation-only coherence factor."""
    time = 0.7
    relaxation_time = 2.0
    equilibrium_excited = 0.2

    equilibrium_density = (
        (1 - equilibrium_excited) * basis(2, 0).proj()
        + equilibrium_excited * basis(2, 1).proj()
    )

    stationary = generalized_amplitude_damping_channel(
        density_matrix=equilibrium_density,
        time=time,
        relaxation_time=relaxation_time,
        equilibrium_excited_population=equilibrium_excited,
    )

    np.testing.assert_allclose(
        stationary.full(),
        equilibrium_density.full(),
        atol=1e-12,
        rtol=0.0,
    )

    plus = (basis(2, 0) + basis(2, 1)).unit()

    evolved = generalized_amplitude_damping_channel(
        density_matrix=plus.proj(),
        time=time,
        relaxation_time=relaxation_time,
        equilibrium_excited_population=equilibrium_excited,
    )

    assert evolved.full()[0, 1] == pytest.approx(
        0.5 * np.exp(-time / (2 * relaxation_time)),
        abs=1e-12,
    )
    assert np.min(evolved.eigenenergies()) > -1e-12
