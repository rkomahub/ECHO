import numpy as np
import pytest
from qutip import basis

from echo_spin.noise.relaxation import (
    amplitude_damping_channel,
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
