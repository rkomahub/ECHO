import numpy as np
import pytest
from qutip import basis

from echo_spin.dynamics.lindblad import static_lindblad_map
from echo_spin.noise.relaxation import (
    spin_one_dephasing_operator,
    spin_one_relaxation_operators,
)


def test_spin_one_independent_relaxation_paths():
    """Verify both population-decay paths and their shared coherence damping."""
    time = 1.3
    rates = np.array([0.4, 0.7])

    state = (
        basis(3, 0) + 1j * basis(3, 2)
    ).unit()

    actual = static_lindblad_map(
        density_matrix=state.proj(),
        hamiltonian=0 * basis(3, 1).proj(),
        time=time,
        collapse_operators=spin_one_relaxation_operators(rates),
    )

    survival_plus, survival_minus = np.exp(-rates * time)

    expected = np.diag([
        survival_plus / 2,
        1 - (survival_plus + survival_minus) / 2,
        survival_minus / 2,
    ]).astype(complex)

    coherence = -0.5j * np.exp(
        -np.sum(rates) * time / 2
    )
    expected[0, 2] = coherence
    expected[2, 0] = np.conjugate(coherence)

    np.testing.assert_allclose(
        actual.full(),
        expected,
        atol=1e-12,
        rtol=0.0,
    )
    assert actual.tr() == pytest.approx(1.0, abs=1e-12)
    assert np.min(actual.eigenenergies()) > -1e-12


def test_spin_one_dephasing_projection_dependence():
    """Verify single- and double-quantum coherence decay under longitudinal dephasing."""
    time = 0.7
    coherence_time = 2.0

    state = (
        basis(3, 0) + basis(3, 1) + basis(3, 2)
    ).unit()

    actual = static_lindblad_map(
        density_matrix=state.proj(),
        hamiltonian=0 * basis(3, 1).proj(),
        time=time,
        collapse_operators=[
            spin_one_dephasing_operator(coherence_time)
        ],
    )

    projections = np.array([1, 0, -1])
    differences = (
        projections[:, None] - projections[None, :]
    )

    expected = np.exp(
        -differences**2 * time / coherence_time
    ) / 3

    np.testing.assert_allclose(
        actual.full(),
        expected,
        atol=1e-12,
        rtol=0.0,
    )
    assert actual.tr() == pytest.approx(1.0, abs=1e-12)
