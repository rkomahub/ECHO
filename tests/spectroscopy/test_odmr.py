import numpy as np
import pytest

from echo_spin.core.operators import spin_operators
from echo_spin.core.system import SpinSystem
from echo_spin.nv.hamiltonians import electronic_nv_hamiltonian
from echo_spin.spectroscopy.odmr import lorentzian_spectrum
from echo_spin.spectroscopy.transitions import allowed_transitions


def test_single_lorentzian_peak_and_half_width():
    """Test single Lorentzian peak and half width."""
    spectrum = lorentzian_spectrum(
        frequencies=[9.8, 10.0, 10.2],
        transitions=[(10.0, 2.0, 0, 1)],
        linewidth=0.2,
    )

    assert np.allclose(spectrum, [1.0, 2.0, 1.0])


def test_empty_transition_list_gives_zero_spectrum():
    """Test empty transition list gives zero spectrum."""
    spectrum = lorentzian_spectrum(
        frequencies=[1.0, 2.0, 3.0],
        transitions=[],
        linewidth=0.1,
    )

    assert np.array_equal(spectrum, [0.0, 0.0, 0.0])


def test_aligned_nv_spectrum_from_allowed_transitions():
    """Connect actual NV selection rules to broadened spectral lines."""
    hamiltonian = electronic_nv_hamiltonian(
        system=SpinSystem([1]),
        electron_site=0,
        D=2.87,
        omega_e=(0.0, 0.0, 0.3),
    )
    sx, _, sz = spin_operators(1)
    centers = np.array([2.57, 3.17])
    linewidth = 0.03

    transverse = lorentzian_spectrum(
        frequencies=centers,
        transitions=allowed_transitions(hamiltonian, sx),
        linewidth=linewidth,
    )

    # Each peak also receives the tail of the other line.
    expected = 0.5 + 0.5 / (1 + (0.6 / linewidth)**2)
    assert np.allclose(transverse, [expected, expected])

    longitudinal = lorentzian_spectrum(
        frequencies=centers,
        transitions=allowed_transitions(hamiltonian, sz),
        linewidth=linewidth,
    )
    assert np.array_equal(longitudinal, [0.0, 0.0])


@pytest.mark.parametrize(
    "frequencies, transitions, linewidth",
    [
        ([1.0], [], 0.0),
        ([1.0], [], np.inf),
        ([np.nan], [], 0.1),
        ([[1.0]], [], 0.1),
        ([1.0], [(1.0, -0.5, 0, 1)], 0.1),
        ([1.0], [(np.nan, 0.5, 0, 1)], 0.1),
    ],
)
def test_lorentzian_spectrum_rejects_invalid_parameters(
    frequencies, transitions, linewidth
):
    """Test Lorentzian spectrum rejects invalide parameters."""
    with pytest.raises(ValueError):
        lorentzian_spectrum(frequencies, transitions, linewidth)
