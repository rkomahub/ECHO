import numpy as np
import pytest

from echo_spin.hamiltonians import electronic_nv_hamiltonian
from echo_spin.system import SpinSystem


def test_electronic_nv_hamiltonian_dimension():
    """One electronic spin-1 NV Hamiltonian should be 3x3."""
    system = SpinSystem([1])

    hamiltonian = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=1.0,
        omega_e=(0.0, 0.0, 0.0),
    )

    assert hamiltonian.shape == (3, 3)


def test_zero_field_splitting_spectrum():
    """With no magnetic field, the ms = ±1 states are split from ms = 0 by D."""
    system = SpinSystem([1])

    D = 2.87

    hamiltonian = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=D,
        omega_e=(0.0, 0.0, 0.0),
    )

    eigenvalues = np.sort(hamiltonian.eigenenergies())

    assert np.allclose(
        eigenvalues,
        [0.0, D, D],
    )


def test_longitudinal_zeeman_splitting():
    """A longitudinal field should split the ms = ±1 levels."""
    system = SpinSystem([1])

    D = 2.87
    omega_z = 0.1

    hamiltonian = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=D,
        omega_e=(0.0, 0.0, omega_z),
    )

    eigenvalues = np.sort(hamiltonian.eigenenergies())

    expected = np.sort([
        0.0,
        D - omega_z,
        D + omega_z,
    ])

    assert np.allclose(eigenvalues, expected)


def test_nv_hamiltonian_embeds_in_larger_system():
    """The electronic Hamiltonian should act on the full composite space."""
    system = SpinSystem([1, 1])

    hamiltonian = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=1.0,
        omega_e=(0.0, 0.0, 0.0),
    )

    assert hamiltonian.shape == (9, 9)


def test_nv_electron_must_have_spin_one():
    """The electronic NV subsystem must be spin 1."""
    system = SpinSystem([0.5])

    with pytest.raises(ValueError):
        electronic_nv_hamiltonian(
            system=system,
            electron_site=0,
            D=1.0,
            omega_e=(0.0, 0.0, 0.0),
        )


def test_omega_e_requires_three_components():
    """The electronic Larmor vector must contain x, y and z components."""
    system = SpinSystem([1])

    with pytest.raises(ValueError):
        electronic_nv_hamiltonian(
            system=system,
            electron_site=0,
            D=1.0,
            omega_e=(0.0, 0.0),
        )