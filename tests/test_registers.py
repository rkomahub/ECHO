import numpy as np
import pytest

from echo_spin.hamiltonians import nv_hamiltonian
from echo_spin.registers import nv_register_hamiltonian
from echo_spin.system import SpinSystem


def test_two_nv_register_dimension():
    """Two complete NV centers should have Hilbert-space dimension 81."""
    system = SpinSystem([1, 1, 1, 1])

    parameters = [
        {
            "D": 2.87,
            "omega_e": (0.0, 0.0, 0.1),
            "Q": -4.95,
            "omega_n": (0.0, 0.0, 0.01),
            "A": np.eye(3),
        },
        {
            "D": 2.87,
            "omega_e": (0.0, 0.0, 0.1),
            "Q": -4.95,
            "omega_n": (0.0, 0.0, 0.01),
            "A": np.eye(3),
        },
    ]

    hamiltonian = nv_register_hamiltonian(
        system=system,
        nv_parameters=parameters,
    )

    assert hamiltonian.shape == (81, 81)

def test_nv_register_equals_sum_of_single_nv_hamiltonians():
    """The register Hamiltonian should equal the sum of its NV Hamiltonians."""
    system = SpinSystem([1, 1, 1, 1])

    parameters = [
        {
            "D": 2.87,
            "omega_e": (0.0, 0.0, 0.1),
            "Q": -4.95,
            "omega_n": (0.0, 0.0, 0.01),
            "A": np.eye(3),
        },
        {
            "D": 2.90,
            "omega_e": (0.0, 0.0, 0.2),
            "Q": -4.90,
            "omega_n": (0.0, 0.0, 0.02),
            "A": 2.0 * np.eye(3),
        },
    ]

    register = nv_register_hamiltonian(
        system=system,
        nv_parameters=parameters,
    )

    expected = (
        nv_hamiltonian(
            system=system,
            electron_site=0,
            nuclear_site=1,
            **parameters[0],
        )
        + nv_hamiltonian(
            system=system,
            electron_site=2,
            nuclear_site=3,
            **parameters[1],
        )
    )

    assert np.allclose(
        register.full(),
        expected.full(),
    )

def test_nv_register_requires_two_subsystems_per_nv():
    """Each NV must contain one electron and one nuclear subsystem."""
    system = SpinSystem([1, 1, 1])

    parameters = [
        {
            "D": 2.87,
            "omega_e": (0.0, 0.0, 0.1),
            "Q": -4.95,
            "omega_n": (0.0, 0.0, 0.01),
            "A": np.eye(3),
        }
    ]

    with pytest.raises(ValueError):
        nv_register_hamiltonian(
            system=system,
            nv_parameters=parameters,
        )