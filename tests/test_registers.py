import numpy as np
import pytest

from echo_spin.hamiltonians import nv_hamiltonian
from echo_spin.interactions import dipolar_interaction
from echo_spin.registers import (
    nv_dipolar_interaction,
    nv_register_hamiltonian,
)
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

def test_two_nv_register_has_one_dipolar_pair():
    """Two NV centers should generate exactly one electronic dipolar pair."""
    system = SpinSystem([1, 1, 1, 1])

    couplings = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    directions = np.zeros((2, 2, 3))
    directions[0, 1] = [0.0, 0.0, 1.0]

    interaction = nv_dipolar_interaction(
        system=system,
        couplings=couplings,
        directions=directions,
    )

    expected = dipolar_interaction(
        system=system,
        site_i=0,
        site_j=2,
        coupling=1.0,
        direction=(0.0, 0.0, 1.0),
    )

    assert np.allclose(
        interaction.full(),
        expected.full(),
    )

def test_three_nv_register_has_three_dipolar_pairs():
    """Three NV centers should generate the three unique electronic pairs."""
    system = SpinSystem([1, 1, 1, 1, 1, 1])

    couplings = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 3.0],
        [2.0, 3.0, 0.0],
    ])

    directions = np.zeros((3, 3, 3))
    directions[0, 1] = [0.0, 0.0, 1.0]
    directions[0, 2] = [1.0, 0.0, 0.0]
    directions[1, 2] = [0.0, 1.0, 0.0]

    interaction = nv_dipolar_interaction(
        system=system,
        couplings=couplings,
        directions=directions,
    )

    expected = (
        dipolar_interaction(system, 0, 2, 1.0, (0.0, 0.0, 1.0))
        + dipolar_interaction(system, 0, 4, 2.0, (1.0, 0.0, 0.0))
        + dipolar_interaction(system, 2, 4, 3.0, (0.0, 1.0, 0.0))
    )

    assert np.allclose(
        interaction.full(),
        expected.full(),
    )

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

def test_zero_dipolar_couplings_leave_register_unchanged():
    """Zero dipolar couplings should reproduce the non-interacting register."""
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

    couplings = np.zeros((2, 2))

    directions = np.zeros((2, 2, 3))
    directions[0, 1] = [0.0, 0.0, 1.0]

    non_interacting = nv_register_hamiltonian(
        system=system,
        nv_parameters=parameters,
    )

    interacting = nv_register_hamiltonian(
        system=system,
        nv_parameters=parameters,
        dipolar_couplings=couplings,
        dipolar_directions=directions,
    )

    assert np.allclose(
        non_interacting.full(),
        interacting.full(),
    )

def test_invalid_dipolar_coupling_shape_raises_error():
    """The coupling matrix must match the number of NV centers."""
    system = SpinSystem([1, 1, 1, 1])

    with pytest.raises(ValueError):
        nv_dipolar_interaction(
            system=system,
            couplings=np.zeros((3, 3)),
            directions=np.zeros((2, 2, 3)),
        )

def test_couplings_without_directions_raise_error():
    """Dipolar couplings and directions must be provided together."""
    system = SpinSystem([1, 1])

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
            dipolar_couplings=np.zeros((1, 1)),
        )

def test_register_can_build_dipolar_interaction_from_positions():
    """NV positions should generate the same interaction as explicit geometry."""
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

    positions = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
    ]

    automatic = nv_register_hamiltonian(
        system=system,
        nv_parameters=parameters,
        positions=positions,
        coupling_prefactor=8.0,
    )

    couplings = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    directions = np.zeros((2, 2, 3))
    directions[0, 1] = [0.0, 0.0, 1.0]
    directions[1, 0] = [0.0, 0.0, -1.0]

    explicit = nv_register_hamiltonian(
        system=system,
        nv_parameters=parameters,
        dipolar_couplings=couplings,
        dipolar_directions=directions,
    )

    assert np.allclose(
        automatic.full(),
        explicit.full(),
    )

def test_positions_require_coupling_prefactor():
    """Position-based interactions require a coupling prefactor."""
    system = SpinSystem([1, 1])

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
            positions=[[0.0, 0.0, 0.0]],
        )

def test_positions_and_explicit_geometry_cannot_be_mixed():
    """Automatic and explicit dipolar geometry should be mutually exclusive."""
    system = SpinSystem([1, 1])

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
            positions=[[0.0, 0.0, 0.0]],
            coupling_prefactor=1.0,
            dipolar_couplings=np.zeros((1, 1)),
            dipolar_directions=np.zeros((1, 1, 3)),
        )