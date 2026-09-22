import numpy as np
import pytest

from qutip import Qobj, basis, jmat

from echo_spin.basis import (
    basis_unitary,
    diagonalize_hamiltonian,
    dressed_spin_operators,
)


def test_diagonalization_orders_eigenstates_by_energy():
    """Hamiltonian eigenstates should be returned in increasing-energy order."""
    hamiltonian = Qobj(
        np.diag([2.0, 0.0, 1.0])
    )

    eigenvalues, _ = diagonalize_hamiltonian(hamiltonian)

    assert np.allclose(
        eigenvalues,
        [0.0, 1.0, 2.0],
    )


def test_basis_unitary_is_unitary():
    """An orthonormal basis should produce a unitary basis transformation."""
    states = [
        basis(3, 2),
        basis(3, 1),
        basis(3, 0),
    ]

    unitary = basis_unitary(states)

    assert np.allclose(
        (unitary.dag() * unitary).full(),
        np.eye(3),
    )


def test_dressed_operators_have_standard_form_in_dressed_basis():
    """Dressed operators should recover standard spin matrices in their basis."""
    states = [
        basis(3, 2),
        basis(3, 1),
        basis(3, 0),
    ]

    unitary = basis_unitary(states)

    sx_tilde, sy_tilde, sz_tilde = dressed_spin_operators(
        basis_states=states,
        spin=1,
    )

    for dressed, standard in zip(
        (sx_tilde, sy_tilde, sz_tilde),
        (
            jmat(1, "x"),
            jmat(1, "y"),
            jmat(1, "z"),
        ),
    ):
        transformed = (
            unitary.dag()
            * dressed
            * unitary
        )

        assert np.allclose(
            transformed.full(),
            standard.full(),
        )


def test_dressed_spin_commutation_relation():
    """Dressed spin operators should satisfy the angular-momentum algebra."""
    states = [
        basis(3, 2),
        basis(3, 1),
        basis(3, 0),
    ]

    sx, sy, sz = dressed_spin_operators(
        basis_states=states,
        spin=1,
    )

    commutator = sx * sy - sy * sx

    assert np.allclose(
        commutator.full(),
        (1j * sz).full(),
    )


def test_nonorthogonal_basis_raises_error():
    """Dressed operators require a complete orthonormal basis."""
    states = [
        basis(3, 0),
        basis(3, 0),
        basis(3, 2),
    ]

    with pytest.raises(ValueError):
        basis_unitary(states)