import numpy as np
import pytest

from qutip import Qobj, basis, tensor, qeye

from echo_spin.basis import operator_in_basis
from echo_spin.interactions import (
    dipolar_geometry,
    dipolar_interaction,
    secular_interaction,
    extract_effective_zz_coupling,
    relative_operator_residual,
)
from echo_spin.system import SpinSystem
from echo_spin.operators import embed_operator, spin_operators

def test_dipolar_interaction_dimension():
    """The dipolar interaction should act on the full composite Hilbert space."""
    system = SpinSystem([1, 1])

    interaction = dipolar_interaction(
        system=system,
        site_i=0,
        site_j=1,
        coupling=1.0,
        direction=(0.0, 0.0, 1.0),
    )

    assert interaction.shape == (9, 9)


def test_zero_coupling_gives_zero_operator():
    """A zero dipolar coupling should produce the zero operator."""
    system = SpinSystem([1, 1])

    interaction = dipolar_interaction(
        system=system,
        site_i=0,
        site_j=1,
        coupling=0.0,
        direction=(0.0, 0.0, 1.0),
    )

    assert np.allclose(
        interaction.full(),
        np.zeros((9, 9)),
    )


def test_direction_is_normalized_internally():
    """The dipolar interaction should be independent of the direction's magnitude."""
    system = SpinSystem([1, 1])

    interaction_1 = dipolar_interaction(
        system,
        0,
        1,
        1.0,
        (0.0, 0.0, 1.0),
    )

    interaction_2 = dipolar_interaction(
        system,
        0,
        1,
        1.0,
        (0.0, 0.0, 5.0),
    )

    assert np.allclose(
        interaction_1.full(),
        interaction_2.full(),
    )


def test_same_site_raises_error():
    """A spin cannot interact dipolarly with itself."""
    system = SpinSystem([1])

    with pytest.raises(ValueError):
        dipolar_interaction(
            system,
            0,
            0,
            1.0,
            (0.0, 0.0, 1.0),
        )


def test_zero_direction_raises_error():
    """The direction vector must be non-zero so that a unit vector can be defined."""
    system = SpinSystem([1, 1])

    with pytest.raises(ValueError):
        dipolar_interaction(
            system,
            0,
            1,
            1.0,
            (0.0, 0.0, 0.0),
        )


def test_dipolar_geometry_two_spins():
    """Two positions should generate the expected distance-dependent coupling."""
    positions = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
    ]

    couplings, directions = dipolar_geometry(
        positions=positions,
        coupling_prefactor=8.0,
    )

    assert np.isclose(couplings[0, 1], 1.0)

    assert np.allclose(
        directions[0, 1],
        [0.0, 0.0, 1.0],
    )


def test_dipolar_geometry_is_symmetric():
    """Couplings should be symmetric and pair directions antisymmetric."""
    positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    couplings, directions = dipolar_geometry(
        positions,
        coupling_prefactor=1.0,
    )

    assert np.isclose(
        couplings[0, 1],
        couplings[1, 0],
    )

    assert np.allclose(
        directions[0, 1],
        -directions[1, 0],
    )


def test_dipolar_geometry_inverse_cube_scaling():
    """Doubling the separation should reduce the coupling by a factor of eight."""
    positions_1 = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    positions_2 = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]

    couplings_1, _ = dipolar_geometry(positions_1, 1.0)
    couplings_2, _ = dipolar_geometry(positions_2, 1.0)

    assert np.isclose(
        couplings_1[0, 1],
        8 * couplings_2[0, 1],
    )


def test_secular_interaction_removes_off_diagonal_terms():
    """The secular approximation should remove dressed-basis transitions."""
    dressed_basis = [
        basis(2, 0),
        basis(2, 1),
    ]

    interaction = Qobj([
        [1.0, 0.5],
        [0.5, 2.0],
    ])

    secular = secular_interaction(
        interaction=interaction,
        dressed_basis=dressed_basis,
    )

    expected = np.diag([1.0, 2.0])

    assert np.allclose(
        secular.full(),
        expected,
    )


def test_secular_interaction_preserves_diagonal_two_spin_terms():
    """Interactions diagonal in the dressed product basis should be unchanged."""
    single_basis = [
        basis(3, 0),
        basis(3, 1),
        basis(3, 2),
    ]

    dressed_basis = [
        tensor(a, b)
        for a in single_basis
        for b in single_basis
    ]

    system = SpinSystem([1, 1])

    interaction = dipolar_interaction(
        system=system,
        site_i=0,
        site_j=1,
        coupling=1.0,
        direction=(0.0, 0.0, 1.0),
    )

    secular = secular_interaction(
        interaction=interaction,
        dressed_basis=dressed_basis,
    )

    dressed_matrix = operator_in_basis(
        secular,
        dressed_basis,
    ).full()

    off_diagonal = (
        dressed_matrix
        - np.diag(np.diag(dressed_matrix))
    )

    assert np.allclose(
        off_diagonal,
        np.zeros((9, 9)),
    )


def test_extract_effective_zz_coupling():
    """The projection should recover a known exact ZZ coupling."""
    system = SpinSystem([1, 1])

    _, _, sz = spin_operators(1)

    sz_1 = embed_operator(sz, 0, system)
    sz_2 = embed_operator(sz, 1, system)

    expected_coupling = 0.37

    interaction = expected_coupling * sz_1 * sz_2

    coupling, residual = extract_effective_zz_coupling(
        secular_hamiltonian=interaction,
        sz_i=sz_1,
        sz_j=sz_2,
    )

    assert np.isclose(
        coupling,
        expected_coupling,
    )

    assert np.allclose(
        residual.full(),
        np.zeros((9, 9)),
    )


def test_effective_zz_projection_ignores_identity_term():
    """An identity energy offset should not modify the extracted ZZ coupling."""
    system = SpinSystem([1, 1])

    _, _, sz = spin_operators(1)

    sz_1 = embed_operator(sz, 0, system)
    sz_2 = embed_operator(sz, 1, system)

    coupling_expected = 0.42

    interaction = (
        coupling_expected * sz_1 * sz_2
        + 3.0 * qeye([3, 3])
    )

    coupling, _ = extract_effective_zz_coupling(
        secular_hamiltonian=interaction,
        sz_i=sz_1,
        sz_j=sz_2,
    )

    assert np.isclose(
        coupling,
        coupling_expected,
    )


def test_relative_operator_residual_zero_for_exact_match():
    """An exact effective interaction should have zero relative residual."""
    system = SpinSystem([1, 1])

    _, _, sz = spin_operators(1)

    sz_1 = embed_operator(sz, 0, system)
    sz_2 = embed_operator(sz, 1, system)

    interaction = 0.5 * sz_1 * sz_2
    residual = 0 * interaction

    error = relative_operator_residual(
        operator=interaction,
        residual=residual,
    )

    assert np.isclose(error, 0.0)