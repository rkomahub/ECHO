import numpy as np
import pytest

from qutip import jmat

from echo_spin.operators import embed_operator, spin_operators
from echo_spin.system import SpinSystem


def test_spin_half_operator_dimensions():
    """Spin-1/2 operators should be 2x2 matrices."""
    sx, sy, sz = spin_operators(0.5)

    assert sx.shape == (2, 2)
    assert sy.shape == (2, 2)
    assert sz.shape == (2, 2)


def test_spin_one_operator_dimensions():
    """Spin-1 operators should be 3x3 matrices."""
    sx, sy, sz = spin_operators(1)

    assert sx.shape == (3, 3)
    assert sy.shape == (3, 3)
    assert sz.shape == (3, 3)


def test_spin_commutation_relation():
    """Angular-momentum operators should satisfy [Sx, Sy] = i Sz."""
    sx, sy, sz = spin_operators(1)

    commutator = sx * sy - sy * sx

    assert np.allclose(
        commutator.full(),
        (1j * sz).full(),
    )


def test_embed_operator_in_two_spin_one_system():
    """A local spin-1 operator should become a 9x9 operator."""
    system = SpinSystem([1, 1])

    _, _, sz = spin_operators(1)

    embedded = embed_operator(sz, 0, system)

    assert embedded.shape == (9, 9)


def test_embed_operator_in_mixed_spin_system():
    """Operator embedding should work for mixed local dimensions."""
    system = SpinSystem([1, 0.5])

    sx = jmat(0.5, "x")

    embedded = embed_operator(sx, 1, system)

    assert embedded.shape == (6, 6)


def test_invalid_site_raises_error():
    """Embedding outside the system should fail."""
    system = SpinSystem([1])

    _, _, sz = spin_operators(1)

    with pytest.raises(IndexError):
        embed_operator(sz, 1, system)


def test_wrong_operator_dimension_raises_error():
    """The local operator must match the selected subsystem dimension."""
    system = SpinSystem([1])

    wrong_operator = jmat(0.5, "z")

    with pytest.raises(ValueError):
        embed_operator(wrong_operator, 0, system)