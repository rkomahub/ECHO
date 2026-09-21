import pytest

from echo_spin.system import SpinSystem


def test_single_spin_half_dimension():
    """A spin-1/2 system should have Hilbert-space dimension 2."""
    system = SpinSystem([0.5])

    assert system.dimensions == (2,)
    assert system.dimension == 2


def test_single_spin_one_dimension():
    """A spin-1 system should have Hilbert-space dimension 3."""
    system = SpinSystem([1])

    assert system.dimensions == (3,)
    assert system.dimension == 3


def test_multiple_spin_one_system():
    """Three spin-1 systems should have total dimension 3^3 = 27."""
    system = SpinSystem([1, 1, 1])

    assert system.number_of_spins == 3
    assert system.dimensions == (3, 3, 3)
    assert system.dimension == 27


def test_mixed_spin_system():
    """Mixed local spins should generate the correct tensor dimension."""
    system = SpinSystem([1, 0.5, 1])

    assert system.dimensions == (3, 2, 3)
    assert system.dimension == 18


def test_empty_spin_system_raises_error():
    """A system without subsystems is not physically meaningful here."""
    with pytest.raises(ValueError):
        SpinSystem([])


def test_negative_spin_raises_error():
    """Negative spin quantum numbers should be rejected."""
    with pytest.raises(ValueError):
        SpinSystem([-0.5])


def test_invalid_spin_quantum_number_raises_error():
    """Spin must be integer or half-integer."""
    with pytest.raises(ValueError):
        SpinSystem([0.3])