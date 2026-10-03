import numpy as np
import pytest
from qutip import Qobj, qeye

from echo_spin.dynamics.propagators import (
    rotating_frame_propagator,
    static_propagator,
    time_dependent_propagator,
)


def test_static_propagator_at_zero_time():
    """Check that zero-time evolution gives the identity."""
    hamiltonian = Qobj([[1.0, 0.0], [0.0, -1.0]])

    propagator = static_propagator(
        hamiltonian=hamiltonian,
        time=0.0,
    )

    assert np.allclose(
        propagator.full(),
        qeye(2).full(),
    )


def test_static_propagator_diagonal_hamiltonian():
    """Check static propagation against an analytically solvable Hamiltonian."""
    hamiltonian = Qobj([[1.0, 0.0], [0.0, -1.0]])
    time = 0.4

    propagator = static_propagator(
        hamiltonian=hamiltonian,
        time=time,
    )

    expected = Qobj([
        [np.exp(-1j * time), 0.0],
        [0.0, np.exp(1j * time)],
    ])

    assert np.allclose(
        propagator.full(),
        expected.full(),
    )


def test_static_propagator_is_unitary():
    """Check that Hermitian static evolution produces a unitary propagator."""
    hamiltonian = Qobj([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    propagator = static_propagator(
        hamiltonian=hamiltonian,
        time=0.7,
    )

    identity = propagator.dag() * propagator

    assert np.allclose(
        identity.full(),
        qeye(2).full(),
    )


def test_rotating_frame_removes_matching_evolution():
    """Check that a matching rotating frame removes static evolution."""
    generator = Qobj([[1.0, 0.0], [0.0, -1.0]])
    time = 0.6

    propagator = static_propagator(
        hamiltonian=generator,
        time=time,
    )

    rotating = rotating_frame_propagator(
        propagator=propagator,
        generator=generator,
        time=time,
    )

    assert np.allclose(
        rotating.full(),
        qeye(2).full(),
    )


def test_time_dependent_propagator_zero_hamiltonian():
    """Check that a zero Hamiltonian produces identity evolution."""
    hamiltonian = lambda time: Qobj([[0.0, 0.0], [0.0, 0.0]])

    propagator = time_dependent_propagator(
        hamiltonian=hamiltonian,
        t0=0.0,
        t1=1.0,
        steps=10,
    )

    assert np.allclose(propagator.full(), qeye(2).full())


def test_time_dependent_propagator_constant_hamiltonian():
    """Check numerical propagation against exact static evolution."""
    h = Qobj([[1.0, 0.0], [0.0, -1.0]])

    hamiltonian = lambda time: h

    numerical = time_dependent_propagator(
        hamiltonian=hamiltonian,
        t0=0.0,
        t1=0.7,
        steps=20,
    )

    exact = static_propagator(
        hamiltonian=h,
        time=0.7,
    )

    assert np.allclose(numerical.full(), exact.full())


def test_time_dependent_propagator_rejects_invalid_steps():
    """Check that the propagator requires a positive number of steps."""
    hamiltonian = lambda time: Qobj([[0.0, 0.0], [0.0, 0.0]])

    with pytest.raises(ValueError):
        time_dependent_propagator(
            hamiltonian=hamiltonian,
            t0=0.0,
            t1=1.0,
            steps=0,
        )