import numpy as np
import pytest
from qutip import qeye

from echo_spin.gates.gates import sqrt_zz_gate, conditional_phase


def test_sqrt_zz_gate_matrix():
    """Check the matrix representation of the sqrt(ZZ) gate."""
    gate = sqrt_zz_gate()

    expected = np.diag([1.0, 1.0j, 1.0j, 1.0])

    assert np.allclose(gate.full(), expected)


def test_sqrt_zz_gate_is_unitary():
    """Check that the sqrt(ZZ) gate is unitary."""
    gate = sqrt_zz_gate()

    identity = gate.dag() * gate

    assert np.allclose(identity.full(), qeye([2, 2]).full())


def test_sqrt_zz_conditional_phase():
    """Check the conditional phase of the sqrt(ZZ) gate."""
    phase = conditional_phase(sqrt_zz_gate())

    assert np.exp(1j * phase) == pytest.approx(-1.0)


def test_inverse_sqrt_zz_conditional_phase():
    """Check the conditional phase of the inverse sqrt(ZZ) gate."""
    phase = conditional_phase(sqrt_zz_gate().dag())

    assert np.exp(1j * phase) == pytest.approx(-1.0)