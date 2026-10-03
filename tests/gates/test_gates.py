import numpy as np
from qutip import qeye

from echo_spin.gates.gates import sqrt_zz_gate


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