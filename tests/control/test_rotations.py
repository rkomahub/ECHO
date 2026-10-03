import numpy as np

from qutip import qeye, sigmax, tensor
from echo_spin.control.rotations import single_qubit_rotation, two_qubit_rotation

def test_pi_x_rotation():
    rotation = single_qubit_rotation(np.pi, "x")

    expected = -1j * sigmax()

    assert np.allclose(rotation.full(), expected.full())


def test_two_qubit_rotation_on_first_qubit():
    rotation = two_qubit_rotation(0, np.pi, "x")

    expected = tensor(-1j * sigmax(), qeye(2))

    assert np.allclose(rotation.full(), expected.full())