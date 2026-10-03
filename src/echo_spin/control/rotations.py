import numpy as np
from qutip import Qobj, qeye, sigmax, sigmay, tensor


def single_qubit_rotation(
    angle: float,
    axis: str,
) -> Qobj:
    """Return an ideal single-qubit rotation."""
    if axis == "x":
        pauli = sigmax()
    elif axis == "y":
        pauli = sigmay()
    else:
        raise ValueError("axis must be 'x' or 'y'.")

    return (-1j * angle * pauli / 2).expm()


def two_qubit_rotation(
    qubit: int,
    angle: float,
    axis: str,
) -> Qobj:
    """Embed an ideal rotation on one qubit of a two-qubit system."""
    rotation = single_qubit_rotation(angle, axis)

    if qubit == 0:
        return tensor(rotation, qeye(2))
    if qubit == 1:
        return tensor(qeye(2), rotation)

    raise ValueError("qubit must be 0 or 1.")