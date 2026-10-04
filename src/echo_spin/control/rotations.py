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


def selective_rotation(
    state_0: Qobj,
    state_1: Qobj,
    angle: float,
    axis: str,
) -> Qobj:
    """Rotate a selected two-level subspace while leaving its complement unchanged."""

    projector_0 = state_0 * state_0.dag()
    projector_1 = state_1 * state_1.dag()

    if axis == "x":
        generator = (
            state_0 * state_1.dag()
            + state_1 * state_0.dag()
        )
    elif axis == "y":
        generator = (
            -1j * state_0 * state_1.dag()
            + 1j * state_1 * state_0.dag()
        )
    else:
        raise ValueError("axis must be 'x' or 'y'.")

    identity = qeye(state_0.dims[0])

    return (
        identity
        - projector_0
        - projector_1
        + (
            -1j * angle * generator / 2
        ).expm()
        * (projector_0 + projector_1)
    )