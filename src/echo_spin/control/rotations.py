from typing import Sequence, Union

import numpy as np
from qutip import Qobj, qeye, sigmax, sigmay, sigmaz, tensor


def single_qubit_rotation(
    angle: float,
    axis: Union[str, Sequence[float], np.ndarray],
) -> Qobj:
    """Return a qubit rotation; angle is in radians and axis is normalized."""
    if not np.isfinite(angle):
        raise ValueError("angle must be finite.")

    if isinstance(axis, str):
        directions = {
            "x": (1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
        }
        if axis not in directions:
            raise ValueError("axis must be 'x', 'y', 'z', or a real 3-vector.")
        direction = np.asarray(directions[axis])
    else:
        if np.iscomplexobj(axis):
            raise ValueError("axis must be real.")

        direction = np.asarray(axis, dtype=float)

        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("axis must contain three finite real components.")

        scale = np.max(np.abs(direction))
        if scale == 0:
            raise ValueError("axis must be nonzero.")

        direction = direction / scale
        direction = direction / np.linalg.norm(direction)

    generator = (
        direction[0] * sigmax()
        + direction[1] * sigmay()
        + direction[2] * sigmaz()
    )

    return (-1j * angle * generator / 2).expm()


def two_qubit_rotation(
    qubit: int,
    angle: float,
    axis: Union[str, Sequence[float], np.ndarray],
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
    axis: Union[str, Sequence[float], np.ndarray],
) -> Qobj:
    """Rotate two orthonormal states, leaving their complement unchanged."""
    if (
        not state_0.isket
        or not state_1.isket
        or state_0.dims != state_1.dims
    ):
        raise ValueError("States must be kets with matching dimensions.")

    if (
        not np.isclose(state_0.norm(), 1.0, atol=1e-12, rtol=0.0)
        or not np.isclose(state_1.norm(), 1.0, atol=1e-12, rtol=0.0)
        or abs(state_0.overlap(state_1)) > 1e-12
    ):
        raise ValueError("States must be normalized and orthogonal.")

    rotation = single_qubit_rotation(angle=angle, axis=axis)
    states = [state_0, state_1]

    result = (
        qeye(state_0.dims[0])
        - state_0.proj()
        - state_1.proj()
    )

    for i in range(2):
        for j in range(2):
            result += rotation[i, j] * states[i] * states[j].dag()

    return result
