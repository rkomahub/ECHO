import numpy as np

from echo_spin.control.rotations import two_qubit_rotation


def xy8_pulses(qubit: int):
    """Return the ideal XY8-1 pi-pulse cycle."""
    phases = ["x", "y", "x", "y", "y", "x", "y", "x"]

    return [
        two_qubit_rotation(
            qubit=qubit,
            angle=np.pi,
            axis=axis,
        )
        for axis in phases
    ]
