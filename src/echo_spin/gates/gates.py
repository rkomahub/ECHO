import numpy as np
from qutip import Qobj


def sqrt_zz_gate() -> Qobj:
    """Return the ideal two-qubit sqrt(ZZ) gate."""
    return Qobj(
        np.diag([1.0, 1.0j, 1.0j, 1.0]),
        dims=[[2, 2], [2, 2]],
    )


def conditional_phase(operator: Qobj) -> float:
    """Return the two-qubit conditional phase of a diagonal operator."""
    diagonal = np.diag(operator.full())

    if np.any(np.abs(diagonal) == 0):
        raise ValueError("Diagonal elements must be nonzero.")

    phases = np.angle(diagonal)

    phase = (
        phases[0]
        - phases[1]
        - phases[2]
        + phases[3]
    )

    return np.angle(np.exp(1j * phase))


def remove_local_z_phases(operator: Qobj) -> Qobj:
    """Remove global and single-qubit Z phases from a two-qubit operator."""
    diagonal = np.diag(operator.full())

    if np.any(np.abs(diagonal) == 0):
        raise ValueError("Diagonal elements must be nonzero.")

    phases = np.angle(diagonal)

    phi_00, phi_01, phi_10, _ = phases

    correction = np.diag([
        np.exp(-1j * phi_00),
        np.exp(-1j * phi_01),
        np.exp(-1j * phi_10),
        np.exp(-1j * (phi_01 + phi_10 - phi_00)),
    ])

    return Qobj(
        correction,
        dims=operator.dims,
    ) * operator