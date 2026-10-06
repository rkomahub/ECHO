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


def average_gate_fidelity(
    dynamical_map,
    target: Qobj,
    basis_states: list[Qobj],
) -> float:
    """Return the average gate fidelity of a dynamical map."""
    dimension = len(basis_states)

    if target.shape != (dimension, dimension):
        raise ValueError(
            "target dimension must match the number of basis states."
        )

    fidelity_sum = 0.0 + 0.0j

    for state_i in basis_states:
        for state_j in basis_states:
            rho_ij = state_i * state_j.dag()
            rho_jj = state_j * state_j.dag()

            evolved_ij = dynamical_map(rho_ij)
            evolved_jj = dynamical_map(rho_jj)

            term_1 = (
                state_i.dag()
                * target.dag()
                * evolved_ij
                * target
                * state_j
            )

            term_2 = (
                state_i.dag()
                * target.dag()
                * evolved_jj
                * target
                * state_i
            )

            fidelity_sum += term_1 + term_2

    return float(
        np.real(fidelity_sum)
        / (dimension * (dimension + 1))
    )
