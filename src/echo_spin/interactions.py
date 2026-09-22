import numpy as np
from qutip import Qobj

from echo_spin.operators import embed_operator, spin_operators
from echo_spin.system import SpinSystem


def dipolar_interaction(
    system: SpinSystem,
    site_i: int,
    site_j: int,
    coupling: float,
    direction,
) -> Qobj:
    """Construct the magnetic dipole-dipole interaction between two spins.

    Parameters
    ----------
    system : SpinSystem
        Composite spin system.
    site_i, site_j : int
        Sites of the interacting spins.
    coupling : float
        Dipolar coupling strength.
    direction : sequence of float
        Unit vector joining the two spins.
    """
    if site_i == site_j:
        raise ValueError("Dipolar interaction requires two different sites.")

    direction = np.asarray(direction, dtype=float)

    if direction.shape != (3,):
        raise ValueError("direction must contain three Cartesian components.")

    norm = np.linalg.norm(direction)

    if np.isclose(norm, 0.0):
        raise ValueError("direction must be non-zero.")

    r_hat = direction / norm

    spin_i = system.spins[site_i]
    spin_j = system.spins[site_j]

    ops_i = [
        embed_operator(op, site_i, system)
        for op in spin_operators(spin_i)
    ]

    ops_j = [
        embed_operator(op, site_j, system)
        for op in spin_operators(spin_j)
    ]

    scalar_product = sum(
        op_i * op_j
        for op_i, op_j in zip(ops_i, ops_j)
    )

    projection_i = sum(
        component * operator
        for component, operator in zip(r_hat, ops_i)
    )

    projection_j = sum(
        component * operator
        for component, operator in zip(r_hat, ops_j)
    )

    return coupling * (
        scalar_product
        - 3 * projection_i * projection_j
    )

def dipolar_geometry(
    positions,
    coupling_prefactor: float,
):
    """Compute pairwise dipolar couplings and directions from positions.

    Parameters
    ----------
    positions : array-like
        Cartesian positions with shape (N, 3).
    coupling_prefactor : float
        Constant C defining J_ij = C / r_ij^3.

    Returns
    -------
    couplings : ndarray
        NxN matrix of dipolar coupling strengths.
    directions : ndarray
        NxNx3 array of unit vectors pointing from i to j.
    """
    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")

    number_of_spins = len(positions)

    couplings = np.zeros((number_of_spins, number_of_spins))
    directions = np.zeros((number_of_spins, number_of_spins, 3))

    for i in range(number_of_spins):
        for j in range(i + 1, number_of_spins):
            displacement = positions[j] - positions[i]
            distance = np.linalg.norm(displacement)

            if np.isclose(distance, 0.0):
                raise ValueError(
                    "Two spins cannot occupy the same position."
                )

            direction = displacement / distance
            coupling = coupling_prefactor / distance**3

            couplings[i, j] = coupling
            couplings[j, i] = coupling

            directions[i, j] = direction
            directions[j, i] = -direction

    return couplings, directions