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