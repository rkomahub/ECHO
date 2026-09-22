import numpy as np

from collections.abc import Sequence
from qutip import Qobj

from echo_spin.hamiltonians import nv_hamiltonian
from echo_spin.system import SpinSystem
from echo_spin.interactions import dipolar_interaction

def nv_register_hamiltonian(
    system: SpinSystem,
    nv_parameters: Sequence[dict],
    dipolar_couplings=None,
    dipolar_directions=None,
) -> Qobj:
    """Construct the Hamiltonian of a register of non-interacting NV centers.

    Each NV center is represented by two consecutive spin-1 subsystems:

        electron site = 2 * i
        nuclear site  = 2 * i + 1

    Parameters
    ----------
    system : SpinSystem
        Composite spin system containing all NV electrons and nuclei.
    nv_parameters : sequence of dict
        Parameters of each NV center. Each dictionary must contain
        D, omega_e, Q, omega_n and A.

    Returns
    -------
    Qobj
        Hamiltonian of the complete NV register.
    """
    number_of_nv = len(nv_parameters)

    if system.number_of_spins != 2 * number_of_nv:
        raise ValueError(
            "An NV register requires two spin subsystems per NV center."
        )

    for spin in system.spins:
        if spin != 1:
            raise ValueError(
                "NV electronic and 14N nuclear spins must both have spin 1."
            )

    hamiltonian = 0

    for i, parameters in enumerate(nv_parameters):
        electron_site = 2 * i
        nuclear_site = 2 * i + 1

        hamiltonian += nv_hamiltonian(
            system=system,
            electron_site=electron_site,
            nuclear_site=nuclear_site,
            D=parameters["D"],
            omega_e=parameters["omega_e"],
            Q=parameters["Q"],
            omega_n=parameters["omega_n"],
            A=parameters["A"],
        )

    if dipolar_couplings is not None or dipolar_directions is not None:
        if dipolar_couplings is None or dipolar_directions is None:
            raise ValueError(
                "Both dipolar_couplings and dipolar_directions must be provided."
            )

        hamiltonian += nv_dipolar_interaction(
            system=system,
            couplings=dipolar_couplings,
            directions=dipolar_directions,
        )

    return hamiltonian

def nv_dipolar_interaction(
    system: SpinSystem,
    couplings,
    directions,
) -> Qobj:
    """Construct all pairwise electronic dipolar interactions in an NV register.

    Parameters
    ----------
    system : SpinSystem
        NV register ordered as electron, nucleus, electron, nucleus, ...
    couplings : array-like
        NxN matrix containing the dipolar coupling J_ij.
    directions : array-like
        NxNx3 array containing the relative direction vectors r_ij.

    Returns
    -------
    Qobj
        Sum of all pairwise electronic dipolar interactions.
    """
    if system.number_of_spins % 2 != 0:
        raise ValueError(
            "An NV register must contain two spin subsystems per NV center."
        )

    number_of_nv = system.number_of_spins // 2

    couplings = np.asarray(couplings, dtype=float)
    directions = np.asarray(directions, dtype=float)

    if couplings.shape != (number_of_nv, number_of_nv):
        raise ValueError(
            "couplings must have shape (N, N)."
        )

    if directions.shape != (number_of_nv, number_of_nv, 3):
        raise ValueError(
            "directions must have shape (N, N, 3)."
        )

    interaction = 0

    """Loop over all unique pairs of NV centers."""
    for i in range(number_of_nv): 
        for j in range(i + 1, number_of_nv):
            electron_i = 2 * i
            electron_j = 2 * j

            interaction += dipolar_interaction(
                system=system,
                site_i=electron_i,
                site_j=electron_j,
                coupling=couplings[i, j],
                direction=directions[i, j],
            )

    return interaction