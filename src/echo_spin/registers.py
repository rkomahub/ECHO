from collections.abc import Sequence
from qutip import Qobj

from echo_spin.hamiltonians import nv_hamiltonian
from echo_spin.system import SpinSystem

def nv_register_hamiltonian(
    system: SpinSystem,
    nv_parameters: Sequence[dict],
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

    return hamiltonian