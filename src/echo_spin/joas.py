from qutip import Qobj

from echo_spin.controls import microwave_hamiltonian


def free_hamiltonian(
    nv_hamiltonians: list[Qobj],
    interaction: Qobj,
) -> Qobj:
    """Construct the free Hamiltonian of an interacting NV register.

    H_free = sum_i H_NV_i + H_int
    """
    if not nv_hamiltonians:
        raise ValueError("At least one NV Hamiltonian must be provided.")

    hamiltonian = sum(nv_hamiltonians) # deliberately not restricted to two NVs.

    return hamiltonian + interaction


def driven_hamiltonian(
    time: float,
    free: Qobj,
    control: Qobj,
    drives: list[dict],
) -> Qobj:
    """Construct the complete noiseless driven Hamiltonian.

    H_driven(t) = H_free + H_mw(t)
    """
    microwave = microwave_hamiltonian(
        time=time,
        control=control,
        drives=drives,
    )

    return free + microwave