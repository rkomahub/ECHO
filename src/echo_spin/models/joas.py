import numpy as np
from qutip import Qobj
from echo_spin.control.microwave import microwave_hamiltonian


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


def reduced_free_hamiltonian(
    delta_1: float,
    delta_2: float,
    coupling: float,
) -> Qobj:
    """Reduced two-qubit free Hamiltonian used in the Joas gate model."""
    return Qobj(
        np.diag([
            delta_1,
            0.0,
            delta_1 + delta_2 - coupling,
            delta_2,
        ]),
        dims=[[2, 2], [2, 2]],
    )


def two_pulse_gate_durations(
    tau_1: float,
    tau_2: float,
) -> list[float]:
    """Return the five free-evolution intervals of the two-pulse gate."""
    if tau_1 <= 0:
        raise ValueError("tau_1 must be positive.")

    if tau_2 < 0:
        raise ValueError("tau_2 must be non-negative.")

    if tau_2 > tau_1 / 2:
        raise ValueError("tau_2 cannot exceed tau_1 / 2.")

    return [
        tau_1 / 2,
        tau_1 / 2 - tau_2,
        tau_1 / 2 + tau_2,
        tau_1 / 2 - tau_2,
        tau_2,
    ]