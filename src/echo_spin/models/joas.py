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


def interaction_time(
    tau_2: float,
    n_pi: int,
) -> float:
    """Effective dipolar evolution time of the Joas sequence."""
    if n_pi <= 0:
        raise ValueError("n_pi must be positive.")

    return n_pi * tau_2


def sequence_duration(
    tau_1: float,
    n_pi: int,
) -> float:
    """Total duration of the ideal Joas refocusing sequence."""
    if tau_1 <= 0:
        raise ValueError("tau_1 must be positive.")

    if n_pi <= 0:
        raise ValueError("n_pi must be positive.")

    return n_pi * tau_1


def xy8_gate_times(
    tau_1: float,
    tau_2: float,
    n_pi: int = 8,
):
    """Pulse-center times for the two staggered NV pulse trains."""

    if tau_1 <= 0:
        raise ValueError("tau_1 must be positive.")

    if abs(tau_2) > tau_1 / 2:
        raise ValueError("tau_2 must satisfy |tau_2| <= tau_1 / 2.")

    if n_pi <= 0:
        raise ValueError("n_pi must be positive.")

    centers = [
        (k + 0.5) * tau_1
        for k in range(n_pi)
    ]

    nv1_times = [
        center - tau_2 / 2
        for center in centers
    ]

    nv2_times = [
        center + tau_2 / 2
        for center in centers
    ]

    return nv1_times, nv2_times