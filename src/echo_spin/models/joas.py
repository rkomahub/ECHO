import numpy as np
from qutip import Qobj, qeye, tensor
from echo_spin.core.basis import dressed_spin_one_basis
from echo_spin.control.microwave import microwave_hamiltonian
from echo_spin.control.rotations import two_qubit_rotation

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
        raise ValueError("tau_1 must be positive")

    if abs(tau_2) > tau_1 / 2:
        raise ValueError("tau_2 must satisfy |tau_2| <= tau_1 / 2")

    if n_pi <= 0:
        raise ValueError("n_pi must be positive")

    nv1_times = [
        (k + 0.5) * tau_1
        for k in range(n_pi)
    ]

    nv2_times = [
        (k + 1.0) * tau_1 - tau_2
        for k in range(n_pi)
    ]

    return nv1_times, nv2_times


def xy8_gate_schedule(
    tau_1: float,
    tau_2: float,
):
    """Return the chronological 16-pulse schedule of the Joas XY8-1 gate."""

    nv1_times, nv2_times = xy8_gate_times(
        tau_1=tau_1,
        tau_2=tau_2,
        n_pi=8,
    )

    phases = ["x", "y", "x", "y", "y", "x", "y", "x"]

    schedule = []

    for time, phase in zip(nv1_times, phases):
        schedule.append((time, 1, phase))

    for time, phase in zip(nv2_times, phases):
        schedule.append((time, 2, phase))

    return sorted(schedule, key=lambda event: event[0])


def xy8_gate_intervals(
    tau_1: float,
    tau_2: float,
):
    """Return free-evolution intervals of the Joas XY8-1 gate."""

    schedule = xy8_gate_schedule(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    pulse_times = [time for time, _, _ in schedule]
    gate_duration = 8 * tau_1

    boundaries = [0.0] + pulse_times + [gate_duration]

    return [
        boundaries[index + 1] - boundaries[index]
        for index in range(len(boundaries) - 1)
    ]


def xy8_gate_pulses(
    tau_1: float,
    tau_2: float,
):
    """Return chronological ideal pi pulses for the Joas XY8-1 gate."""

    schedule = xy8_gate_schedule(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    pulses = []

    for _, nv, phase in schedule:
        qubit = 1 if nv == 1 else 0

        pulses.append(
            two_qubit_rotation(
                qubit=qubit,
                angle=np.pi,
                axis=phase,
            )
        )

    return pulses


def two_electron_hamiltonian(
    h_nv1: Qobj,
    h_nv2: Qobj,
    interaction: Qobj,
) -> Qobj:
    """Hamiltonian of two spin-1 NV electrons."""

    identity = qeye(3)

    return (
        tensor(h_nv1, identity)
        + tensor(identity, h_nv2)
        + interaction
    )


def two_electron_logical_basis(
    nv1_states: tuple[Qobj, Qobj],
    nv2_states: tuple[Qobj, Qobj],
) -> list[Qobj]:
    """Construct the two-electron logical product basis."""

    nv1_0, nv1_1 = nv1_states
    nv2_0, nv2_1 = nv2_states

    return [
        tensor(nv1_0, nv2_0),
        tensor(nv1_0, nv2_1),
        tensor(nv1_1, nv2_0),
        tensor(nv1_1, nv2_1),
    ]


def electron_logical_states(
    hamiltonian: Qobj,
    excited_state: str = "+1",
) -> tuple[Qobj, Qobj]:
    """Return dressed |0> and |1> electron-qubit states."""

    plus_one, zero, minus_one = dressed_spin_one_basis(
        hamiltonian
    )

    if excited_state == "+1":
        one = plus_one
    elif excited_state == "-1":
        one = minus_one
    else:
        raise ValueError(
            "excited_state must be '+1' or '-1'."
        )

    return zero, one