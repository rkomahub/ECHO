import numpy as np
from qutip import Qobj, qeye, tensor

from echo_spin.control.microwave import microwave_hamiltonian
from echo_spin.control.rotations import two_qubit_rotation
from echo_spin.core.basis import dressed_spin_one_basis
from echo_spin.dynamics.sequences import (
    finite_pulse_intervals,
    finite_pulse_sequence_propagator,
    logical_sine_pulse_propagator,
)


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
    """Return pulse centers for two staggered trains within the gate duration."""
    if not np.isfinite(tau_1) or tau_1 <= 0:
        raise ValueError("tau_1 must be finite and positive.")

    if (
        not np.isfinite(tau_2)
        or tau_2 < 0
        or tau_2 > tau_1 / 2
    ):
        raise ValueError("tau_2 must satisfy 0 <= tau_2 <= tau_1 / 2.")

    if (
        isinstance(n_pi, (bool, np.bool_))
        or not isinstance(n_pi, (int, np.integer))
        or n_pi <= 0
    ):
        raise ValueError("n_pi must be a positive integer.")

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
    n_pi: int = 8,
):
    """Return the chronological schedule for complete Joas XY8 cycles."""
    nv1_times, nv2_times = xy8_gate_times(
        tau_1=tau_1,
        tau_2=tau_2,
        n_pi=n_pi,
    )

    if n_pi % 8 != 0:
        raise ValueError("n_pi must be a multiple of 8 for complete XY8 cycles.")

    cycle = ["x", "y", "x", "y", "y", "x", "y", "x"]
    phases = cycle * (n_pi // 8)

    schedule = [
        (time, nv, phase)
        for nv, times in [(1, nv1_times), (2, nv2_times)]
        for time, phase in zip(times, phases)
    ]

    return sorted(schedule, key=lambda event: event[0])


def xy8_gate_intervals(
    tau_1: float,
    tau_2: float,
    n_pi: int = 8,
):
    """Return free intervals for complete Joas XY8 cycles."""
    schedule = xy8_gate_schedule(tau_1, tau_2, n_pi)
    pulse_times = [time for time, _, _ in schedule]
    gate_duration = sequence_duration(tau_1, n_pi)

    boundaries = [0.0] + pulse_times + [gate_duration]

    return [
        following - previous
        for previous, following in zip(boundaries[:-1], boundaries[1:])
    ]


def xy8_gate_pulses(
    tau_1: float,
    tau_2: float,
    n_pi: int = 8,
    nv_order: tuple[int, int] = (2, 1),
):
    """Return ideal XY8 pulses with NV labels ordered from left to right."""
    if (
        not isinstance(nv_order, tuple)
        or nv_order not in ((1, 2), (2, 1))
    ):
        raise ValueError("nv_order must be (1, 2) or (2, 1).")

    schedule = xy8_gate_schedule(tau_1, tau_2, n_pi)
    nv_to_qubit = {
        nv: index
        for index, nv in enumerate(nv_order)
    }

    return [
        two_qubit_rotation(
            qubit=nv_to_qubit[nv],
            angle=np.pi,
            axis=phase,
        )
        for _, nv, phase in schedule
    ]


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


def joas_rb_primitive_builders(
    free_hamiltonian,
    controls,
    carriers,
    transition_elements,
    pi_pulse_duration,
    tau_1,
    tau_2,
    pulse_steps=5000,
) -> dict:
    """Build laboratory-frame RB primitives with full drift and absolute pulse timing."""
    controls = dict(controls)
    carriers = dict(carriers)
    transition_elements = dict(transition_elements)

    if any(
        set(mapping) != {1, 2}
        for mapping in [controls, carriers, transition_elements]
    ):
        raise ValueError("Provide controls, carriers and elements for NV1 and NV2.")

    if not np.isfinite(pi_pulse_duration) or pi_pulse_duration <= 0:
        raise ValueError("pi_pulse_duration must be finite and positive.")

    schedule = xy8_gate_schedule(
        tau_1=tau_1,
        tau_2=tau_2,
    )
    gate_duration = 8 * tau_1

    free_intervals = finite_pulse_intervals(
        pulse_centers=[center for center, _, _ in schedule],
        pulse_duration=pi_pulse_duration,
        total_duration=gate_duration,
    )

    def half_pulse_builder(nv, axis):
        """Create a positive logical half-pi pulse builder for the selected NV."""
        def build(start_time):
            """Propagate the full register during this absolute-time half-pi pulse."""
            duration = pi_pulse_duration / 2

            propagator = logical_sine_pulse_propagator(
                free_hamiltonian=free_hamiltonian,
                control_operator=controls[nv],
                carrier=carriers[nv],
                transition_element=transition_elements[nv],
                angle=np.pi / 2,
                axis=axis,
                duration=duration,
                steps=pulse_steps,
                start_time=start_time,
            )

            return propagator, duration

        return build

    def build_entangler(start_time):
        """Propagate the validated XY8 waveform shifted to the circuit's current time."""
        pulse_propagators = []

        for center, nv, axis in schedule:
            pulse_start = (
                start_time + center - pi_pulse_duration / 2
            )

            # Negative Y pi reproduces the existing +pi/2 carrier-phase waveform.
            angle = np.pi if axis == "x" else -np.pi

            pulse_propagators.append(
                logical_sine_pulse_propagator(
                    free_hamiltonian=free_hamiltonian,
                    control_operator=controls[nv],
                    carrier=carriers[nv],
                    transition_element=transition_elements[nv],
                    angle=angle,
                    axis=axis,
                    duration=pi_pulse_duration,
                    steps=pulse_steps,
                    start_time=pulse_start,
                )
            )

        propagator = finite_pulse_sequence_propagator(
            free_hamiltonian=free_hamiltonian,
            durations=free_intervals,
            pulse_propagators=pulse_propagators,
        )

        return propagator, gate_duration

    return {
        0: half_pulse_builder(1, "x"),
        1: half_pulse_builder(1, "y"),
        2: half_pulse_builder(2, "x"),
        3: half_pulse_builder(2, "y"),
        4: build_entangler,
    }
