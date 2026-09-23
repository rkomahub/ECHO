import numpy as np
from qutip import Qobj


def control_operator(
    electronic_x_operators: list[Qobj],
    nuclear_x_operators: list[Qobj],
    gamma_ratio: float,
) -> Qobj:
    """Construct the microwave control operator.

    H_control = sum_i Sx_i + gamma_ratio * sum_i Ix_i
    """
    if len(electronic_x_operators) != len(nuclear_x_operators):
        raise ValueError(
            "Electronic and nuclear operator lists must have the same length."
        )

    if not electronic_x_operators:
        raise ValueError("At least one NV center must be provided.")

    electronic_term = sum(electronic_x_operators)
    nuclear_term = sum(nuclear_x_operators)

    return electronic_term + gamma_ratio * nuclear_term


def control_amplitude(value, time: float) -> float:
    """Evaluate a constant or time-dependent control amplitude."""
    if callable(value):
        return value(time)

    return float(value)


def microwave_coefficient(
    time: float,
    omega: float,
    phase: float,
    omega_x,
    omega_y,
) -> float:
    """Return the microwave coefficient for one carrier frequency."""
    amplitude_x = control_amplitude(omega_x, time)
    amplitude_y = control_amplitude(omega_y, time)

    return np.sqrt(2.0) * (
        amplitude_x * np.cos(omega * time + phase)
        + amplitude_y * np.sin(omega * time + phase)
    )


def microwave_hamiltonian(
    time: float,
    control: Qobj,
    drives: list[dict],
) -> Qobj:
    """Construct the microwave Hamiltonian at a specified time.

    Each drive dictionary must contain:
        omega
        phase
        omega_x
        omega_y
    """
    coefficient = 0.0

    for drive in drives:
        coefficient += microwave_coefficient(
            time=time,
            omega=drive["omega"],
            phase=drive["phase"],
            omega_x=drive["omega_x"],
            omega_y=drive["omega_y"],
        )

    return coefficient * control

""" later describe two MW carries as (REMOVE LATER)
drives = [
    {
        "omega": omega_1,
        "phase": 0.0,
        "omega_x": Omega_1_x,
        "omega_y": Omega_1_y,
    },
    {
        "omega": omega_2,
        "phase": 0.0,
        "omega_x": Omega_2_x,
        "omega_y": Omega_2_y,
    },
]
"""