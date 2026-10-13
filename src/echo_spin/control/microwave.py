import numpy as np
from qutip import Qobj, sigmax, sigmay, sigmaz


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


def control_hamiltonian(
    time: float,
    operators: list[Qobj],
    amplitudes,
) -> Qobj:
    """Return Hc(t) = sum_a f_a(t) A_a.

    Operators must be Hermitian with matching dimensions.
    Amplitudes are real constants or callables of absolute time.
    Operator normalization and coefficient units are supplied by the caller.
    No rotating-frame or RWA approximation is applied.
    """
    if not np.isfinite(time):
        raise ValueError("time must be finite.")

    if not operators or len(operators) != len(amplitudes):
        raise ValueError("Provide one amplitude per operator and at least one term.")

    dimensions = operators[0].dims
    result = 0 * operators[0]

    for operator, amplitude in zip(operators, amplitudes):
        if operator.dims != dimensions or not operator.isherm:
            raise ValueError(
                "Control operators must be Hermitian with matching dimensions."
            )

        coefficient = control_amplitude(amplitude, time)

        if np.iscomplexobj(coefficient) or not np.isfinite(coefficient):
            raise ValueError("Control amplitudes must be finite and real.")

        result += coefficient * operator

    return result


def linear_drive_qubit_hamiltonian(
    time: float,
    omega_0: float,
    carrier: float,
    rabi_frequency: float,
    phase: float = 0.0,
    frame: str = "lab",
    rwa: bool = False,
) -> Qobj:
    """Construct a linearly driven qubit in a specified modelling frame.

    H_lab = omega_0 Z/2 + rabi_frequency cos(carrier*t + phase) X.

    All frequencies are angular frequencies. The rotating frame uses
    R(t) = exp(-i carrier*t Z/2). RWA is available only in that frame.
    """
    parameters = [time, omega_0, carrier, rabi_frequency, phase]

    if any(
        np.iscomplexobj(value) or not np.isfinite(value)
        for value in parameters
    ):
        raise ValueError("Parameters must be finite and real.")

    if carrier <= 0:
        raise ValueError("carrier must be positive.")

    if frame not in ("lab", "rotating"):
        raise ValueError("frame must be 'lab' or 'rotating'.")

    if frame == "lab":
        if rwa:
            raise ValueError("RWA requires frame='rotating'.")

        return control_hamiltonian(
            time,
            operators=[sigmaz() / 2, sigmax()],
            amplitudes=[
                omega_0,
                rabi_frequency * np.cos(carrier * time + phase),
            ],
        )

    x_coefficient = rabi_frequency * np.cos(phase)
    y_coefficient = rabi_frequency * np.sin(phase)

    if not rwa:
        counter_phase = 2 * carrier * time + phase
        x_coefficient += rabi_frequency * np.cos(counter_phase)
        y_coefficient -= rabi_frequency * np.sin(counter_phase)

    return control_hamiltonian(
        time,
        operators=[sigmaz() / 2, sigmax() / 2, sigmay() / 2],
        amplitudes=[
            omega_0 - carrier,
            x_coefficient,
            y_coefficient,
        ],
    )


def logical_axis_carrier_phase(
    axis: str,
    transition_element: complex,
) -> float:
    """Return the cosine-carrier phase for a positive logical X or Y rotation."""
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'.")

    element = complex(transition_element)

    if not np.isfinite(element) or abs(element) == 0:
        raise ValueError("transition_element must be finite and non-zero.")

    phase = np.angle(element)

    if axis == "y":
        phase -= np.pi / 2

    return float(phase)
