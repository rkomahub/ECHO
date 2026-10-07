import numpy as np
from qutip import Qobj, qeye, sigmaz, tensor


def exponential_coherence(
    time: float,
    coherence_time: float,
) -> float:
    """Return exponential coherence exp(-t / T2)."""
    if time < 0:
        raise ValueError("time must be non-negative.")

    if coherence_time <= 0:
        raise ValueError("coherence_time must be positive.")

    return float(
        np.exp(-time / coherence_time)
    )


def dephasing_channel(
    density_matrix: Qobj,
    time: float,
    coherence_time: float,
) -> Qobj:
    """Apply exponential pure dephasing to a single qubit."""
    if density_matrix.shape != (2, 2):
        raise ValueError(
            "dephasing_channel requires a single-qubit density matrix."
        )

    coherence = exponential_coherence(
        time=time,
        coherence_time=coherence_time,
    )

    identity = qeye(2)
    sigma_z = sigmaz()

    k0 = np.sqrt(
        (1 + coherence) / 2
    ) * identity

    k1 = np.sqrt(
        (1 - coherence) / 2
    ) * sigma_z

    return (
        k0 * density_matrix * k0.dag()
        + k1 * density_matrix * k1.dag()
    )


def two_qubit_dephasing_channel(
    density_matrix: Qobj,
    time: float,
    coherence_times: tuple[float, float],
) -> Qobj:
    """Apply independent pure dephasing to two qubits."""
    if density_matrix.shape != (4, 4):
        raise ValueError(
            "two_qubit_dephasing_channel requires a two-qubit density matrix."
        )

    t2_1, t2_2 = coherence_times

    coherence_1 = exponential_coherence(
        time=time,
        coherence_time=t2_1,
    )
    coherence_2 = exponential_coherence(
        time=time,
        coherence_time=t2_2,
    )

    identity = qeye(2)
    sigma_z = sigmaz()

    kraus_1 = [
        np.sqrt((1 + coherence_1) / 2) * identity,
        np.sqrt((1 - coherence_1) / 2) * sigma_z,
    ]

    kraus_2 = [
        np.sqrt((1 + coherence_2) / 2) * identity,
        np.sqrt((1 - coherence_2) / 2) * sigma_z,
    ]

    result = 0 * density_matrix

    for k1 in kraus_1:
        for k2 in kraus_2:
            kraus = tensor(k1, k2)
            result += kraus * density_matrix * kraus.dag()

    return result


def ornstein_uhlenbeck_noise(
    times: np.ndarray,
    sigma: float,
    correlation_time: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a stationary Ornstein-Uhlenbeck noise trajectory."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    if correlation_time <= 0:
        raise ValueError("correlation_time must be positive.")

    if len(times) < 2:
        raise ValueError("times must contain at least two points.")

    dt = np.diff(times)

    if np.any(dt <= 0):
        raise ValueError("times must be strictly increasing.")

    noise = np.empty(len(times))

    noise[0] = rng.normal(
        scale=sigma
    )

    for index, step in enumerate(dt, start=1):
        decay = np.exp(
            -step / correlation_time
        )

        noise[index] = (
            decay * noise[index - 1]
            + sigma
            * np.sqrt(1 - decay**2)
            * rng.normal()
        )

    return noise


def dephasing_hamiltonian(
    noise_value: float,
) -> Qobj:
    """Construct a single-qubit longitudinal dephasing Hamiltonian."""
    return (
        noise_value
        / 2
        * sigmaz()
    )


def stochastic_dephasing_propagator(
    times: np.ndarray,
    noise: np.ndarray,
) -> Qobj:
    """Propagate a qubit under a sampled longitudinal noise trajectory."""
    if len(times) != len(noise):
        raise ValueError(
            "times and noise must have the same length."
        )

    if len(times) < 2:
        raise ValueError(
            "times must contain at least two points."
        )

    dt = np.diff(times)

    if np.any(dt <= 0):
        raise ValueError(
            "times must be strictly increasing."
        )

    propagator = qeye(2)

    for index, step in enumerate(dt):
        hamiltonian = dephasing_hamiltonian(
            noise_value=noise[index],
        )

        propagator = (
            -1j * hamiltonian * step
        ).expm() * propagator

    return propagator
