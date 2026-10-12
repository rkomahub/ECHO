import numpy as np
from qutip import Qobj, qeye, sigmaz, tensor

from echo_spin.control.microwave import control_hamiltonian


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


def ensemble_dephasing_map(
    density_matrix: Qobj,
    times: np.ndarray,
    sigma: float,
    correlation_time: float,
    realizations: int,
    rng: np.random.Generator,
) -> Qobj:
    """Average qubit evolution over Ornstein-Uhlenbeck noise realizations."""
    if realizations <= 0:
        raise ValueError(
            "realizations must be positive."
        )

    average_density = 0 * density_matrix

    for _ in range(realizations):
        noise = ornstein_uhlenbeck_noise(
            times=times,
            sigma=sigma,
            correlation_time=correlation_time,
            rng=rng,
        )

        propagator = stochastic_dephasing_propagator(
            times=times,
            noise=noise,
        )

        average_density += (
            propagator
            * density_matrix
            * propagator.dag()
        )

    return average_density / realizations


def quasistatic_coherence(time: float, sigma: float) -> float:
    """Return Gaussian Ramsey coherence for a constant random detuning."""
    if not np.isfinite(time) or time < 0:
        raise ValueError("time must be finite and non-negative.")

    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and non-negative.")

    return float(np.exp(-0.5 * (sigma * time) ** 2))


def quasistatic_dephasing_channel(
    density_matrix: Qobj,
    time: float,
    sigma: float,
) -> Qobj:
    """Average single-qubit evolution over zero-mean Gaussian static detuning."""
    if density_matrix.shape != (2, 2):
        raise ValueError(
            "quasistatic_dephasing_channel requires a single qubit."
        )

    coherence = quasistatic_coherence(time=time, sigma=sigma)

    identity = qeye(2)
    sigma_z = sigmaz()

    return (
        (1 + coherence) / 2 * density_matrix
        + (1 - coherence) / 2
        * sigma_z * density_matrix * sigma_z
    )


def stochastic_noise_hamiltonian(
    time: float,
    operators,
    amplitudes,
) -> Qobj:
    """Construct a stochastic Hamiltonian from supplied real noise coefficients."""
    return control_hamiltonian(
        time=time,
        operators=operators,
        amplitudes=amplitudes,
    )


def sampled_noise_propagator(
    times,
    coefficients,
    operators,
) -> Qobj:
    """Propagate piecewise-constant noise coefficients in chronological order."""
    if np.iscomplexobj(times):
        raise ValueError("times must be real.")

    times = np.asarray(times, dtype=float)

    if (
        times.ndim != 1
        or len(times) < 2
        or not np.all(np.isfinite(times))
    ):
        raise ValueError("times must contain at least two finite values.")

    durations = np.diff(times)

    if np.any(durations <= 0):
        raise ValueError("times must be strictly increasing.")

    operators = list(operators)

    if not operators:
        raise ValueError("At least one noise operator is required.")

    if np.iscomplexobj(coefficients):
        raise ValueError("coefficients must be real.")

    coefficients = np.asarray(coefficients, dtype=float)

    expected_shape = (len(durations), len(operators))

    if coefficients.shape != expected_shape:
        raise ValueError(
            f"coefficients must have shape {expected_shape}."
        )

    if not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients must be finite.")

    propagator = None

    for index, duration in enumerate(durations):
        hamiltonian = stochastic_noise_hamiltonian(
            time=times[index],
            operators=operators,
            amplitudes=coefficients[index],
        )

        if propagator is None:
            propagator = qeye(hamiltonian.dims[0])

        propagator = (
            (-1j * hamiltonian * duration).expm()
            * propagator
        )

    return propagator


def sampled_noise_ensemble_map(
    density_matrix: Qobj,
    times,
    coefficients,
    operators,
) -> Qobj:
    """Average evolution over equally weighted sampled noise realizations."""
    if np.iscomplexobj(coefficients):
        raise ValueError("coefficients must be real.")

    coefficients = np.asarray(coefficients, dtype=float)

    if coefficients.ndim != 3 or coefficients.shape[0] == 0:
        raise ValueError(
            "coefficients must have shape "
            "(realizations, intervals, operators), "
            "with at least one realization."
        )

    operators = list(operators)
    result = 0 * density_matrix

    for realization in coefficients:
        propagator = sampled_noise_propagator(
            times=times,
            coefficients=realization,
            operators=operators,
        )

        if density_matrix.dims != propagator.dims:
            raise ValueError(
                "Input operator dimensions must match the noise system."
            )

        result += (
            propagator
            * density_matrix
            * propagator.dag()
        )

    return result / len(coefficients)
