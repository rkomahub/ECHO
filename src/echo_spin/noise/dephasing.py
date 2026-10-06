import numpy as np
from qutip import Qobj, qeye, sigmaz


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
