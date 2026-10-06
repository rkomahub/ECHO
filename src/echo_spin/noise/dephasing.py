import numpy as np


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
