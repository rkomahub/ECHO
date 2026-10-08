from typing import Sequence

import numpy as np


def modulation_function(
    times: Sequence[float],
    pulse_times: Sequence[float],
) -> np.ndarray:
    """Return longitudinal sign modulation from ideal transverse pi pulses.

    Times are measured from sequence start and use the same units.
    Pulse times must be strictly increasing and non-negative.
    At a pulse time, the returned sign is the post-pulse value.
    An empty pulse schedule gives y(t) = +1.
    """
    if np.iscomplexobj(times) or np.iscomplexobj(pulse_times):
        raise ValueError("Times must be real.")

    samples = np.asarray(times, dtype=float)
    pulses = np.asarray(pulse_times, dtype=float)

    if (
        samples.ndim != 1
        or not np.all(np.isfinite(samples))
        or np.any(samples < 0)
    ):
        raise ValueError("times must be a finite non-negative 1D array.")

    if (
        pulses.ndim != 1
        or not np.all(np.isfinite(pulses))
        or np.any(pulses < 0)
        or np.any(np.diff(pulses) <= 0)
    ):
        raise ValueError(
            "pulse_times must be finite, non-negative and strictly increasing."
        )

    pulse_counts = np.searchsorted(pulses, samples, side="right")
    return 1 - 2 * (pulse_counts % 2)
