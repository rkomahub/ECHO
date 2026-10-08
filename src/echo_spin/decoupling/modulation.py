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


def effective_zz_coupling(
    coupling: float,
    total_duration: float,
    pulse_times_i: Sequence[float],
    pulse_times_j: Sequence[float],
) -> float:
    """Return the time-averaged ZZ coefficient under ideal pi pulses.

    The input and output use the same coupling convention and units.
    Pulse times must lie within [0, total_duration].
    """
    if not np.isfinite(coupling):
        raise ValueError("coupling must be finite.")

    if not np.isfinite(total_duration) or total_duration <= 0:
        raise ValueError("total_duration must be finite and positive.")

    # Reuse the existing schedule validation before merging pulse times.
    modulation_function([0.0], pulse_times_i)
    modulation_function([0.0], pulse_times_j)

    pulses_i = np.asarray(pulse_times_i, dtype=float)
    pulses_j = np.asarray(pulse_times_j, dtype=float)

    if (
        np.any(pulses_i > total_duration)
        or np.any(pulses_j > total_duration)
    ):
        raise ValueError("Pulse times must lie within the sequence.")

    boundaries = np.unique(
        np.concatenate(([0.0, total_duration], pulses_i, pulses_j))
    )
    durations = np.diff(boundaries)
    midpoints = boundaries[:-1] + durations / 2

    signs_i = modulation_function(midpoints, pulses_i)
    signs_j = modulation_function(midpoints, pulses_j)

    overlap = float((signs_i * signs_j) @ durations)
    return float(coupling * overlap / total_duration)
