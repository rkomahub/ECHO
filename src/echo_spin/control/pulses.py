import numpy as np


def sine_envelope(
    time: float,
    duration: float,
    peak_amplitude: float,
) -> float:
    """Return a sine-shaped pulse envelope."""
    if duration <= 0:
        raise ValueError("duration must be positive.")

    if time < 0 or time > duration:
        return 0.0

    return peak_amplitude * np.sin(np.pi * time / duration)