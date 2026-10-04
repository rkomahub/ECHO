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


def sine_pi_pulse_amplitude(duration: float) -> float:
    """Return the peak amplitude of a sine-shaped pi pulse."""
    if duration <= 0:
        raise ValueError("duration must be positive.")

    rabi_frequency = np.pi / duration
    return (np.pi / 2) * rabi_frequency


def centered_sine_envelope(
    time: float,
    center: float,
    duration: float,
    peak_amplitude: float,
) -> float:
    """Return a sine pulse centered at an absolute time."""
    local_time = time - center + duration / 2

    return sine_envelope(
        time=local_time,
        duration=duration,
        peak_amplitude=peak_amplitude,
    )