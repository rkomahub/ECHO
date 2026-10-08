from typing import Sequence

import numpy as np


def lorentzian_spectrum(
    frequencies: Sequence[float],
    transitions: Sequence[tuple],
    linewidth: float,
) -> np.ndarray:
    """Broaden (frequency, strength, i, j) transitions into Lorentzian lines.

    linewidth is the positive half width at half maximum.
    Frequencies, transition centers and linewidth must use the same units.
    Each isolated line has peak height equal to its transition strength.
    Population differences and fluorescence contrast are not included.
    """
    if not np.isfinite(linewidth) or linewidth <= 0:
        raise ValueError("linewidth must be finite and positive.")

    if np.iscomplexobj(frequencies):
        raise ValueError("frequencies must be real.")

    frequencies = np.asarray(frequencies, dtype=float)

    if frequencies.ndim != 1 or not np.all(np.isfinite(frequencies)):
        raise ValueError("frequencies must be a finite 1D array.")

    spectrum = np.zeros_like(frequencies)

    for center, strength, _, _ in transitions:
        if (
            not np.isfinite(center)
            or center < 0
            or not np.isfinite(strength)
            or strength < 0
        ):
            raise ValueError(
                "Transition frequencies and strengths must be finite "
                "and non-negative."
            )

        detuning = (frequencies - center) / linewidth
        spectrum += strength / (1 + detuning**2)

    return spectrum
