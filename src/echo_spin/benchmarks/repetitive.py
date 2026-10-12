import numpy as np
from qutip import Qobj


def repetitive_benchmark_signal(
    initial_density: Qobj,
    gate_map,
    repetitions,
    readout,
) -> np.ndarray:
    """Repeat a state map without resetting its environment and evaluate count-dependent readout."""
    counts = np.asarray(repetitions)

    if (
        counts.ndim != 1
        or counts.size == 0
        or counts.dtype.kind not in "iu"
        or np.any(counts < 0)
    ):
        raise ValueError(
            "repetitions must be a nonempty sequence of non-negative integers."
        )

    requested = {int(count) for count in counts}
    measurements = {}
    state = initial_density.copy()

    for count in range(int(counts.max()) + 1):
        if count in requested:
            value = complex(readout(state, count))

            if (
                not np.isfinite(value)
                or abs(value.imag) > 1e-10
            ):
                raise ValueError("readout must return a finite real value.")

            measurements[count] = value.real

        if count < int(counts.max()):
            state = gate_map(state)

            if (
                not isinstance(state, Qobj)
                or state.dims != initial_density.dims
            ):
                raise ValueError(
                    "gate_map must preserve the input operator dimensions."
                )

    return np.array([
        measurements[int(count)] for count in counts
    ])
