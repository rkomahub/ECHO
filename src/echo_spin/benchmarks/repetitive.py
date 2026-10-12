import numpy as np
from qutip import Qobj
from scipy.optimize import curve_fit


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


def fit_repetitive_decay(
    repetitions,
    signal,
    standard_errors=None,
) -> dict:
    """Fit offset-plus-exponential repetitive decay and report its pseudo-error."""
    counts = np.asarray(repetitions)

    if (
        counts.ndim != 1
        or counts.dtype.kind not in "iu"
        or np.any(counts < 0)
        or len(np.unique(counts)) < 4
    ):
        raise ValueError(
            "Provide at least four distinct non-negative integer counts."
        )

    if np.iscomplexobj(signal):
        raise ValueError("signal must be real.")

    values = np.asarray(signal, dtype=float)

    if values.shape != counts.shape or not np.all(np.isfinite(values)):
        raise ValueError("signal must contain one finite value per count.")

    if np.ptp(values) == 0:
        raise ValueError("A constant signal does not identify a decay parameter.")

    sigma = None

    if standard_errors is not None:
        if np.iscomplexobj(standard_errors):
            raise ValueError("standard_errors must be real.")

        sigma = np.asarray(standard_errors, dtype=float)

        if (
            sigma.shape != counts.shape
            or not np.all(np.isfinite(sigma))
            or np.any(sigma <= 0)
        ):
            raise ValueError(
                "standard_errors must contain one finite positive value per count."
            )

    def decay_model(count, offset, amplitude, decay_parameter):
        """Return the offset-plus-exponential benchmarking signal."""
        return offset + amplitude * decay_parameter**count

    first = int(np.argmin(counts))
    last = int(np.argmax(counts))

    parameters, covariance = curve_fit(
        decay_model,
        counts,
        values,
        p0=[
            values[last],
            values[first] - values[last],
            0.95,
        ],
        bounds=(
            [-np.inf, -np.inf, 0.0],
            [np.inf, np.inf, 1.0],
        ),
        sigma=sigma,
        absolute_sigma=sigma is not None,
        maxfev=20000,
    )

    offset, amplitude, decay_parameter = parameters
    fitted = decay_model(counts, *parameters)
    residuals = values - fitted

    return {
        "offset": float(offset),
        "amplitude": float(amplitude),
        "decay_parameter": float(decay_parameter),
        "pseudo_error_per_gate": float(1 - decay_parameter),
        "residual_rms": float(np.sqrt(np.mean(residuals**2))),
        "fitted_signal": fitted,
        "residuals": residuals,
        "parameter_covariance": covariance,
    }
