import numpy as np
from qutip import Qobj, tensor
from scipy.optimize import curve_fit

from echo_spin.core.basis import project_operator


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


def logical_repetitive_return_signal(
    initial_density: Qobj,
    propagator: Qobj,
    logical_basis,
    nuclear_states,
    target: Qobj,
    repetitions,
    logical_readout=None,
) -> np.ndarray:
    """Measure ideal-recovery logical return while retaining nuclei across repeated gates."""
    logical_basis = list(logical_basis)
    nuclear_states = list(nuclear_states)

    if len(nuclear_states) != 2:
        raise ValueError("Provide two nuclear density matrices.")

    dimension = initial_density.shape[0]

    if len(logical_basis) != dimension:
        raise ValueError("logical_basis must match the logical input dimension.")

    if target.dims != initial_density.dims or not target.isunitary:
        raise ValueError("target must be a matching unitary logical gate.")

    isometry = np.column_stack([
        state.full().ravel()
        for state in logical_basis
    ])

    if not np.allclose(
        isometry.conj().T @ isometry,
        np.eye(dimension),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("logical_basis must be orthonormal.")

    physical_density = Qobj(
        isometry @ initial_density.full() @ isometry.conj().T,
        dims=logical_basis[0].proj().dims,
    )

    joint_density = tensor(
        physical_density,
        nuclear_states[0],
        nuclear_states[1],
    ).permute([0, 2, 1, 3])

    if propagator.dims != joint_density.dims:
        raise ValueError(
            "Joint dimensions do not match: "
            f"propagator={propagator.dims}, "
            f"state={joint_density.dims}."
        )

    matrix = propagator.full()
    unitary_defect = np.linalg.norm(
        matrix.conj().T @ matrix
        - np.eye(matrix.shape[0]),
        ord="fro",
    )

    # Explicit tolerance for numerical propagation; do not renormalize.
    if not np.isfinite(unitary_defect) or unitary_defect > 1e-8:
        raise ValueError(
            "Propagator is not sufficiently unitary: "
            f"Frobenius defect={unitary_defect:.3e}."
        )

    def gate_map(density_matrix):
        """Apply another fixed joint gate without resetting the nuclei."""
        return propagator * density_matrix * propagator.dag()

    def readout(density_matrix, count):
        """Trace nuclei and evaluate unnormalized return after ideal logical recovery."""
        reduced = density_matrix.ptrace([0, 2])

        logical_density = project_operator(
            operator=reduced,
            basis_states=logical_basis,
        ).full()

        if logical_readout is not None:
            return logical_readout(
                Qobj(
                    logical_density,
                    dims=initial_density.dims,
                ),
                count,
            )

        recovery = (target.dag() ** count).full()
        recovered = (
            recovery @ logical_density @ recovery.conj().T
        )

        return np.trace(
            initial_density.full() @ recovered
        )

    return repetitive_benchmark_signal(
        initial_density=joint_density,
        gate_map=gate_map,
        repetitions=repetitions,
        readout=readout,
    )
