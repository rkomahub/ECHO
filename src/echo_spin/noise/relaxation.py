import numpy as np
from qutip import Qobj, basis

from echo_spin.noise.dephasing import dephasing_channel


def amplitude_damping_channel(
    density_matrix: Qobj,
    time: float,
    relaxation_time: float,
) -> Qobj:
    """Apply qubit relaxation from logical excited |1> to ground |0>."""
    if density_matrix.dims != [[2], [2]]:
        raise ValueError("Amplitude damping requires a single-qubit operator.")

    if not np.isfinite(time) or time < 0:
        raise ValueError("time must be finite and non-negative.")

    if not np.isfinite(relaxation_time) or relaxation_time <= 0:
        raise ValueError("relaxation_time must be finite and positive.")

    survival = np.exp(-time / relaxation_time)
    decay_probability = -np.expm1(-time / relaxation_time)

    ground = basis(2, 0)
    excited = basis(2, 1)

    k_zero = (
        ground.proj()
        + np.sqrt(survival) * excited.proj()
    )
    k_one = (
        np.sqrt(decay_probability)
        * ground * excited.dag()
    )

    return (
        k_zero * density_matrix * k_zero.dag()
        + k_one * density_matrix * k_one.dag()
    )


def relaxation_dephasing_channel(
    density_matrix: Qobj,
    time: float,
    relaxation_time: float,
    pure_dephasing_time: float,
) -> Qobj:
    """Combine qubit relaxation with an independent pure-dephasing channel."""
    relaxed = amplitude_damping_channel(
        density_matrix=density_matrix,
        time=time,
        relaxation_time=relaxation_time,
    )

    return dephasing_channel(
        density_matrix=relaxed,
        time=time,
        coherence_time=pure_dephasing_time,
    )
