import numpy as np


def longitudinal_qubit_parameters(
    bare_detunings,
    dipolar_tensors,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return detunings, ZZ coefficients and constant for the electronic 0/-1 encoding."""
    if (
        np.iscomplexobj(bare_detunings)
        or np.iscomplexobj(dipolar_tensors)
    ):
        raise ValueError("Detunings and tensors must be real.")

    detunings = np.asarray(bare_detunings, dtype=float)
    tensors = np.asarray(dipolar_tensors, dtype=float)

    if (
        detunings.ndim != 1
        or detunings.size == 0
        or not np.all(np.isfinite(detunings))
    ):
        raise ValueError("bare_detunings must be a finite nonempty 1D array.")

    number_of_nv = detunings.size
    if (
        tensors.shape != (number_of_nv, number_of_nv, 3, 3)
        or not np.all(np.isfinite(tensors))
    ):
        raise ValueError("dipolar_tensors must be finite with shape (N, N, 3, 3).")

    if not np.allclose(
        tensors,
        tensors.transpose(1, 0, 3, 2),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError("Pair tensors must satisfy K_ji = K_ij.T.")

    if np.any(tensors[np.arange(number_of_nv), np.arange(number_of_nv)] != 0):
        raise ValueError("Self-interaction tensors must vanish.")

    longitudinal = tensors[:, :, 2, 2]
    couplings = longitudinal / 4
    corrected_detunings = detunings - longitudinal.sum(axis=1) / 2
    constant = float(np.triu(couplings, k=1).sum())

    return corrected_detunings, couplings, constant
