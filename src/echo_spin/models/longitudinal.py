import numpy as np
from qutip import Qobj, qeye, sigmaz

from echo_spin.core.operators import embed_operator
from echo_spin.core.system import SpinSystem


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


def longitudinal_register_hamiltonian(
    detunings,
    couplings,
    constant: float = 0.0,
) -> Qobj:
    """Build the ideal longitudinal qubit Hamiltonian in rad/us and site order."""
    if any(np.iscomplexobj(value) for value in (detunings, couplings, constant)):
        raise ValueError("Hamiltonian coefficients must be real.")

    detunings = np.asarray(detunings, dtype=float)
    couplings = np.asarray(couplings, dtype=float)
    offset = np.asarray(constant, dtype=float)

    if (
        detunings.ndim != 1
        or detunings.size == 0
        or not np.all(np.isfinite(detunings))
    ):
        raise ValueError("detunings must be a finite nonempty 1D array.")

    number_of_qubits = detunings.size
    if (
        couplings.shape != (number_of_qubits, number_of_qubits)
        or not np.all(np.isfinite(couplings))
    ):
        raise ValueError("couplings must be a finite (N, N) array.")

    if not np.allclose(couplings, couplings.T, atol=1e-12, rtol=1e-12):
        raise ValueError("couplings must be symmetric.")

    if np.any(np.diag(couplings) != 0.0):
        raise ValueError("couplings must have zero diagonal.")

    if offset.shape != () or not np.isfinite(offset):
        raise ValueError("constant must be a finite scalar.")

    system = SpinSystem([0.5] * number_of_qubits)
    z_operators = [
        embed_operator(sigmaz(), site, system)
        for site in range(number_of_qubits)
    ]
    hamiltonian = float(offset) * qeye(list(system.dimensions))

    for i in range(number_of_qubits):
        hamiltonian += detunings[i] * z_operators[i] / 2
        for j in range(i + 1, number_of_qubits):
            hamiltonian += couplings[i, j] * z_operators[i] * z_operators[j]

    return hamiltonian
