from itertools import product

import numpy as np
from qutip import Qobj, qeye, sigmax, sigmay, sigmaz, tensor


def two_qubit_hamiltonian_reduction(
    hamiltonian: Qobj,
    logical_basis,
) -> dict:
    """Project a Hamiltonian onto two logical qubits and diagnose discarded coupling."""
    matrix = hamiltonian.full()

    if (
        not hamiltonian.isoper
        or not np.all(np.isfinite(matrix))
        or not np.allclose(
            matrix,
            matrix.conj().T,
            atol=1e-10,
            rtol=1e-12,
        )
    ):
        raise ValueError("hamiltonian must be a finite Hermitian operator.")

    states = list(logical_basis)

    if len(states) != 4:
        raise ValueError("Provide four states ordered as 00, 01, 10, 11.")

    for state in states:
        if (
            not state.isket
            or state.dims[0] != hamiltonian.dims[0]
        ):
            raise ValueError("Logical states must be matching joint-system kets.")

    embedding = np.column_stack([
        state.full().ravel()
        for state in states
    ])

    if (
        not np.all(np.isfinite(embedding))
        or not np.allclose(
            embedding.conj().T @ embedding,
            np.eye(4),
            atol=1e-10,
            rtol=0.0,
        )
    ):
        raise ValueError("Logical states must be orthonormal.")

    reduced = embedding.conj().T @ matrix @ embedding

    paulis = {
        "I": qeye(2),
        "X": sigmax(),
        "Y": sigmay(),
        "Z": sigmaz(),
    }

    coefficients = {}

    for first, second in product(paulis, repeat=2):
        operator = tensor(paulis[first], paulis[second]).full()
        coefficients[first + second] = float(
            np.trace(operator @ reduced).real / 4
        )

    # This is Q H V, with Q projecting outside the logical subspace.
    discarded_coupling = matrix @ embedding - embedding @ reduced

    return {
        "hamiltonian": Qobj(reduced, dims=[[2, 2], [2, 2]]),
        "pauli_coefficients": coefficients,
        "discarded_coupling_norm": float(
            np.linalg.norm(discarded_coupling, ord=2)
        ),
    }
