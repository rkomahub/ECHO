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


def projected_unitary_kraus(
    propagator: Qobj,
    embeddings,
    environment_probabilities,
    logical_dimensions=(2, 2),
) -> tuple:
    """Extract a logical channel for a specified initially uncorrelated environment."""
    matrices = [np.asarray(value, dtype=complex) for value in embeddings]
    probabilities = np.asarray(environment_probabilities, dtype=float)
    dimensions = list(logical_dimensions)
    logical_dimension = int(np.prod(dimensions))
    unitary = propagator.full()

    if (
        not propagator.isoper
        or not np.all(np.isfinite(unitary))
        or np.linalg.norm(
            unitary.conj().T @ unitary - np.eye(unitary.shape[0]),
            ord="fro",
        ) > 1e-8
    ):
        raise ValueError("propagator must be a finite unitary operator.")

    if not matrices or any(
        matrix.shape != (unitary.shape[0], logical_dimension)
        or not np.all(np.isfinite(matrix))
        for matrix in matrices
    ):
        raise ValueError("Provide matching finite logical embedding matrices.")

    combined = np.column_stack(matrices)

    if not np.allclose(
        combined.conj().T @ combined,
        np.eye(combined.shape[1]),
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError("The combined embeddings must be orthonormal.")

    if (
        probabilities.shape != (len(matrices),)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or not np.isclose(
            probabilities.sum(), 1.0,
            atol=1e-12, rtol=0.0,
        )
    ):
        raise ValueError("Environment probabilities must be normalized.")

    return tuple(
        Qobj(
            np.sqrt(probability)
            * output.conj().T @ unitary @ initial,
            dims=[dimensions, dimensions],
        )
        for initial, probability in zip(matrices, probabilities)
        for output in matrices
    )


def conditional_two_qubit_hamiltonians(
    hamiltonian: Qobj,
    embeddings,
) -> dict:
    """Extract sector-conditioned Hamiltonians and diagnose omitted couplings."""
    matrices = [np.asarray(value, dtype=complex) for value in embeddings]

    if not matrices or any(
        matrix.shape != (hamiltonian.shape[0], 4)
        or not np.all(np.isfinite(matrix))
        for matrix in matrices
    ):
        raise ValueError("Provide finite embedding matrices with four columns.")

    combined = np.column_stack(matrices)

    if not np.allclose(
        combined.conj().T @ combined,
        np.eye(combined.shape[1]),
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError("The combined embeddings must be orthonormal.")

    reductions = []

    for matrix in matrices:
        states = [
            Qobj(
                matrix[:, column:column + 1],
                dims=[
                    hamiltonian.dims[0],
                    [1] * len(hamiltonian.dims[0]),
                ],
            )
            for column in range(4)
        ]

        reductions.append(
            two_qubit_hamiltonian_reduction(hamiltonian, states)
        )

    full = hamiltonian.full()
    projected = combined.conj().T @ full @ combined
    block_diagonal = np.zeros_like(projected)

    for index, reduction in enumerate(reductions):
        block = slice(4 * index, 4 * (index + 1))
        block_diagonal[block, block] = reduction["hamiltonian"].full()

    return {
        "hamiltonians": tuple(
            reduction["hamiltonian"]
            for reduction in reductions
        ),
        "pauli_coefficients": tuple(
            reduction["pauli_coefficients"]
            for reduction in reductions
        ),
        "sector_mixing_norm": float(
            np.linalg.norm(projected - block_diagonal, ord=2)
        ),
        "discarded_coupling_norm": float(
            np.linalg.norm(
                full @ combined - combined @ projected,
                ord=2,
            )
        ),
    }
