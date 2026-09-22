import numpy as np
from qutip import Qobj, jmat, basis
from itertools import permutations


def diagonalize_hamiltonian(
    hamiltonian: Qobj,
) -> tuple[np.ndarray, list[Qobj]]:
    """Return eigenvalues and eigenstates ordered by increasing energy."""
    eigenvalues, eigenstates = hamiltonian.eigenstates()

    order = np.argsort(eigenvalues)

    eigenvalues = np.asarray(eigenvalues)[order]
    eigenstates = [eigenstates[i] for i in order]

    return eigenvalues, eigenstates


def basis_unitary(
    basis_states: list[Qobj],
) -> Qobj:
    """Construct the unitary matrix whose columns are the given basis states."""
    if not basis_states:
        raise ValueError("At least one basis state must be provided.")

    dimension = basis_states[0].shape[0]

    if len(basis_states) != dimension:
        raise ValueError(
            "The number of basis states must equal the Hilbert-space dimension."
        )

    matrix = np.column_stack([
        state.full().ravel()
        for state in basis_states
    ])

    if not np.allclose(
        matrix.conj().T @ matrix,
        np.eye(dimension),
    ):
        raise ValueError("The basis states must form an orthonormal basis.")

    dims = basis_states[0].dims[0]

    return Qobj(
        matrix,
        dims=[dims, dims],
    )


def dressed_spin_operators(
    basis_states: list[Qobj],
    spin: float,
) -> tuple[Qobj, Qobj, Qobj]:
    """Construct spin operators associated with a chosen dressed basis.

    The basis states must be ordered according to the usual angular-momentum
    convention m = s, s-1, ..., -s.
    """
    dimension = int(2 * spin + 1)

    if len(basis_states) != dimension:
        raise ValueError(
            "The number of basis states does not match the spin dimension."
        )

    unitary = basis_unitary(basis_states)

    sx = jmat(spin, "x")
    sy = jmat(spin, "y")
    sz = jmat(spin, "z")

    return (
        unitary * sx * unitary.dag(),
        unitary * sy * unitary.dag(),
        unitary * sz * unitary.dag(),
    )


def order_states_by_reference(
    states: list[Qobj],
    reference_states: list[Qobj],
) -> list[Qobj]:
    """Order eigenstates by maximum overlap with a reference basis.

    The returned states follow the ordering of reference_states.
    """
    if len(states) != len(reference_states):
        raise ValueError(
            "states and reference_states must have the same length."
        )

    number_of_states = len(states)

    best_score = -1.0
    best_order = None

    for permutation in permutations(range(number_of_states)):
        score = 0.0

        for reference_index, state_index in enumerate(permutation):
            overlap = reference_states[reference_index].overlap(
                states[state_index]
            )

            score += abs(overlap) ** 2

        if score > best_score:
            best_score = score
            best_order = permutation

    return [
        states[index]
        for index in best_order
    ]


def dressed_spin_one_basis(
    hamiltonian: Qobj,
) -> list[Qobj]:
    """Return spin-1 eigenstates ordered as +1, 0, -1.

    The Hamiltonian eigenstates are matched to the bare spin-1 basis
    by maximizing their total overlap.
    """
    _, eigenstates = diagonalize_hamiltonian(hamiltonian)

    reference_states = [
        basis(3, 0),  # |+1>
        basis(3, 1),  # |0>
        basis(3, 2),  # |-1>
    ]

    return order_states_by_reference(
        states=eigenstates,
        reference_states=reference_states,
    )