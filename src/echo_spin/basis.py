import numpy as np
from qutip import Qobj, jmat


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