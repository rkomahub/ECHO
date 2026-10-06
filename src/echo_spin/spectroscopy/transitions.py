import numpy as np
from qutip import Qobj

from echo_spin.core.system import SpinSystem
from echo_spin.nv.hamiltonians import electronic_nv_hamiltonian


def transition_frequencies(hamiltonian: Qobj) -> np.ndarray:
    """Return transition frequencies from the ground eigenstate."""
    eigenvalues = np.sort(hamiltonian.eigenenergies())

    return np.asarray(eigenvalues[1:] - eigenvalues[0])


def electronic_spectrum(
    D: float,
    omega_e,
) -> np.ndarray:
    """Return the electronic transition frequencies of a single NV center."""
    system = SpinSystem([1])

    hamiltonian = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=D,
        omega_e=omega_e,
    )

    return transition_frequencies(hamiltonian)


def allowed_transitions(
    hamiltonian: Qobj,
    control_operator: Qobj,
    threshold: float = 1e-10,
) -> list[tuple]:
    """Return transitions coupled by a control operator."""
    eigenvalues, eigenstates = hamiltonian.eigenstates()

    transitions = []

    for i in range(len(eigenstates)):
        for j in range(i + 1, len(eigenstates)):
            matrix_element = (
                eigenstates[j].dag()
                * control_operator
                * eigenstates[i]
            )

            strength = abs(matrix_element) ** 2

            if strength > threshold:
                transitions.append(
                    (
                        eigenvalues[j] - eigenvalues[i],
                        strength,
                        i,
                        j,
                    )
                )

    return transitions
