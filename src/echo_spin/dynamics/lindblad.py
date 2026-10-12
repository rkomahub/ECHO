import numpy as np
from qutip import (
    Qobj,
    liouvillian,
    operator_to_vector,
    vector_to_operator,
)


def static_lindblad_map(
    density_matrix: Qobj,
    hamiltonian: Qobj,
    time: float,
    collapse_operators,
) -> Qobj:
    """Evolve an operator under a static Lindblad generator without renormalizing."""
    if not np.isfinite(time) or time < 0:
        raise ValueError("time must be finite and non-negative.")

    if not hamiltonian.isoper or not hamiltonian.isherm:
        raise ValueError("hamiltonian must be a Hermitian operator.")

    if not np.all(np.isfinite(hamiltonian.full())):
        raise ValueError("hamiltonian must contain finite entries.")

    if (
        not density_matrix.isoper
        or density_matrix.dims != hamiltonian.dims
    ):
        raise ValueError(
            "Input operator dimensions must match the Hamiltonian."
        )

    collapse_operators = list(collapse_operators)

    for operator in collapse_operators:
        if not operator.isoper or operator.dims != hamiltonian.dims:
            raise ValueError(
                "Collapse operators must match the Hamiltonian dimensions."
            )
        if not np.all(np.isfinite(operator.full())):
            raise ValueError("Collapse operators must contain finite entries.")

    generator = liouvillian(
        hamiltonian,
        collapse_operators,
    )

    evolved_vector = (
        (time * generator).expm()
        * operator_to_vector(density_matrix)
    )

    return vector_to_operator(evolved_vector)
