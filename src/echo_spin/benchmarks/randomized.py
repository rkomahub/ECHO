import numpy as np
from qutip import qeye

from echo_spin.control.rotations import single_qubit_rotation


def single_qubit_cliffords() -> tuple:
    """Generate the 24 single-qubit Clifford representatives modulo global phase."""
    generators = [
        single_qubit_rotation(np.pi / 2, "x"),
        single_qubit_rotation(np.pi / 2, "y"),
    ]

    cliffords = [qeye(2)]
    next_index = 0

    while next_index < len(cliffords):
        current = cliffords[next_index]
        next_index += 1

        for generator in generators:
            candidate = generator * current

            already_present = any(
                np.isclose(
                    abs((existing.dag() * candidate).tr()),
                    2.0,
                    atol=1e-10,
                    rtol=0.0,
                )
                for existing in cliffords
            )

            if not already_present:
                cliffords.append(candidate)

                if len(cliffords) > 24:
                    raise RuntimeError("Clifford generation did not close.")

    if len(cliffords) != 24:
        raise RuntimeError("Expected 24 single-qubit Cliffords.")

    return tuple(cliffords)


def random_single_qubit_clifford_sequence(
    length: int,
    rng: np.random.Generator,
) -> dict:
    """Sample uniform single-qubit Cliffords and compute their chronological recovery."""
    if (
        isinstance(length, (bool, np.bool_))
        or not isinstance(length, (int, np.integer))
        or length < 0
    ):
        raise ValueError("length must be a non-negative integer.")

    cliffords = single_qubit_cliffords()
    indices = rng.integers(0, len(cliffords), size=length)

    gates = tuple(cliffords[int(index)] for index in indices)
    total = qeye(2)

    for gate in gates:
        total = gate * total

    return {
        "indices": indices,
        "gates": gates,
        "recovery": total.dag(),
    }
