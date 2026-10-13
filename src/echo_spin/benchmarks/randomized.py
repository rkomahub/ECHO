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


def single_qubit_rb_signal(
    initial_density,
    lengths,
    sequences_per_length,
    rng,
    gate_map,
    recovery_map,
    readout,
) -> dict:
    """Average single-qubit Clifford sequences with explicit gate, recovery and readout maps."""
    lengths = np.asarray(lengths)

    if (
        lengths.ndim != 1
        or lengths.size == 0
        or lengths.dtype.kind not in "iu"
        or np.any(lengths < 0)
    ):
        raise ValueError("lengths must be non-negative integers.")

    if (
        isinstance(sequences_per_length, (bool, np.bool_))
        or not isinstance(sequences_per_length, (int, np.integer))
        or sequences_per_length < 2
    ):
        raise ValueError("Use at least two sequences per length.")

    cliffords = single_qubit_cliffords()
    samples = np.empty((len(lengths), sequences_per_length))

    for row, length in enumerate(lengths):
        for column in range(sequences_per_length):
            state = initial_density.copy()
            ideal_total = qeye(2)

            indices = rng.integers(
                0, len(cliffords), size=int(length)
            )

            for index in indices:
                clifford = cliffords[int(index)]
                state = gate_map(state, clifford)
                ideal_total = clifford * ideal_total

                if state.dims != initial_density.dims:
                    raise ValueError("gate_map must preserve state dimensions.")

            state = recovery_map(state, ideal_total.dag())

            if state.dims != initial_density.dims:
                raise ValueError("recovery_map must preserve state dimensions.")

            value = complex(readout(state))

            if not np.isfinite(value) or abs(value.imag) > 1e-10:
                raise ValueError("readout must return a finite real value.")

            samples[row, column] = value.real

    return {
        "lengths": lengths.copy(),
        "samples": samples,
        "mean": samples.mean(axis=1),
        "standard_error": (
            samples.std(axis=1, ddof=1)
            / np.sqrt(sequences_per_length)
        ),
    }


def rb_error_per_clifford(decay_parameter: float, qubits: int) -> float:
    """Convert an RB decay parameter into error per Clifford for the stated dimension."""
    if (
        not np.isfinite(decay_parameter)
        or not 0 <= decay_parameter <= 1
    ):
        raise ValueError("decay_parameter must lie between zero and one.")

    if (
        isinstance(qubits, (bool, np.bool_))
        or not isinstance(qubits, (int, np.integer))
        or qubits < 1
    ):
        raise ValueError("qubits must be a positive integer.")

    return float(
        (1 - 2.0 ** (-int(qubits)))
        * (1 - decay_parameter)
    )
