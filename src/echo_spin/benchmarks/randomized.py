from functools import lru_cache

import numpy as np
from qutip import Qobj, qeye

from echo_spin.control.rotations import single_qubit_rotation
from echo_spin.gates.gates import sqrt_zz_gate


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


@lru_cache(maxsize=1)
def two_qubit_clifford_catalog() -> dict:
    """Build all two-qubit Cliffords modulo global phase and retain chronological generator words."""
    identity = np.eye(2, dtype=complex)
    rx = single_qubit_rotation(np.pi / 2, "x").full()
    ry = single_qubit_rotation(np.pi / 2, "y").full()

    generators = np.array([
        np.kron(rx, identity),
        np.kron(ry, identity),
        np.kron(identity, rx),
        np.kron(identity, ry),
        sqrt_zz_gate().dag().full(),
    ])

    paulis = [
        identity,
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.diag([1, -1]).astype(complex),
    ]
    pauli_basis = np.array([
        np.kron(first, second)
        for first in paulis
        for second in paulis
    ])

    # X1, Z1, X2, Z2 uniquely determine a Clifford modulo global phase.
    defining_paulis = pauli_basis[[4, 12, 1, 3]]

    def signature(unitary):
        """Identify a Clifford through its four signed generator-Pauli images."""
        transformed = (
            unitary @ defining_paulis
            @ unitary.conj().T
        )
        overlaps = np.einsum(
            "aij,bji->ab",
            pauli_basis,
            transformed,
        ).real / 4

        positions = np.argmax(np.abs(overlaps), axis=0)
        selected = overlaps[positions, np.arange(4)]

        if not np.allclose(
            np.abs(selected), 1.0,
            atol=1e-10, rtol=0.0,
        ):
            raise RuntimeError("Generated operator is not Clifford.")

        return tuple(
            int(position + 1) * (1 if value > 0 else -1)
            for position, value in zip(positions, selected)
        )

    matrices = [np.eye(4, dtype=complex)]
    words = [()]
    known = {signature(matrices[0])}
    next_index = 0

    while next_index < len(matrices):
        current = matrices[next_index]
        word = words[next_index]
        next_index += 1

        for generator_index, generator in enumerate(generators):
            candidate = generator @ current
            key = signature(candidate)

            if key not in known:
                known.add(key)
                matrices.append(candidate)
                words.append(word + (generator_index,))

                if len(matrices) > 11520:
                    raise RuntimeError("Two-qubit Clifford generation did not close.")

    if len(matrices) != 11520:
        raise RuntimeError("Expected 11520 two-qubit Cliffords.")

    matrices = np.array(matrices)
    matrices.setflags(write=False)
    generators.setflags(write=False)

    return {
        "matrices": matrices,
        "words": tuple(words),
        "generators": generators,
        "generator_labels": (
            "RX1_pi_over_2",
            "RY1_pi_over_2",
            "RX2_pi_over_2",
            "RY2_pi_over_2",
            "sqrt_ZZ_dagger",
        ),
    }


def random_two_qubit_clifford_sequence(length, rng) -> dict:
    """Sample uniform two-qubit Cliffords with generator words and ideal recovery."""
    if (
        isinstance(length, (bool, np.bool_))
        or not isinstance(length, (int, np.integer))
        or length < 0
    ):
        raise ValueError("length must be a non-negative integer.")

    catalog = two_qubit_clifford_catalog()
    indices = rng.integers(
        0, len(catalog["matrices"]), size=length
    )

    gates = tuple(
        Qobj(
            catalog["matrices"][int(index)],
            dims=[[2, 2], [2, 2]],
        )
        for index in indices
    )

    total = qeye([2, 2])

    for gate in gates:
        total = gate * total

    recovery_matrix = total.dag().full()

    # Identify recovery modulo global phase in the complete Clifford catalogue.
    overlaps = np.einsum(
        "nij,ij->n",
        catalog["matrices"].conj(),
        recovery_matrix,
    ) / 4

    recovery_index = int(np.argmax(np.abs(overlaps)))

    if not np.isclose(
        abs(overlaps[recovery_index]),
        1.0,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError("Recovery was not found in the Clifford catalogue.")

    return {
        "indices": indices,
        "gates": gates,
        "words": tuple(
            catalog["words"][int(index)]
            for index in indices
        ),
        "recovery": total.dag(),
        "recovery_index": recovery_index,
        "recovery_word": catalog["words"][recovery_index],
    }


def two_qubit_native_gate_counts(word) -> tuple:
    """Count single-qubit and entangling primitives in an unoptimized generator word."""
    word = tuple(word)

    for index in word:
        if (
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            or not 0 <= index < 5
        ):
            raise ValueError("Generator indices must be integers from zero to four.")

    return (
        sum(index < 4 for index in word),
        sum(index == 4 for index in word),
    )


def two_qubit_rb_signal(
    initial_density,
    lengths,
    sequences_per_length,
    rng,
    primitive_map,
    readout,
) -> dict:
    """Average compiled two-qubit RB circuits with the same primitive map during recovery."""
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

    shape = (len(lengths), sequences_per_length)
    samples = np.empty(shape)
    gate_counts = np.zeros(shape + (2,), dtype=int)
    recovery_counts = np.zeros(shape + (2,), dtype=int)
    sampled_indices = []

    for row, length in enumerate(lengths):
        row_indices = []

        for column in range(sequences_per_length):
            sequence = random_two_qubit_clifford_sequence(
                int(length),
                rng,
            )
            row_indices.append(
                tuple(int(index) for index in sequence["indices"])
            )
            state = initial_density.copy()

            for word in sequence["words"]:
                gate_counts[row, column] += (
                    two_qubit_native_gate_counts(word)
                )

                for generator_index in word:
                    state = primitive_map(state, generator_index)

                    if state.dims != initial_density.dims:
                        raise ValueError("primitive_map must preserve dimensions.")

            recovery_word = sequence["recovery_word"]
            recovery_counts[row, column] = (
                two_qubit_native_gate_counts(recovery_word)
            )

            for generator_index in recovery_word:
                state = primitive_map(state, generator_index)

                if state.dims != initial_density.dims:
                    raise ValueError("primitive_map must preserve dimensions.")

            value = complex(readout(state))

            if not np.isfinite(value) or abs(value.imag) > 1e-10:
                raise ValueError("readout must return a finite real value.")

            samples[row, column] = value.real

        sampled_indices.append(tuple(row_indices))

    return {
        "lengths": lengths.copy(),
        "samples": samples,
        "mean": samples.mean(axis=1),
        "standard_error": (
            samples.std(axis=1, ddof=1)
            / np.sqrt(sequences_per_length)
        ),
        "gate_counts": gate_counts,
        "recovery_counts": recovery_counts,
        "clifford_indices": tuple(sampled_indices),
    }
