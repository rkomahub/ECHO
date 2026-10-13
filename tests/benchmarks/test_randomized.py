import numpy as np
import pytest
from qutip import basis, qeye, sigmax, sigmay, sigmaz

from echo_spin.benchmarks.randomized import (
    random_single_qubit_clifford_sequence,
    random_two_qubit_clifford_sequence,
    rb_error_per_clifford,
    single_qubit_cliffords,
    single_qubit_rb_signal,
    two_qubit_clifford_catalog,
    two_qubit_native_gate_counts,
)


def test_single_qubit_cliffords_map_pauli_axes():
    """Verify that all 24 distinct Clifford representatives permute signed Pauli axes."""
    cliffords = single_qubit_cliffords()
    paulis = [sigmax(), sigmay(), sigmaz()]

    assert len(cliffords) == 24

    for index, clifford in enumerate(cliffords):
        np.testing.assert_allclose(
            (clifford.dag() * clifford).full(),
            np.eye(2),
            atol=1e-12,
            rtol=0.0,
        )

        for earlier in cliffords[:index]:
            assert abs(
                (earlier.dag() * clifford).tr()
            ) < 2 - 1e-8

        for pauli in paulis:
            transformed = clifford * pauli * clifford.dag()

            overlaps = np.array([
                float(np.real((axis * transformed).tr() / 2))
                for axis in paulis
            ])

            np.testing.assert_allclose(
                np.sort(np.abs(overlaps)),
                [0.0, 0.0, 1.0],
                atol=1e-12,
                rtol=0.0,
            )


@pytest.mark.parametrize("length", [0, 1, 10])
def test_random_clifford_sequence_recovery(length):
    """Verify exact ideal recovery after chronological randomized Clifford evolution."""
    sequence = random_single_qubit_clifford_sequence(
        length=length,
        rng=np.random.default_rng(42),
    )

    total = qeye(2)

    for gate in sequence["gates"]:
        total = gate * total

    recovered = sequence["recovery"] * total

    np.testing.assert_allclose(
        recovered.full(),
        np.eye(2),
        atol=1e-12,
        rtol=0.0,
    )
    assert len(sequence["indices"]) == length


def test_single_qubit_rb_matches_depolarizing_reference():
    """Verify randomized survival against exact depolarizing decay with ideal recovery."""
    decay_parameter = 0.97

    def gate_map(density_matrix, clifford):
        """Apply the Clifford followed by a trace-preserving depolarizing channel."""
        evolved = clifford * density_matrix * clifford.dag()
        return (
            decay_parameter * evolved
            + (1 - decay_parameter)
            * evolved.tr() * qeye(2) / 2
        )

    def recovery_map(density_matrix, recovery):
        """Apply ideal recovery without additional noise."""
        return recovery * density_matrix * recovery.dag()

    def readout(density_matrix):
        """Measure computational ground-state return probability."""
        return (
            basis(2, 0).dag()
            * density_matrix
            * basis(2, 0)
        )

    lengths = np.array([0, 1, 5, 15])

    result = single_qubit_rb_signal(
        initial_density=basis(2, 0).proj(),
        lengths=lengths,
        sequences_per_length=8,
        rng=np.random.default_rng(42),
        gate_map=gate_map,
        recovery_map=recovery_map,
        readout=readout,
    )

    expected = (1 + decay_parameter**lengths) / 2

    np.testing.assert_allclose(
        result["samples"],
        np.broadcast_to(expected[:, None], (4, 8)),
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "qubits, expected_error",
    [(1, 0.015), (2, 0.0225)],
)
def test_rb_error_per_clifford_dimension_factor(qubits, expected_error):
    """Verify the dimension-dependent conversion from RB decay to Clifford error."""
    assert rb_error_per_clifford(
        decay_parameter=0.97,
        qubits=qubits,
    ) == pytest.approx(expected_error, abs=1e-12)


def test_two_qubit_clifford_catalog_size_and_unitarity():
    """Verify full Clifford-group size and numerical unitarity of every representative."""
    catalog = two_qubit_clifford_catalog()
    matrices = catalog["matrices"]

    assert matrices.shape == (11520, 4, 4)
    assert len(catalog["words"]) == 11520

    products = (
        matrices.conj().transpose(0, 2, 1)
        @ matrices
    )

    np.testing.assert_allclose(
        products,
        np.broadcast_to(np.eye(4), products.shape),
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize("length", [0, 1, 10])
def test_two_qubit_random_sequence_words_and_recovery(length):
    """Verify chronological native words and ideal recovery of sampled two-qubit Cliffords."""
    catalog = two_qubit_clifford_catalog()
    sequence = random_two_qubit_clifford_sequence(
        length,
        np.random.default_rng(42),
    )

    total = np.eye(4, dtype=complex)

    for gate, word in zip(sequence["gates"], sequence["words"]):
        reconstructed = np.eye(4, dtype=complex)

        for generator_index in word:
            reconstructed = (
                catalog["generators"][generator_index]
                @ reconstructed
            )

        np.testing.assert_allclose(
            reconstructed,
            gate.full(),
            atol=1e-12,
            rtol=0.0,
        )
        total = gate.full() @ total

    np.testing.assert_allclose(
        sequence["recovery"].full() @ total,
        np.eye(4),
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize("length", [0, 1, 10])
def test_two_qubit_recovery_native_word(length):
    """Verify that compiled recovery cancels the ideal sequence modulo global phase."""
    catalog = two_qubit_clifford_catalog()
    sequence = random_two_qubit_clifford_sequence(
        length,
        np.random.default_rng(123),
    )

    total = np.eye(4, dtype=complex)

    for gate in sequence["gates"]:
        total = gate.full() @ total

    recovery = np.eye(4, dtype=complex)

    for index in sequence["recovery_word"]:
        recovery = catalog["generators"][index] @ recovery

    recovered = recovery @ total
    global_phase = np.trace(recovered) / 4

    assert abs(global_phase) == pytest.approx(1.0, abs=1e-12)

    np.testing.assert_allclose(
        recovered,
        global_phase * np.eye(4),
        atol=1e-12,
        rtol=0.0,
    )


def test_two_qubit_native_gate_counts():
    """Verify explicit primitive counts without assuming article compiler averages."""
    assert two_qubit_native_gate_counts(
        [0, 0, 2, 4, 1, 4]
    ) == (4, 2)

    assert two_qubit_native_gate_counts(()) == (0, 0)
