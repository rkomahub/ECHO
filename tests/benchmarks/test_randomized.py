import numpy as np
import pytest
from qutip import basis, qeye, sigmax, sigmay, sigmaz

from echo_spin.benchmarks.randomized import (
    random_single_qubit_clifford_sequence,
    rb_error_per_clifford,
    single_qubit_cliffords,
    single_qubit_rb_signal,
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
