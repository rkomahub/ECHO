import numpy as np
import pytest
from qutip import qeye, sigmax, sigmay, sigmaz

from echo_spin.benchmarks.randomized import (
    random_single_qubit_clifford_sequence,
    single_qubit_cliffords,
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
