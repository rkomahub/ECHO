import numpy as np

from echo_spin.benchmarks.randomized import (
    two_qubit_clifford_catalog,
    two_qubit_native_gate_counts,
)

catalog = two_qubit_clifford_catalog()
counts = np.array([
    two_qubit_native_gate_counts(word)
    for word in catalog["words"]
])

print("Cliffords =", len(counts))
print("Mean single-qubit primitives =", counts[:, 0].mean())
print("Mean entangling primitives =", counts[:, 1].mean())
print("Maximum total primitives =", counts.sum(axis=1).max())
