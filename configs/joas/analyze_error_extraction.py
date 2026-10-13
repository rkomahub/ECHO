from echo_spin.benchmarks.randomized import extract_entangling_gate_error

# Rounded experimental quantities from the supplied article text.
entangling_error = extract_entangling_gate_error(
    error_per_clifford=0.149,
    single_qubit_error=1 - 0.9923,
    single_qubit_gates_per_clifford=10.5,
    entangling_gates_per_clifford=1.8,
)

print("Entangling error from rounded article inputs =", entangling_error)
print("Corresponding gate fidelity =", 1 - entangling_error)
