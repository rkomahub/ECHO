from echo_spin.benchmarks.randomized import estimate_entangling_gate_error

result = estimate_entangling_gate_error(
    error_per_clifford=0.149,
    single_qubit_error=1 - 0.9923,
    single_qubit_gates_per_clifford=10.5,
    entangling_gates_per_clifford=1.8,
    epc_standard_error=0.027,
    single_qubit_error_standard_error=0.0012,
)

print(
    "Fidelity from rounded article inputs: "
    f"{100 * result['entangling_fidelity']:.4f}% "
    f"+/- {100 * result['standard_error']:.4f} percentage points"
)
