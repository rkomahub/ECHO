import numpy as np

from echo_spin.benchmarks.randomized import (
    estimate_entangling_gate_error,
    two_qubit_clifford_catalog,
    two_qubit_native_gate_counts,
)


def summarize_compiler(include_entangler):
    """Report exact forward-Clifford primitive counts for a compiler."""
    catalog = two_qubit_clifford_catalog(
        include_entangler=include_entangler,
    )

    counts = np.array([
        two_qubit_native_gate_counts(word)
        for word in catalog["words"]
    ])

    family = "full" if include_entangler else "local"

    print(f"\nECHO {family} compiler:")
    print("Cliffords =", len(counts))
    print("Mean counts (single-qubit, entangling) =", counts.mean(axis=0))
    print("Maximum total primitives =", counts.sum(axis=1).max())
    print("Counts exclude the circuit recovery.")


summarize_compiler(include_entangler=True)
summarize_compiler(include_entangler=False)

# Rounded article inputs check the extraction formula independently of ECHO.
article = estimate_entangling_gate_error(
    error_per_clifford=0.149,
    single_qubit_error=0.0077,
    single_qubit_gates_per_clifford=10.5,
    entangling_gates_per_clifford=1.8,
    epc_standard_error=0.027,
    single_qubit_error_standard_error=0.0012,
)

print("\nArticle-input reference:")
print(
    f"Inferred fidelity = {100 * article['entangling_fidelity']:.4f}% "
    f"+/- {100 * article['standard_error']:.4f} percentage points"
)
print("Uncertainty assumes independent inputs and fixed compiler counts.")
print("This reference is not an ECHO fidelity measurement.")
