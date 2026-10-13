import numpy as np

from echo_spin.benchmarks.randomized import fit_rb_decay

lengths = np.array([0, 1, 2, 4, 8, 16, 32, 64])
true_signal = 0.25 + 0.70 * 0.97**lengths

shots = 100000
rng = np.random.default_rng(42)
measured = rng.binomial(shots, true_signal) / shots
errors = np.sqrt(measured * (1 - measured) / shots)

for maximum_length in [16, 32, 64]:
    selected = lengths <= maximum_length

    result = fit_rb_decay(
        lengths=lengths[selected],
        signal=measured[selected],
        qubits=2,
        offset=0.25,
        standard_errors=errors[selected],
    )

    print(
        f"Range 0–{maximum_length}: "
        f"p={result['decay_parameter']:.8f}, "
        f"EPC={result['error_per_clifford']:.8f} "
        f"+/- {result['error_per_clifford_standard_error']:.8f}, "
        f"reduced chi-squared={result['reduced_chi_squared']:.4f}"
    )
