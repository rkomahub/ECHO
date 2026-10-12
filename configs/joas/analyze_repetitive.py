import numpy as np

from echo_spin.benchmarks.repetitive import fit_normalized_repetitive_decay

counts = np.arange(0, 33, 2)

# Recorded coherent shared-drive result; no phenomenological dephasing or SPAM.
signal = np.array([
    1.0,
    0.98072328,
    0.96195574,
    0.91797349,
    0.88101259,
    0.82725390,
    0.78615306,
    0.73294446,
    0.69316016,
    0.64224807,
    0.60659626,
    0.55625414,
    0.52187827,
    0.47690606,
    0.44662651,
    0.39947064,
    0.36511028,
])

for maximum_count in [16, 24, 32]:
    selected = counts <= maximum_count

    result = fit_normalized_repetitive_decay(
        repetitions=counts[selected],
        signal=signal[selected],
    )

    print(
        f"Range 0–{maximum_count}: "
        f"p={result['decay_parameter']:.8f}, "
        f"pEPG={result['pseudo_error_per_gate']:.8f}, "
        f"offset={result['offset']:.8f}, "
        f"amplitude={result['amplitude']:.8f}, "
        f"residual RMS={result['residual_rms']:.8f}"
    )
    print("Residuals =", result["residuals"])
