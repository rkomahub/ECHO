import argparse

import numpy as np

from echo_spin.benchmarks.randomized import fit_rb_decay


def analyze_archive(path, offsets):
    """Compare full-Clifford decay fits under explicitly assumed offsets."""
    with np.load(path, allow_pickle=False) as archive:
        family = archive["clifford_family"].item()
        lengths = archive["lengths"]
        samples = archive["samples"]

    if family != "full":
        raise ValueError("This script requires a full-Clifford archive.")

    # Zero-length circuits have no physical recovery; fit positive lengths.
    selected = lengths > 0
    lengths = lengths[selected]
    samples = samples[selected]

    if len(np.unique(lengths)) < 4:
        raise ValueError("Provide at least four distinct positive lengths.")

    if samples.ndim != 2 or samples.shape[1] < 2:
        raise ValueError("Provide at least two circuits per length.")

    means = samples.mean(axis=1)
    errors = samples.std(axis=1, ddof=1) / np.sqrt(samples.shape[1])

    if np.any(errors <= 0):
        raise ValueError("Circuit standard errors must be positive.")

    print("Positive lengths =", lengths)
    print("Mean returns =", means)
    print("Circuit standard errors =", errors)

    for offset in offsets:
        result = fit_rb_decay(
            lengths=lengths,
            signal=means,
            qubits=2,
            offset=offset,
            standard_errors=errors,
        )

        print(
            f"\nAssumed offset={offset:.4f}: "
            f"p={result['decay_parameter']:.6f} "
            f"+/- {result['decay_parameter_standard_error']:.6f}, "
            f"amplitude={result['amplitude']:.6f}, "
            f"reduced chi-squared={result['reduced_chi_squared']:.4f}"
        )
        print("Residuals =", result["residuals"])

    print(
        "\nOffsets are diagnostic assumptions. "
        "No calibrated EPC or native-gate fidelity is claimed."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument(
        "--offsets",
        type=float,
        nargs="+",
        default=[0.0, 0.25],
    )
    arguments = parser.parse_args()

    analyze_archive(arguments.archive, arguments.offsets)
