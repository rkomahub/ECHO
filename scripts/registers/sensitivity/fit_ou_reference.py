import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


def ou_factors(x):
    """Evaluate dimensionless Ramsey and Hahn OU exponents stably."""
    if x < 1e-3:
        ramsey = x**2 * (
            0.5 - x / 6 + x**2 / 24 - x**3 / 120
        )
        hahn = x**3 * (
            1 / 12 - x / 32 + 7 * x**2 / 960 - x**3 / 768
        )
        return ramsey, hahn
    return (
        x + np.expm1(-x),
        x + 4 * np.expm1(-x / 2) - np.expm1(-x),
    )


def main():
    """Save a conditional OU fit to published NV1 coherence scales."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error("Output exists; choose a new filename.")

    ramsey_time = 2.60
    hahn_time = 55.0

    def mismatch(log_tau):
        """Compare the two unit-coherence-exponent constraints."""
        tau = np.exp(log_tau)
        ramsey = ou_factors(ramsey_time / tau)[0]
        hahn = ou_factors(hahn_time / tau)[1]
        return np.log(hahn) - np.log(ramsey)

    log_tau = brentq(
        mismatch,
        np.log(ramsey_time * 1e-6),
        np.log(hahn_time * 1e6),
    )
    tau = float(np.exp(log_tau))
    sigma = float(
        1 / (tau * np.sqrt(ou_factors(ramsey_time / tau)[0]))
    )

    specification = json.loads(Path(
        "configs/registers/linear/n3/ideal/scenario.json"
    ).read_text(encoding="utf-8"))
    gamma = abs(specification["sites"][0]["gamma_e"])

    checks = {
        "ramsey_chi": sigma**2 * tau**2
        * ou_factors(ramsey_time / tau)[0],
        "hahn_chi": sigma**2 * tau**2
        * ou_factors(hahn_time / tau)[1],
    }
    report = {
        "status": "literature_based",
        "source": "https://arxiv.org/abs/2406.04199",
        "reference_site": "Joas NV1; not calibrated for the ECHO register",
        "ramsey_time_us": ramsey_time,
        "hahn_time_us": hahn_time,
        "sigma_rad_per_us": sigma,
        "correlation_time_us": tau,
        "sigma_field_tesla": sigma / gamma,
        "gamma_coefficient_rad_per_us_per_t": gamma,
        "checks": {key: float(value) for key, value in checks.items()},
        "assumptions": [
            "Stationary Gaussian single-component OU noise.",
            "Coherence times interpreted as environmental 1/e envelope times.",
            "Deterministic nitrogen beating excluded from the fitted envelope.",
            "Negligible relaxation and pulse errors in the reference fit.",
            "Nominal central values only; uncertainties not propagated.",
            "Transfer to the three-NV register is a scenario choice.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
