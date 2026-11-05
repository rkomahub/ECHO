import argparse
import json
from pathlib import Path

import numpy as np

from echo_spin.noise.dephasing import ornstein_uhlenbeck_noise


def main():
    """Save independent axial OU field trajectories for the joint gate trial."""
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("gate_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--noise-step-us", type=float, default=0.025)
    arguments = parser.parse_args()

    if arguments.output.exists():
        parser.error("Output exists; choose a new filename.")
    if (
        not np.isfinite(arguments.noise_step_us)
        or arguments.noise_step_us <= 0
        or arguments.seed < 0
    ):
        parser.error("Provide a positive finite time step and nonnegative seed.")

    reference = json.loads(
        arguments.reference.read_text(encoding="utf-8")
    )
    gate = json.loads(
        arguments.gate_report.read_text(encoding="utf-8")
    )
    duration = float(gate["schedule"]["duration_us"])
    number_of_nv = len(
        gate["source_local_trial"]["cases"]["gradient"]
        ["specification"]["register"]["positions"]
    )
    sigma = float(reference["sigma_field_tesla"])
    tau = float(reference["correlation_time_us"])
    if (
        not np.all(np.isfinite([duration, sigma, tau]))
        or duration <= 0 or sigma < 0 or tau <= 0
    ):
        parser.error("Invalid duration or OU reference parameters.")

    intervals = max(2, 2 * int(np.ceil(
        duration / (2 * arguments.noise_step_us)
    )))
    times = np.linspace(0.0, duration, intervals + 1)
    profiles = np.zeros((number_of_nv, number_of_nv, 3))
    for site in range(number_of_nv):
        profiles[site, site, 2] = 1.0

    rng = np.random.default_rng(arguments.seed)
    amplitudes = np.column_stack([
        ornstein_uhlenbeck_noise(
            times=times,
            sigma=sigma,
            correlation_time=tau,
            rng=rng,
        )
        for _ in range(number_of_nv)
    ])
    report = {
        "status": "literature_based",
        "source": (
            "Conditional OU fit to Joas NV1 coherence scales; "
            "same scale borrowed for all three sites."
        ),
        "units": {
            "times": "us",
            "amplitudes": "T",
            "profiles": "dimensionless",
        },
        "profiles": profiles.tolist(),
        "times_us": times.tolist(),
        "amplitudes_tesla": amplitudes.tolist(),
        "seed": arguments.seed,
        "spatial_correlation": "independent",
        "interpolation": "linear",
        "reference": reference,
        "gate_report_path": str(arguments.gate_report),
        "limits": [
            "One trajectory per site, not an ensemble estimate.",
            "Independent axial fields are a chosen spatial model.",
            "Not calibrated for the ECHO register.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(times)} time samples for {number_of_nv} sites.")


if __name__ == "__main__":
    main()
