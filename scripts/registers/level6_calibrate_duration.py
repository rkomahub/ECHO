import argparse
import json
from pathlib import Path

import numpy as np

from echo_spin.evaluation import finite_evaluation_report
from echo_spin.provenance import software_provenance


def main():
    """Estimate and verify a local duration correction for selective recoupling."""
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol_report", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    source = json.loads(
        arguments.protocol_report.read_text(encoding="utf-8")
    )
    evaluations = {}

    def evaluate(scale):
        """Evaluate square-pulse selective recoupling at a duration scale."""
        if scale not in evaluations:
            evaluations[scale] = finite_evaluation_report(
                source,
                pulse_duration_us=0.02,
                shape="square",
                phase_cycle="xy8_targets",
                duration_scale=scale,
            )
        return 1 - evaluations[scale]["protocols"][
            "hadamard_selective"
        ]["average_gate_fidelity"]

    h = 0.01
    minus, centre, plus = [evaluate(s) for s in (1-h, 1.0, 1+h)]
    curvature = minus - 2 * centre + plus

    if curvature > 0:
        estimate = float(np.clip(
            1 + h * (minus - plus) / (2 * curvature),
            1-h,
            1+h,
        ))
        evaluate(estimate)

    best = min(evaluations, key=evaluate)
    report = {
        "software": software_provenance(),
        "method": "Three-point quadratic estimate with direct verification.",
        "search_interval": [1-h, 1+h],
        "baseline_infidelity": centre,
        "selected_infidelity": evaluate(best),
        "selected_evaluation": evaluations[best],
        "sampled_points": [
            {"duration_scale": scale, "infidelity": evaluate(scale)}
            for scale in sorted(evaluations)
        ],
        "limit": "Local calibration; no global optimum is established.",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")

    selected = evaluations[best]
    print(f"Duration scale: {best:.10f}")
    print(f"Duration: {selected['duration_us']:.10f} us")
    print(f"Baseline infidelity: {centre:.6e}")
    print(f"Selected infidelity: {evaluate(best):.6e}")


if __name__ == "__main__":
    main()
