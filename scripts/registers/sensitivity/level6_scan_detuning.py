import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from echo_spin.evaluation import evaluate_finite_protocols
from echo_spin.provenance import software_provenance


def main():
    """Save and plot selective-gate detuning sensitivity across phase cycles."""
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("figure", type=Path)
    parser.add_argument(
        "--detunings-mhz", nargs="+", type=float, required=True
    )
    arguments = parser.parse_args()

    source = json.loads(
        arguments.protocol_report.read_text(encoding="utf-8")
    )
    records = []
    for frequency in arguments.detunings_mhz:
        for cycle in ("x", "xy4", "xy8_targets"):
            result = evaluate_finite_protocols(
                source,
                pulse_duration_us=0.02,
                shape="square",
                phase_cycle=cycle,
                amplitude_error=0.0,
                detuning_error_rad_per_us=2 * np.pi * frequency,
            )
            selective = result["protocols"]["hadamard_selective"]
            records.append({
                "detuning_mhz": frequency,
                "detuning_error_rad_per_us": 2 * np.pi * frequency,
                "phase_cycle": cycle,
                "average_gate_fidelity": selective["average_gate_fidelity"],
                "unitarity_error": selective["unitarity_error"],
                "resources": selective["resources"],
            })

    report = {
        "report_version": 1,
        "source_protocol_report": source,
        "software": software_provenance(),
        "pulse_shape": "square",
        "pulse_duration_us": 0.02,
        "amplitude_error": 0.0,
        "records": records,
        "convention": "Common static Delta*sum(Z_i)/2 acts throughout.",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")

    saved = json.loads(arguments.output.read_text(encoding="utf-8"))
    figure, axis = plt.subplots(constrained_layout=True)
    labels = {
        "x": "X-only",
        "xy4": "XY4",
        "xy8_targets": "XY8 targets / XY4 spectator",
    }
    for cycle, label in labels.items():
        points = sorted(
            (
                record for record in saved["records"]
                if record["phase_cycle"] == cycle
            ),
            key=lambda record: record["detuning_mhz"],
        )
        axis.semilogy(
            [point["detuning_mhz"] for point in points],
            [
                max(1 - point["average_gate_fidelity"], 1e-15)
                for point in points
            ],
            "o-",
            label=label,
        )

    axis.set_xlabel(r"Common detuning $\Delta/(2\pi)$ (MHz)")
    axis.set_ylabel(r"Whole-register gate infidelity $1-F_{\rm avg}$")
    axis.set_title("Selective recoupling — square pulses, 20 ns")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    arguments.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.figure, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
