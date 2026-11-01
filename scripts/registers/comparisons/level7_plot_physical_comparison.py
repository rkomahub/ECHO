import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_point(path):
    """Read the highest-resolution result from one evaluated scan point."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["status"] != "evaluated":
        raise ValueError(f"Point was not evaluated: {path}")
    result = max(
        report["results"], key=lambda item: item["steps_per_event"]
    )
    return {
        "source": str(path),
        "separation_nm": report["separation_nm"],
        "field_scale": report["field_scale"],
        "gradient_scale": report["gradient_scale"],
        "steps_per_event": result["steps_per_event"],
        "average_gate_fidelity": result["average_gate_fidelity"],
        "infidelity": 1 - result["average_gate_fidelity"],
        "average_leakage": result["average_leakage"],
        "duration_us": report["schedule"]["duration_us"],
    }


def main():
    """Plot saved electronic spacing and gradient comparisons."""
    parser = argparse.ArgumentParser()
    parser.add_argument("reports_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()

    filenames = {
        "baseline": "physical_d15_reference.json",
        "d10": "physical_d10.json",
        "d20": "physical_d20.json",
        "gradient_half": "physical_d15_gradient_half.json",
        "gradient_double": "physical_d15_gradient_double_refined.json",
    }
    points = {
        label: read_point(arguments.reports_directory / filename)
        for label, filename in filenames.items()
    }
    spacing = sorted(
        [points[name] for name in ("d10", "baseline", "d20")],
        key=lambda point: point["separation_nm"],
    )
    gradient = sorted(
        [points[name] for name in (
            "gradient_half", "baseline", "gradient_double"
        )],
        key=lambda point: point["gradient_scale"],
    )

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    table_path = arguments.output_directory / "physical_comparison.csv"
    figure_path = arguments.output_directory / "physical_comparison.pdf"
    if table_path.exists() or figure_path.exists():
        parser.error("Output exists; choose a new output directory.")

    figure, axes = plt.subplots(
        1, 2, figsize=(8, 3.2), sharey=True, constrained_layout=True
    )
    for axis, records, key, label in (
        (
            axes[0], spacing, "separation_nm",
            r"Nearest-neighbour spacing $d$ (nm)",
        ),
        (
            axes[1], gradient, "gradient_scale",
            r"Gradient scale $s_G$",
        ),
    ):
        x = [point[key] for point in records]
        errors = [point["infidelity"] for point in records]
        axis.semilogy(x, errors, "o-", color="tab:blue")
        axis.set_xticks(x)
        axis.set_xlabel(label)
        axis.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel(r"Whole-register infidelity $1-F_{\mathrm{avg}}$")
    axes[0].set_title(r"Fixed gradient: $1\,\mathrm{G/nm}$")
    axes[1].set_title(r"Fixed spacing: $15\,\mathrm{nm}$")
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)

    rows = list(points.values())
    with table_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    print(figure_path)
    print(table_path)


if __name__ == "__main__":
    main()
