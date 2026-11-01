import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from echo_spin.plotting import plot_phase_cycle_comparison


def main():
    """Plot phase-cycle comparisons from labelled amplitude-error scans."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--scan",
        nargs=2,
        action="append",
        required=True,
        metavar=("LABEL", "REPORT"),
    )
    arguments = parser.parse_args()

    try:
        reports = {}
        for label, filename in arguments.scan:
            if label in reports:
                raise ValueError(f"Duplicate scan label: {label}.")
            reports[label] = json.loads(
                Path(filename).read_text(encoding="utf-8")
            )

        figure = plot_phase_cycle_comparison(reports)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(arguments.output, bbox_inches="tight")
        plt.close(figure)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
