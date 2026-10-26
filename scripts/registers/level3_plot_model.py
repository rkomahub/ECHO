import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from echo_spin.plotting import plot_dipolar_comparison


def main():
    """Save a dipolar comparison figure from a Level 3 model report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--component",
        choices=("real", "imag"),
        default="real",
    )
    arguments = parser.parse_args()

    with arguments.report.open(encoding="utf-8") as stream:
        report = json.load(stream)

    figure = plot_dipolar_comparison(
        report,
        component=arguments.component,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
