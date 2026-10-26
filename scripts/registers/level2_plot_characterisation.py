import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from echo_spin.plotting import plot_characterisation_report


def main():
    """Save characterisation figures from a previously generated Level 2 report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()

    with arguments.report.open(encoding="utf-8") as stream:
        report = json.load(stream)

    figures = plot_characterisation_report(report)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    for name, figure in figures.items():
        figure.savefig(
            arguments.output_directory / f"{name}.pdf",
            bbox_inches="tight",
        )
        plt.close(figure)


if __name__ == "__main__":
    main()
