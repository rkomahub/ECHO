import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from echo_spin.plotting import plot_protocol_modulations


def main():
    """Save ideal sign-modulation plots from a Level 4 protocol report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    with arguments.report.open(encoding="utf-8") as stream:
        report = json.load(stream)

    figure = plot_protocol_modulations(report)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
