import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from echo_spin.plotting import plot_amplitude_error_comparison


def main():
    """Plot a saved systematic amplitude-error comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    report = json.loads(
        arguments.report.read_text(encoding="utf-8")
    )
    figure = plot_amplitude_error_comparison(report)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
