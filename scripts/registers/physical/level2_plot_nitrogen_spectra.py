import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from echo_spin.plotting import plot_local_transition_spectra


def main():
    """Plot saved hyperfine-resolved magnetic transition lines."""
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--range-mhz", nargs=2, type=float)
    arguments = parser.parse_args()

    report = json.loads(
        arguments.report.read_text(encoding="utf-8")
    )
    figure = plot_local_transition_spectra(
        report["spectra"],
        title="Local electron–14N transitions under the shared microwave field",
        frequency_range_mhz=arguments.range_mhz,
        strength_label="strength\n(rad/µs)²",
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
