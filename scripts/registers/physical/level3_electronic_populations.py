import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from echo_spin.characterisation import characterise_longitudinal_scenario
from echo_spin.electronic_reference import electronic_population_diagnostics
from echo_spin.plotting import plot_electronic_populations
from echo_spin.provenance import software_provenance
from echo_spin.scenario import longitudinal_scenario_from_dict


def main():
    """Save and plot full electronic populations over the nominal gate duration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("figure", type=Path)
    arguments = parser.parse_args()

    specification = json.loads(
        arguments.scenario.read_text(encoding="utf-8")
    )
    scenario = longitudinal_scenario_from_dict(specification)
    parameters = characterise_longitudinal_scenario(scenario)
    a, b = scenario.target_pair
    duration = float(parameters["ideal_gate_times_us"][a, b])
    if not np.isfinite(duration):
        parser.error("The target pair must have nonzero coupling.")

    report = electronic_population_diagnostics(
        scenario, np.linspace(0, duration, 201)
    )
    report.update({
        "source_specification": specification,
        "software": software_provenance(),
        "nominal_gate_duration_us": duration,
    })

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")

    saved = json.loads(arguments.output.read_text(encoding="utf-8"))
    figure = plot_electronic_populations(saved)
    arguments.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.figure, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
