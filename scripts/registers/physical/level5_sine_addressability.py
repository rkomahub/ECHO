import json
from pathlib import Path

import numpy as np

from echo_spin.control.pulses import sine_envelope
from echo_spin.electronic_reference import shared_pulse_populations
from echo_spin.provenance import software_provenance
from echo_spin.scenario import longitudinal_scenario_from_dict


def main():
    """Evaluate and refine one interacting shared sine pulse in the gradient case."""
    root = Path("results/linear/n3/physical/electronic/level_05_controls")
    source = json.loads(
        (root / "local_addressability.json").read_text(encoding="utf-8")
    )
    scenario = longitudinal_scenario_from_dict(
        source["cases"]["gradient"]["specification"]
    )
    duration = source["duration_us"]

    def envelope(time):
        """Return the equal-area dimensionless sine envelope."""
        return sine_envelope(time, duration, np.pi / 2)

    results = []
    for steps in (512, 1024):
        results.append(shared_pulse_populations(
            scenario,
            source["field_profile_tesla"],
            source["carrier_rad_per_us"],
            duration,
            envelope=envelope,
            steps=steps,
        ))

    coarse, fine = results
    population_change = max(
        np.max(np.abs(
            np.asarray(actual["eigenstate_populations"])
            - np.asarray(previous["eigenstate_populations"])
        ))
        for actual, previous in zip(
            fine["local_populations"], coarse["local_populations"]
        )
    )
    changes = {
        "maximum_local_population_change": float(population_change),
        "target_probability_change": abs(
            fine["target_population_probability"]
            - coarse["target_population_probability"]
        ),
        "leakage_probability_change": abs(
            fine["leakage_probability"] - coarse["leakage_probability"]
        ),
    }
    report = {
        "software": software_provenance(),
        "source_local_trial": source,
        "selected_case": "gradient",
        "envelope": {
            "shape": "sine",
            "dimensionless_peak": float(np.pi / 2),
            "duration_us": duration,
        },
        "results": results,
        "refinement_changes": changes,
        "limit": "Population convergence does not establish full gate convergence.",
    }
    with (root / "sine_addressability.json").open(
        "x", encoding="utf-8"
    ) as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")

    print("Target probability:", fine["target_population_probability"])
    print("Leakage:", fine["leakage_probability"])
    for name, value in changes.items():
        print(name, value)


if __name__ == "__main__":
    main()
