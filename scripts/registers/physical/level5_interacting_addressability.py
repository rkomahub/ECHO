import json
from pathlib import Path

from echo_spin.electronic_reference import shared_square_pulse_populations
from echo_spin.provenance import software_provenance
from echo_spin.scenario import longitudinal_scenario_from_dict


def main():
    """Repeat the saved local addressing trial with dipolar interactions active."""
    root = Path("results/linear/n3/physical/electronic/level_05_controls")
    source = json.loads(
        (root / "local_addressability.json").read_text(encoding="utf-8")
    )
    cases = {}

    for label, case in source["cases"].items():
        result = shared_square_pulse_populations(
            longitudinal_scenario_from_dict(case["specification"]),
            source["field_profile_tesla"],
            source["carrier_rad_per_us"],
            source["duration_us"],
        )
        cases[label] = result
        print(label)
        for site in result["local_populations"]:
            print(" site", site["site"], site["eigenstate_populations"])
        print(" target population:", result["target_population_probability"])
        print(" leakage:", result["leakage_probability"])

    report = {
        "software": software_provenance(),
        "source_local_trial": source,
        "cases": cases,
        "limits": [
            "Common-carrier excitation-number RWA.",
            "Single initial product state; not a gate-fidelity calculation.",
            "No nuclei or noise.",
        ],
    }
    with (root / "interacting_addressability.json").open(
        "x", encoding="utf-8"
    ) as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
