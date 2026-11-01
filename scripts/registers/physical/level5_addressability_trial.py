import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from echo_spin.characterisation import characterise_local_electrons
from echo_spin.electronic_reference import local_square_pulse_populations
from echo_spin.provenance import software_provenance
from echo_spin.scenario import longitudinal_scenario_from_dict


def main():
    """Save a shared-pulse local addressability comparison for the three-site chain."""
    source = Path("configs/registers/linear/n3/ideal/scenario.json")
    specification = json.loads(source.read_text(encoding="utf-8"))
    gradient = deepcopy(specification)
    gradient["register"]["gradient"] = [
        [0, 0, 1e5], [0, 0, 0], [1e5, 0, 0]
    ]
    fields = np.tile([1e-4, 0.0, 0.0], (3, 1))
    cases = {}
    carrier = duration = None

    for label, configuration in (
        ("uniform", specification), ("gradient", gradient)
    ):
        spectra = characterise_local_electrons(
            longitudinal_scenario_from_dict(configuration),
            field_profile_tesla=fields,
        )
        if carrier is None:
            transition = spectra[0]["transitions"][0]
            carrier = transition["frequency_rad_per_us"]
            duration = np.pi / np.sqrt(transition["strength"])

        populations = local_square_pulse_populations(
            spectra, carrier, duration
        )
        cases[label] = {
            "specification": configuration,
            "populations": populations,
        }
        for site in populations:
            print(label, site["site"], site["eigenstate_populations"])

    report = {
        "software": software_provenance(),
        "carrier_rad_per_us": float(carrier),
        "duration_us": float(duration),
        "field_profile_tesla": fields.tolist(),
        "model": "Isolated-site three-level square-pulse RWA.",
        "limits": "No dipolar dynamics, nuclei or noise during the pulse.",
        "cases": cases,
    }
    output = Path(
        'results/linear/n3/physical/electronic/level_05_controls/local_addressability.json'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
