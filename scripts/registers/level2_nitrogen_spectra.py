import json
from pathlib import Path

import numpy as np

from echo_spin.nitrogen_reference import (
    build_nitrogen_reference,
    nitrogen_local_spectra,
    nitrogen_magnetic_drive_operator,
)
from echo_spin.provenance import software_provenance
from echo_spin.scenario import longitudinal_scenario_from_dict


def main():
    """Save local hyperfine-resolved lines for the gradient register."""
    root = Path("results/linear/n3")
    trial = json.loads(
        (root / "level_05_controls/local_addressability.json").read_text()
    )
    nuclei = json.loads(
        Path("configs/registers/linear/n3/nitrogen_14.json").read_text()
    )
    specification = trial["cases"]["gradient"]["specification"]
    scenario = longitudinal_scenario_from_dict(specification)
    reference = build_nitrogen_reference(scenario, nuclei)
    drive = nitrogen_magnetic_drive_operator(
        scenario, nuclei, trial["field_profile_tesla"]
    )
    spectra = nitrogen_local_spectra(reference, drive)

    for site in spectra:
        parameters = specification["sites"][site["site"]]
        centre = (
            parameters["D"]
            - parameters["gamma_e"]
            * scenario.register.magnetic_fields[site["site"], 2]
        )
        nearby = [
            line for line in site["transitions"]
            if abs(line["frequency_rad_per_us"] - centre) < 2 * np.pi * 10
        ]
        strongest = max(line["strength"] for line in nearby)
        print("Site", site["site"])
        for line in nearby:
            if line["strength"] > strongest / 100:
                print(
                    f"  {line['frequency_rad_per_us'] / (2*np.pi):.6f} MHz"
                    f"  coupling {np.sqrt(line['strength']) / (2*np.pi):.6f} MHz"
                )

    report = {
        "software": software_provenance(),
        "specification": specification,
        "nuclear_parameters": nuclei,
        "field_profile_tesla": trial["field_profile_tesla"],
        "spectra": spectra,
        "limit": "Isolated-site lines; no preparation-dependent weighting.",
    }
    output = root / "level_02_characterisation/nitrogen_spectra.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
