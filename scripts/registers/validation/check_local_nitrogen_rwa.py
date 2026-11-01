import argparse
import json
from pathlib import Path

import numpy as np

from echo_spin.core.system import SpinSystem
from echo_spin.dynamics.sequences import periodic_square_propagator
from echo_spin.nitrogen_reference import (
    build_nitrogen_reference,
    calibrate_local_nitrogen_square_pulse,
    nitrogen_magnetic_drive_operator,
)
from echo_spin.provenance import software_provenance
from echo_spin.scenario import longitudinal_scenario_from_dict


def main():
    """Compare a local nitrogen-aware RWA pulse with exact-carrier evolution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error("Output exists; choose a new filename.")

    source = json.loads(Path(
        "results/linear/n3/physical/electronic/"
        "level_05_controls/local_addressability.json"
    ).read_text(encoding="utf-8"))
    nuclei = json.loads(Path(
        "configs/registers/linear/n3/nitrogen_14.json"
    ).read_text(encoding="utf-8"))
    scenario = longitudinal_scenario_from_dict(
        source["cases"]["gradient"]["specification"]
    )
    reference = build_nitrogen_reference(scenario, nuclei)
    magnetic = nitrogen_magnetic_drive_operator(
        scenario, nuclei, source["field_profile_tesla"]
    )
    normalisation = 9 ** (scenario.register.number_of_nv - 1)
    local = reference["local_hamiltonian"].ptrace([0, 1]) / normalisation
    magnetic = magnetic.ptrace([0, 1]) / normalisation

    # Remove a common energy offset before numerical propagation.
    from qutip import qeye

    local -= float(np.real(local.tr())) / 9 * qeye([3, 3])
    local_reference = {
        "system": SpinSystem([1, 1]),
        "local_hamiltonian": local,
    }
    calibration = calibrate_local_nitrogen_square_pulse(
        local_reference, magnetic
    )
    predicted = calibration["propagator"].full()

    records = []
    previous = None
    for steps in (512, 1024):
        exact = periodic_square_propagator(
            local,
            magnetic,
            calibration["carrier_rad_per_us"],
            calibration["duration_us"],
            steps_per_period=steps,
        ).full()
        overlap = np.einsum(
            "ij,ij->", predicted.conj(), exact, optimize=False
        )
        gram = np.einsum(
            "ki,kj->ij", exact.conj(), exact, optimize=False
        )
        record = {
            "steps_per_period": steps,
            "joint_rwa_agreement": float(
                (abs(overlap)**2 + 9) / 90
            ),
            "unitarity_error": float(
                np.linalg.norm(gram - np.eye(9)) / 3
            ),
            "refinement_output_change": (
                None if previous is None
                else float(np.linalg.norm(exact - previous) / 3)
            ),
        }
        records.append(record)
        previous = exact
        print(record)

    report = {
        "software": software_provenance(),
        "source_local_trial": source,
        "nuclear_parameters": nuclei,
        "site": 0,
        "calibration": {
            key: value for key, value in calibration.items()
            if key != "propagator"
        },
        "records": records,
        "limits": [
            "Local square pulse at the original amplitude.",
            "No intersite interactions or stochastic noise.",
            "Joint model agreement, not target-gate fidelity.",
            "Does not validate the complete register sequence.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
