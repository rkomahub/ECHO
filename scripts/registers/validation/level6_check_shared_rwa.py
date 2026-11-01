import json
from pathlib import Path

import numpy as np
from qutip import Qobj, tensor

from echo_spin.dynamics.propagators import hermitian_propagator
from echo_spin.dynamics.sequences import periodic_square_propagator
from echo_spin.electronic_reference import (
    build_shared_drive_rwa,
    electronic_magnetic_drive_operator,
)
from echo_spin.provenance import software_provenance
from echo_spin.representation import build_electronic_reference
from echo_spin.scenario import longitudinal_scenario_from_dict


def main():
    """Compare exact-carrier and RWA square pulses on arbitrary logical inputs."""
    root = Path("results/linear/n3")
    source = json.loads(
        (root / 'physical/electronic/level_05_controls/local_addressability.json').read_text()
    )
    scenario = longitudinal_scenario_from_dict(
        source["cases"]["gradient"]["specification"]
    )
    fields = source["field_profile_tesla"]
    carrier = source["carrier_rad_per_us"]
    duration = source["duration_us"]

    full = build_electronic_reference(scenario)
    rwa = build_shared_drive_rwa(scenario, fields, carrier)
    magnetic = electronic_magnetic_drive_operator(scenario, fields)
    transform = rwa["basis_to_lab"]

    local_embedding = Qobj(
        [[1, 0], [0, 1], [0, 0]], dims=[[3], [2]]
    )
    embedding = tensor(
        [local_embedding] * scenario.register.number_of_nv
    )
    dimension = embedding.shape[1]

    predicted = hermitian_propagator(
        rwa["drift_hamiltonian"] + rwa["drive_operator"], duration
    )
    # Convert the RWA result back to the laboratory frame.
    frame = hermitian_propagator(
        carrier * rwa["excitation_operator"], duration
    )
    predicted_output = transform * frame * predicted * embedding

    records = []
    previous = None
    for steps in (512, 1024):
        exact = periodic_square_propagator(
            full["hamiltonian"],
            magnetic,
            carrier,
            duration,
            steps_per_period=steps,
        )
        exact_output = exact * transform * embedding
        overlap = predicted_output.dag() * exact_output
        agreement = (
            overlap.norm("fro")**2 + abs(complex(overlap.tr()))**2
        ) / (dimension * (dimension + 1))

        record = {
            "steps_per_period": steps,
            "logical_input_rwa_agreement": float(agreement),
            "isometry_error": float(
                (
                    exact_output.dag() * exact_output
                    - Qobj(np.eye(dimension), dims=embedding.dims[1:]*2)
                ).norm("fro") / np.sqrt(dimension)
            ),
            "refinement_output_change": (
                None if previous is None else float(
                    (exact_output - previous).norm("fro")
                    / np.sqrt(dimension)
                )
            ),
        }
        records.append(record)
        previous = exact_output
        print(record)

    report = {
        "software": software_provenance(),
        "source_local_trial": source,
        "selected_case": "gradient",
        "envelope": "square",
        "records": records,
        "metric": "Haar-averaged exact/RWA output overlap on logical inputs.",
        "limit": "Not fidelity against the desired pi-X gate.",
    }
    output = root / 'validation/level_06_evaluation/shared_rwa_check_refined.json'
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
