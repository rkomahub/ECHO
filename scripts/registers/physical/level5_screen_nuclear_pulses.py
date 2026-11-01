import argparse
import json
from pathlib import Path

import numpy as np
from qutip import Qobj, basis, qeye, tensor

from echo_spin.dynamics.propagators import hermitian_propagator
from echo_spin.nitrogen_reference import (
    build_nitrogen_reference,
    calibrate_local_nitrogen_square_pulse,
    nitrogen_electronic_gate_metrics,
    nitrogen_logical_input_columns,
    nitrogen_magnetic_drive_operator,
    nitrogen_product_preparation,
)
from echo_spin.provenance import software_provenance
from echo_spin.scenario import longitudinal_scenario_from_dict


def main():
    """Screen local square-pulse inversion across all nitrogen branches."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error("Output exists; choose a new filename.")

    root = Path("results/linear/n3")
    trial = json.loads((
        root / "physical/electronic/level_05_controls/"
        "local_addressability.json"
    ).read_text(encoding="utf-8"))

    # The longitudinal scenario requires a target pair: retain its
    # original record for validation, then isolate the physical site below.
    # Local extraction avoids changing that scenario's task definition.
    full_specification = trial["cases"]["gradient"]["specification"]
    scenario = longitudinal_scenario_from_dict(full_specification)

    nuclei = json.loads(Path(
        "configs/registers/linear/n3/nitrogen_14.json"
    ).read_text(encoding="utf-8"))
    reference = build_nitrogen_reference(scenario, nuclei)
    magnetic = nitrogen_magnetic_drive_operator(
        scenario, nuclei, trial["field_profile_tesla"]
    )

    # Only these local 9D operators enter pulse calibration and evolution.
    from echo_spin.core.system import SpinSystem

    normalisation = 9 ** (scenario.register.number_of_nv - 1)
    local_reference = {
        "system": SpinSystem([1, 1]),
        "local_hamiltonian": (
            reference["local_hamiltonian"].ptrace([0, 1]) / normalisation
        ),
    }
    local_magnetic = magnetic.ptrace([0, 1]) / normalisation

    excitation = tensor(
        basis(3, 0).proj() + basis(3, 2).proj(),
        qeye(3),
    )
    target = -1j * Qobj([[0, 1], [1, 0]])
    preparations = {
        label: nitrogen_logical_input_columns(
            nitrogen_product_preparation([projection])
        )
        for label, projection in (
            ("polarised", 1),
            ("unpolarised", None),
        )
    }

    records = []
    for label, branch, scale in (
        ("original", 1, 1.0),
        ("centred", 0, 1.0),
        ("centred_x2", 0, 2.0),
        ("centred_x4", 0, 4.0),
    ):
        calibration = calibrate_local_nitrogen_square_pulse(
            local_reference,
            local_magnetic,
            nuclear_projection=branch,
            amplitude_scale=scale,
        )
        propagator = calibration["propagator"]
        carrier_frame = hermitian_propagator(
            calibration["carrier_rad_per_us"] * excitation,
            -calibration["duration_us"],
        )
        compared = carrier_frame * propagator
        gate_metrics = {
            label: nitrogen_electronic_gate_metrics(
                np.einsum(
                    "ij,jk->ik",
                    compared.full(),
                    inputs["columns"],
                    optimize=False,
                ),
                inputs,
                target,
            )
            for label, inputs in preparations.items()
        }
        inversion = []
        leakage = []
        for projection in (-1, 0, 1):
            initial = tensor(
                basis(3, 1), basis(3, 1 - projection)
            )
            evolved = propagator * initial
            populations = evolved.ptrace(0).diag().real
            inversion.append(float(populations[2]))
            leakage.append(float(populations[0]))

        record = {
            "label": label,
            "calibration": {
                key: value for key, value in calibration.items()
                if key != "propagator"
            },
            "nuclear_projection_order": [-1, 0, 1],
            "inversion_probabilities": inversion,
            "minimum_inversion": min(inversion),
            "unpolarised_inversion": sum(inversion) / 3,
            "leakage_probabilities": leakage,
            "local_gate_metrics": gate_metrics,
            "comparison_frame": "candidate_carrier",
        }
        records.append(record)
        print(
            label,
            "mean inversion:", record["unpolarised_inversion"],
            "worst branch:", record["minimum_inversion"],
            "maximum leakage:", max(leakage),
        )

        for preparation, metrics in gate_metrics.items():
            print(
                " ", preparation,
                "fidelity:", metrics["average_gate_fidelity"],
                "leakage:", metrics["average_leakage"],
            )

    report = {
        "software": software_provenance(),
        "source_local_trial": trial,
        "nuclear_parameters": nuclei,
        "site": 0,
        "records": records,
        "limits": [
            "Local square-pulse screening, not a register gate.",
            "No noise or intersite dynamics during local propagation.",
            "Inversion probabilities do not certify coherent gate phases.",
            "Amplitude scales are candidate choices, not hardware validation.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
