import argparse
import copy
import json
from pathlib import Path

import numpy as np
from qutip import Qobj

from echo_spin.characterisation import longitudinal_characterisation_report
from echo_spin.electronic_reference import (
    build_shared_drive_rwa,
    compile_shared_selective_events,
)
from echo_spin.evaluation import evaluate_shared_electronic_gate
from echo_spin.protocols import ideal_protocol_report
from echo_spin.provenance import software_provenance
from echo_spin.representation import longitudinal_model_report
from echo_spin.scenario import longitudinal_scenario_from_dict


def main():
    """Evaluate one physical electronic spacing and magnetic-field configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--separation-nm", type=float, required=True)
    parser.add_argument("--field-scale", type=float, default=1.0)
    parser.add_argument("--gradient-scale", type=float, default=1.0)
    parser.add_argument("--steps", nargs="+", type=int, default=[512, 1024])
    arguments = parser.parse_args()

    if arguments.output.exists():
        parser.error("Output exists; choose a new filename.")
    values = [
        arguments.separation_nm,
        arguments.field_scale,
        arguments.gradient_scale,
    ]
    if (
        not np.all(np.isfinite(values))
        or arguments.separation_nm <= 0
        or arguments.field_scale < 0
        or arguments.gradient_scale < 0
        or any(step < 1 for step in arguments.steps)
    ):
        parser.error("Provide positive spacing/steps and nonnegative field scales.")

    source_path = Path(
        "results/linear/n3/physical/electronic/"
        "level_05_controls/local_addressability.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    specification = copy.deepcopy(
        source["cases"]["gradient"]["specification"]
    )
    geometry = specification["register"]
    positions = np.asarray(geometry["positions"], dtype=float)
    a, b = specification["task"]["target_pair"]
    anchor = positions[a].copy()
    scale = (
        arguments.separation_nm * 1e-9
        / np.linalg.norm(positions[b] - anchor)
    )
    geometry.pop("lattice", None)
    geometry["positions"] = (
        anchor + scale * (positions - anchor)
    ).tolist()
    geometry["field_at_reference"] = (
        arguments.field_scale
        * np.asarray(geometry["field_at_reference"], dtype=float)
    ).tolist()
    geometry["gradient"] = (
        arguments.gradient_scale
        * np.asarray(geometry["gradient"], dtype=float)
    ).tolist()

    report = {
        "software": software_provenance(),
        "source_local_trial_path": str(source_path),
        "specification": specification,
        "separation_nm": arguments.separation_nm,
        "field_scale": arguments.field_scale,
        "gradient_scale": arguments.gradient_scale,
        "microwave_field_profile_tesla": source["field_profile_tesla"],
        "pulse_duration_us": source["duration_us"],
        "pulse_shape": "sine",
        "phase_cycle": "xy8_targets",
        "comparison_frame": "Nominal local rotating-Hamiltonian interaction frame.",
        "limits": [
            "Electronic microwave RWA; no nuclei, noise or dissipation.",
            "Fixed pulse duration and microwave spatial profile.",
            "Carriers and tone amplitudes reconstructed by local calibration.",
            "No fidelity optimisation or entangling-duration recalibration.",
        ],
    }
    try:
        scenario = longitudinal_scenario_from_dict(specification)
        protocol = ideal_protocol_report(
            longitudinal_model_report(
                longitudinal_characterisation_report(scenario)
            )
        )
        # The reference carrier is a coordinate choice; compiled tones
        # are reconstructed from this configuration's local energies.
        model = build_shared_drive_rwa(
            scenario,
            source["field_profile_tesla"],
            source["carrier_rad_per_us"],
        )
        schedule = compile_shared_selective_events(
            protocol, model, source["duration_us"]
        )
        diagonal = protocol["target_diagonal"]
        number = scenario.register.number_of_nv
        target = Qobj(
            np.diag(
                np.asarray(diagonal["real"])
                + 1j * np.asarray(diagonal["imag"])
            ),
            dims=[[2] * number, [2] * number],
        )
        report["schedule"] = schedule
        report["results"] = []
        for steps in arguments.steps:
            result = evaluate_shared_electronic_gate(
                model, schedule, target, steps=steps
            )
            result["steps_per_event"] = steps
            report["results"].append(result)
            print(result, flush=True)
        report["status"] = "evaluated"
        if len(report["results"]) >= 2:
            report["fidelity_refinement_change"] = abs(
                report["results"][-1]["average_gate_fidelity"]
                - report["results"][-2]["average_gate_fidelity"]
            )
    except ValueError as error:
        report["status"] = "rejected"
        report["reason"] = str(error)
        print("Rejected:", error)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
