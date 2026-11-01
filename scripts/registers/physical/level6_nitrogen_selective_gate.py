import argparse
import json
from pathlib import Path

import numpy as np
from qutip import Qobj, qeye, tensor

from echo_spin.characterisation import longitudinal_characterisation_report
from echo_spin.core.operators import embed_operator
from echo_spin.dynamics.propagators import hermitian_propagator
from echo_spin.electronic_reference import (
    build_shared_drive_rwa,
    compile_shared_selective_events,
)
from echo_spin.nitrogen_reference import (
    build_nitrogen_drive_rwa,
    build_nitrogen_reference,
    calibrate_local_nitrogen_square_pulse,
    joint_magnetic_noise_drift,
    nitrogen_electronic_gate_metrics,
    nitrogen_logical_input_columns,
    nitrogen_magnetic_drive_operator,
    nitrogen_product_preparation,
    propagate_nitrogen_events,
)
from echo_spin.protocols import ideal_protocol_report
from echo_spin.provenance import software_provenance
from echo_spin.representation import (
    build_electronic_reference,
    longitudinal_model_report,
)
from echo_spin.scenario import longitudinal_scenario_from_dict


def main():
    """Evaluate a complete nitrogen-aware sequence for two nuclear preparations."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", nargs="+", type=int, default=[256, 512]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            'results/linear/n3/physical/electron_nitrogen/level_06_evaluation/nitrogen_selective_gate.json'
        ),
    )
    parser.add_argument(
        "--magnetic-trajectory",
        type=Path,
        help="Saved magnetic-field trajectory with explicit units and provenance.",
    )
    parser.add_argument(
        "--noise-free-step-us",
        type=float,
        default=0.05,
        help="Maximum free-interval integration step when a trajectory is active.",
    )
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error("Output exists; choose a new --output filename.")
    if any(steps < 1 for steps in arguments.steps):
        parser.error("Step counts must be positive.")
    if (
        not np.isfinite(arguments.noise_free_step_us)
        or arguments.noise_free_step_us <= 0
    ):
        parser.error("--noise-free-step-us must be finite and positive.")

    root = Path("results/linear/n3")
    source = json.loads(
        (root / 'physical/electronic/level_05_controls/local_addressability.json').read_text()
    )
    nuclei = json.loads(
        Path("configs/registers/linear/n3/nitrogen_14.json").read_text()
    )
    specification = source["cases"]["gradient"]["specification"]
    scenario = longitudinal_scenario_from_dict(specification)
    fields = source["field_profile_tesla"]
    number_of_nv = scenario.register.number_of_nv

    reference = build_nitrogen_reference(scenario, nuclei)
    magnetic = nitrogen_magnetic_drive_operator(
        scenario, nuclei, fields
    )
    calibrations = [
        calibrate_local_nitrogen_square_pulse(
            reference, magnetic, site=site
        )
        for site in range(number_of_nv)
    ]
    pulse_duration = calibrations[scenario.target_pair[0]]["duration_us"]
    model = build_nitrogen_drive_rwa(
        reference,
        magnetic,
        calibrations[scenario.target_pair[0]]["carrier_rad_per_us"],
    )

    protocol = ideal_protocol_report(
        longitudinal_model_report(
            longitudinal_characterisation_report(scenario)
        )
    )
    electronic_model = build_shared_drive_rwa(
        scenario, fields, source["carrier_rad_per_us"]
    )
    schedule = compile_shared_selective_events(
        protocol, electronic_model, pulse_duration
    )
    for event in schedule["events"]:
        for tone in event["tones"]:
            calibration = calibrations[tone["intended_site"]]
            tone["carrier_rad_per_us"] = calibration["carrier_rad_per_us"]
            tone["phase_rad"] = (
                calibration["logical_x_carrier_phase_rad"]
                - (np.pi / 2 if tone["logical_axis"] == "Y" else 0.0)
            )
            tone["envelope_scale"] = (
                calibration["duration_us"] / pulse_duration
            )

    preparations = {
        "polarised": nitrogen_logical_input_columns(
            nitrogen_product_preparation([1] * number_of_nv)
        ),
        "unpolarised": nitrogen_logical_input_columns(
            nitrogen_product_preparation([None] * number_of_nv)
        ),
    }
    complete = preparations["unpolarised"]
    polar_coefficients = np.einsum(
        "ki,kj->ij",
        complete["columns"].conj(),
        preparations["polarised"]["columns"],
        optimize=False,
    )
    transform = model["basis_to_lab"].to("CSR").data.as_scipy()
    initial = transform.conjugate().transpose().dot(complete["columns"])

    duration = schedule["duration_us"]
    noise_record = None
    noisy_drift = None
    if arguments.magnetic_trajectory is not None:
        noise_record = json.loads(
            arguments.magnetic_trajectory.read_text(encoding="utf-8")
        )
        expected_units = {
            "times": "us",
            "amplitudes": "T",
            "profiles": "dimensionless",
        }
        if noise_record.get("units") != expected_units:
            parser.error("Magnetic trajectory units must be us, T and dimensionless.")
        if noise_record.get("status") not in {
            "exploratory", "literature_based", "experimentally_calibrated"
        }:
            parser.error("Provide an explicit magnetic-trajectory status.")
        if not isinstance(noise_record.get("source"), str) or not (
            noise_record["source"].strip()
        ):
            parser.error("Provide a nonempty magnetic-trajectory source.")

        noisy_drift = joint_magnetic_noise_drift(
            model,
            scenario,
            nuclei,
            noise_record["profiles"],
            noise_record["times_us"],
            noise_record["amplitudes_tesla"],
        )
        # Reject incomplete coverage before beginning propagation.
        noisy_drift(0.0)
        noisy_drift(duration)

    rotating_phase = np.exp(
        -1j * model["carrier_rad_per_us"] * duration
        * model["excitation_operator"].diag().real
    )
    bare_electronic = build_electronic_reference(scenario)
    bare_reference = bare_electronic["local_hamiltonian"]
    calibrated_reference = bare_reference.copy()

    for site, calibration in enumerate(calibrations):
        local = bare_reference.ptrace(site) / 3**(number_of_nv - 1)
        energies, states = local.eigenstates()
        bare_frequency = energies[1] - energies[0]
        offset = calibration["carrier_rad_per_us"] - bare_frequency
        calibrated_reference += offset * embed_operator(
            states[1].proj(), site, bare_electronic["system"]
        )

    permutation = [
        index
        for site in range(number_of_nv)
        for index in (site, number_of_nv + site)
    ]
    comparison_frames = {}
    for label, generator in (
        ("bare_electronic", bare_reference),
        ("calibrated_carriers", calibrated_reference),
    ):
        electron_frame = hermitian_propagator(generator, -duration)
        comparison_frames[label] = tensor(
            electron_frame, qeye([3] * number_of_nv)
        ).permute(permutation).to("CSR").data.as_scipy()

    target_diagonal = (
        np.asarray(protocol["target_diagonal"]["real"])
        + 1j * np.asarray(protocol["target_diagonal"]["imag"])
    )
    target = Qobj(
        np.diag(target_diagonal),
        dims=[[2] * number_of_nv, [2] * number_of_nv],
    )

    print("Preparing the fixed-drift eigensystem...", flush=True)
    energies, states = model["drift_hamiltonian"].eigenstates()
    free_eigensystem = (
        np.asarray(energies),
        np.column_stack([state.full().ravel() for state in states]),
    )

    results = []
    for steps in arguments.steps:
        print(f"Propagating all nuclear inputs: {steps} steps/event...", flush=True)
        evolved = propagate_nitrogen_events(
            model,
            schedule,
            initial,
            steps=steps,
            free_eigensystem=free_eigensystem,
            drift_at_time=noisy_drift,
            free_step_us=arguments.noise_free_step_us,
        )
        lab = transform.dot(rotating_phase[:, None] * evolved)
        record = {
            "steps_per_event": steps,
            "frames": {},
        }
        for frame_name, comparison_frame in comparison_frames.items():
            compared = comparison_frame.dot(lab)
            outputs = {
                "unpolarised": compared,
                "polarised": np.einsum(
                    "ki,ij->kj", compared, polar_coefficients, optimize=False
                ),
            }
            frame_results = {}
            for label, columns in outputs.items():
                metrics = nitrogen_electronic_gate_metrics(
                    columns, preparations[label], target
                )
                frame_results[label] = metrics
                print(steps, frame_name, label, metrics, flush=True)
            record["frames"][frame_name] = frame_results

        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = arguments.output.with_name(
            f"{arguments.output.stem}_steps{steps}_joint.npz"
        )
        np.savez_compressed(
            checkpoint,
            lab_columns=lab,
            initial_columns=complete["columns"],
            probabilities=complete["probabilities"],
            logical_rows=complete["logical_rows"],
            polar_coefficients=polar_coefficients,
        )
        record["joint_columns_file"] = str(checkpoint)
        results.append(record)

    report = {
        "software": software_provenance(),
        "specification": specification,
        "nuclear_parameters": nuclei,
        "field_profile_tesla": fields,
        "source_protocol_report": protocol,
        "schedule": schedule,
        "calibrations": [
            {key: value for key, value in calibration.items()
             if key != "propagator"}
            for calibration in calibrations
        ],
        "pulse_shape": "sine",
        "results": results,
        "comparison_frames": ["bare_electronic", "calibrated_carriers"],
        "target_convention": (
            "The target matrix is interpreted separately in each stated frame."
        ),
        "limits": [
            "Dressed-manifold microwave RWA.",
            "Same polarised-branch calibration for both preparations.",
            "Nuclear memory retained throughout the complete sequence.",
            "No nuclear reset or physical local corrections.",
            "No dissipative relaxation.",
            (
                "No stochastic magnetic trajectory applied."
                if noise_record is None
                else "One prescribed magnetic trajectory; not an ensemble average."
            ),
        ],
        "magnetic_trajectory": noise_record,
        "noise_free_step_us": (
            arguments.noise_free_step_us if noisy_drift is not None else None
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
