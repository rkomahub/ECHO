from copy import deepcopy

import numpy as np

from echo_spin.characterisation import longitudinal_characterisation_report
from echo_spin.evaluation import (
    evaluate_finite_protocols,
    evaluate_ideal_implementation,
)
from echo_spin.implementation import ideal_implementation_report
from echo_spin.protocols import ideal_protocol_report
from echo_spin.representation import longitudinal_model_report
from echo_spin.scenario import longitudinal_scenario_from_dict


def scan_pair_separations(specification, separations_m) -> list:
    """Compare ideal gates under uniform geometry scaling about a target site."""
    baseline = longitudinal_scenario_from_dict(specification)
    uncertainty = baseline.environment["uncertainty"]
    if (
        uncertainty["status"] == "specified"
        and uncertainty["parameters"]["model"] != "deterministic"
    ):
        raise ValueError("This scan does not execute uncertainty ensembles.")

    separations = list(separations_m)
    if not separations or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value <= 0
        for value in separations
    ):
        raise ValueError("Provide finite positive pair separations in metres.")

    a, b = baseline.target_pair
    positions = baseline.register.positions
    anchor = positions[a].copy()
    original_separation = np.linalg.norm(positions[b] - anchor)
    records = []

    for separation in separations:
        scale = float(separation / original_separation)
        candidate = deepcopy(baseline.specification)
        geometry = candidate["register"]
        geometry.pop("lattice", None)
        geometry["positions"] = (
            anchor + scale * (positions - anchor)
        ).tolist()

        record = {
            "requested_pair_separation_m": float(separation),
            "scale_factor": scale,
            "specification": candidate,
        }

        try:
            scenario = longitudinal_scenario_from_dict(candidate)
            characterisation = longitudinal_characterisation_report(scenario)
            model = longitudinal_model_report(characterisation)
            implementation = ideal_implementation_report(
                ideal_protocol_report(model)
            )
            evaluation = evaluate_ideal_implementation(implementation)
        except ValueError as error:
            record.update({
                "status": "rejected",
                "reason": str(error),
            })
        else:
            record.update({
                "status": "evaluated",
                "reason": None,
                "target_coupling_rad_per_us": (
                    characterisation["couplings_rad_per_us"][a][b]
                ),
                "duration_us": evaluation["duration_us"],
                "protocols": {
                    name: {
                        "average_gate_fidelity": result["average_gate_fidelity"],
                        "unitarity_error": result["unitarity_error"],
                        "phase_prediction_error": result["phase_prediction_error"],
                        "phase_aligned_frobenius_error": (
                            result["phase_aligned_frobenius_error"]
                        ),
                        "phases": result["phases"],
                        "resources": result["resources"],
                    }
                    for name, result in evaluation["protocols"].items()
                },
            })

        records.append(record)

    return records


def scan_pulse_durations(protocol_report, durations_us, *, steps=128) -> dict:
    """Compare finite envelopes and record sine time-resolution sensitivity."""
    durations = list(durations_us)
    if not durations or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value <= 0
        for value in durations
    ):
        raise ValueError("Provide finite positive pulse durations.")

    if (
        isinstance(steps, (bool, np.bool_))
        or not isinstance(steps, (int, np.integer))
        or steps < 1
    ):
        raise ValueError("steps must be a positive integer.")

    records = []
    for duration in durations:
        for shape in ("square", "sine"):
            record = {
                "pulse_duration_us": float(duration),
                "pulse_shape": shape,
            }
            try:
                coarse = evaluate_finite_protocols(
                    protocol_report,
                    duration,
                    shape=shape,
                    steps=steps,
                )
                fine = (
                    evaluate_finite_protocols(
                        protocol_report,
                        duration,
                        shape=shape,
                        steps=2 * steps,
                    )
                    if shape == "sine"
                    else coarse
                )
            except ValueError as error:
                record.update(status="rejected", reason=str(error))
            else:
                protocols = {}
                for name, result in fine["protocols"].items():
                    protocols[name] = {
                        key: result[key]
                        for key in (
                            "average_gate_fidelity",
                            "phase_aligned_frobenius_error",
                            "unitarity_error",
                            "resources",
                        )
                    }
                    protocols[name]["fidelity_resolution_change"] = (
                        abs(
                            result["average_gate_fidelity"]
                            - coarse["protocols"][name][
                                "average_gate_fidelity"
                            ]
                        )
                        if shape == "sine"
                        else None
                    )
                    protocols[name]["resolution_change"] = (
                        float(np.linalg.norm(
                            result["propagator"].full()
                            - coarse["protocols"][name]["propagator"].full(),
                            ord="fro",
                        ) / np.sqrt(fine["dimension"]))
                        if shape == "sine"
                        else None
                    )

                record.update(
                    status="evaluated",
                    reason=None,
                    peak_rabi_frequency_rad_per_us=(
                        fine["peak_rabi_frequency_rad_per_us"]
                    ),
                    integration_steps=fine["integration_steps"],
                    protocols=protocols,
                )
            records.append(record)

    return {
        "report_version": 1,
        "comparison": "finite_pulse_duration",
        "source_protocol_report": protocol_report,
        "duration_us": protocol_report["duration_us"],
        "coarse_sine_steps": int(steps),
        "reported_sine_steps": 2 * int(steps),
        "records": records,
        "limits": [
            "Longitudinal qubit drift and independent local controls.",
            "Fixed total duration and pulse-event end times.",
            "Equal pulse duration and area, not equal peak amplitude.",
            "Resolution change is a numerical diagnostic, not an error bound.",
            "No leakage, nuclear dynamics, noise or hardware amplitude bounds.",
        ],
    }
