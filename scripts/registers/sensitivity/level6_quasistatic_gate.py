import argparse
import json
from pathlib import Path

import numpy as np
from qutip import Qobj, tensor

from echo_spin.dynamics.propagators import hermitian_propagator
from echo_spin.electronic_reference import (
    build_shared_drive_rwa,
    propagate_shared_events,
)
from echo_spin.noise.dephasing import (
    electronic_detuning_noise_model,
    quasistatic_detuning_samples,
)
from echo_spin.provenance import software_provenance
from echo_spin.scenario import longitudinal_scenario_from_dict


def main():
    """Evaluate exploratory quasistatic noise on the saved physical gate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--sigma-mhz", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=1024)
    arguments = parser.parse_args()
    if arguments.samples < 2:
        parser.error("At least two samples are required.")

    root = Path("results/linear/n3/sensitivity/effective_noise/level_06_evaluation")
    source = json.loads(Path('results/linear/n3/physical/electronic/physical/electronic/level_06_evaluation/shared_selective_gate.json').read_text())
    trial = source["source_local_trial"]
    scenario = longitudinal_scenario_from_dict(
        trial["cases"]["gradient"]["specification"]
    )
    model = build_shared_drive_rwa(
        scenario, trial["field_profile_tesla"], trial["carrier_rad_per_us"]
    )
    number_of_nv = scenario.register.number_of_nv
    parameters = {
        "model": "quasistatic",
        "operator": "Z/2",
        "spatial_correlation": "independent",
        "sites": [
            {"sigma_rad_per_us": float(2 * np.pi * arguments.sigma_mhz)}
            for _ in range(number_of_nv)
        ],
        "seed": 2026,
    }
    samples = quasistatic_detuning_samples(parameters, arguments.samples)
    schedule = source["schedule"]
    embedding = tensor([
        Qobj([[1, 0], [0, 1], [0, 0]], dims=[[3], [2]])
    ] * number_of_nv)
    dimension = embedding.shape[1]
    diagonal = source["source_protocol_report"]["target_diagonal"]
    target = Qobj(
        np.diag(np.asarray(diagonal["real"]) + 1j * np.asarray(diagonal["imag"])),
        dims=[[2] * number_of_nv, [2] * number_of_nv],
    )
    frame = hermitian_propagator(
        model["local_rotating_hamiltonian"], -schedule["duration_us"]
    )

    def evaluate(candidate, steps):
        """Evaluate one realisation in the fixed nominal comparison frame."""
        propagator = propagate_shared_events(
            candidate, schedule, shape=source["pulse_shape"], steps=steps
        )
        evolved = frame * propagator * embedding
        logical = embedding.dag() * evolved
        overlap = complex((target.dag() * logical).tr())
        return {
            "average_gate_fidelity": float(
                (logical.norm("fro")**2 + abs(overlap)**2)
                / (dimension * (dimension + 1))
            ),
            "average_leakage": float(
                (evolved - embedding * logical).norm("fro")**2 / dimension
            ),
        }

    baseline = evaluate(model, arguments.steps)
    records = []
    for index, offsets in enumerate(samples):
        result = evaluate(
            electronic_detuning_noise_model(model, offsets), arguments.steps
        )
        records.append({
            "offsets_rad_per_us": offsets.tolist(),
            **result,
        })
        print(f"Sample {index + 1}/{len(samples)}", flush=True)

    fidelities = np.array([
        record["average_gate_fidelity"] for record in records
    ])
    refined = evaluate(
        electronic_detuning_noise_model(model, samples[0]),
        2 * arguments.steps,
    )
    summary = {
        "mean_fidelity": float(fidelities.mean()),
        "fidelity_standard_error": float(
            fidelities.std(ddof=1) / np.sqrt(len(fidelities))
        ),
        "mean_leakage": float(np.mean([
            record["average_leakage"] for record in records
        ])),
        "first_sample_refinement_change": abs(
            refined["average_gate_fidelity"]
            - records[0]["average_gate_fidelity"]
        ),
    }
    report = {
        "software": software_provenance(),
        "source_gate_report": source,
        "noise_parameters": parameters,
        "sample_count": arguments.samples,
        "steps_per_event": arguments.steps,
        "noise_operator": "diag(1,-1,0)/2 in each local electronic energy basis.",
        "baseline": baseline,
        "records": records,
        "summary": summary,
        "limits": [
            "Exploratory effective noise; no experimental calibration.",
            "Electronic RWA; no explicit nuclei.",
            "Fixed nominal controls and comparison frame.",
            "Numerical refinement checked on the first sample only.",
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / "quasistatic_shared_gate.json").open(
        "x", encoding="utf-8"
    ) as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")

    print("Baseline:", baseline)
    print("Ensemble:", summary)


if __name__ == "__main__":
    main()
