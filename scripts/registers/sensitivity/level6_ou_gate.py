import argparse
import json
from pathlib import Path

import numpy as np
from qutip import Qobj

from echo_spin.electronic_reference import build_shared_drive_rwa
from echo_spin.evaluation import evaluate_shared_electronic_gate
from echo_spin.noise.dephasing import (
    electronic_detuning_noise_model,
    ou_detuning_samples,
)
from echo_spin.provenance import software_provenance
from echo_spin.scenario import longitudinal_scenario_from_dict


def main():
    """Save an exploratory OU ensemble with paired numerical and grid checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--sigma-mhz", type=float, default=0.2)
    parser.add_argument("--tau-us", type=float, default=1.0)
    parser.add_argument("--noise-step-us", type=float, default=0.025)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--common-fraction", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        arguments.samples < 2 or arguments.steps < 1
        or not np.isfinite(arguments.noise_step_us)
        or arguments.noise_step_us <= 0
    ):
        parser.error("Provide at least two samples and positive numerical settings.")

    root = Path("results/linear/n3/sensitivity/effective_noise/level_06_evaluation")
    output = arguments.output or root / "ou_shared_gate.json"
    if output.exists():
        parser.error("Output exists; preserve it or choose a new output path.")

    source = json.loads(Path('results/linear/n3/physical/electronic/physical/electronic/level_06_evaluation/shared_selective_gate.json').read_text())
    trial = source["source_local_trial"]
    scenario = longitudinal_scenario_from_dict(
        trial["cases"]["gradient"]["specification"]
    )
    model = build_shared_drive_rwa(
        scenario, trial["field_profile_tesla"], trial["carrier_rad_per_us"]
    )
    schedule = source["schedule"]
    number_of_nv = scenario.register.number_of_nv
    diagonal = source["source_protocol_report"]["target_diagonal"]
    target = Qobj(
        np.diag(np.asarray(diagonal["real"]) + 1j*np.asarray(diagonal["imag"])),
        dims=[[2]*number_of_nv, [2]*number_of_nv],
    )
    parameters = {
        "model": "ornstein_uhlenbeck",
        "operator": "Z/2",
        "spatial_correlation": (
            "independent" if arguments.common_fraction == 0
            else {
                "model": "common_fraction",
                "fraction": arguments.common_fraction,
            }
        ),
        "sites": [
            {
                "sigma_rad_per_us": float(2*np.pi*arguments.sigma_mhz),
                "correlation_time_us": arguments.tau_us,
            }
            for _ in range(number_of_nv)
        ],
        "seed": 2026,
    }
    intervals = 2 * int(np.ceil(
        schedule["duration_us"] / (2*arguments.noise_step_us)
    ))
    times = np.linspace(0, schedule["duration_us"], intervals + 1)
    trajectories = ou_detuning_samples(
        parameters, times, arguments.samples
    )
    free_step = (times[1] - times[0]) / 2

    def evaluate(index, steps, stride=1, free_step_us=free_step):
        """Evaluate one fixed trajectory with optional coarsening of its grid."""
        knots = times[::stride]
        values = trajectories[index, ::stride, :]

        def drift(time):
            """Interpolate the same noise path throughout the sequence."""
            offsets = [
                np.interp(time, knots, values[:, site])
                for site in range(number_of_nv)
            ]
            return electronic_detuning_noise_model(
                model, offsets
            )["drift_hamiltonian"]

        return evaluate_shared_electronic_gate(
            model, schedule, target, steps=steps,
            drift_at_time=drift, free_step_us=free_step_us,
        )

    baseline = evaluate_shared_electronic_gate(
        model, schedule, target, steps=arguments.steps
    )
    records = []
    for index in range(arguments.samples):
        records.append(evaluate(index, arguments.steps))
        print(f"Sample {index+1}/{arguments.samples}", flush=True)

    refined = evaluate(
        0, 2*arguments.steps, free_step_us=free_step/2
    )
    coarse_grid = evaluate(
        0, 2*arguments.steps, stride=2, free_step_us=free_step/2
    )
    fidelities = np.array([
        record["average_gate_fidelity"] for record in records
    ])
    summary = {
        "mean_fidelity": float(fidelities.mean()),
        "fidelity_standard_error": float(
            fidelities.std(ddof=1) / np.sqrt(arguments.samples)
        ),
        "mean_leakage": float(np.mean([
            record["average_leakage"] for record in records
        ])),
        "first_sample_numerical_change": abs(
            refined["average_gate_fidelity"]
            - records[0]["average_gate_fidelity"]
        ),
        "first_sample_noise_grid_change": abs(
            refined["average_gate_fidelity"]
            - coarse_grid["average_gate_fidelity"]
        ),
    }
    report = {
        "software": software_provenance(),
        "source_gate_report": source,
        "noise_parameters": parameters,
        "steps_per_event": arguments.steps,
        "noise_times_us": times.tolist(),
        "trajectories_rad_per_us": trajectories.tolist(),
        "interpolation": "Linear between saved OU samples.",
        "baseline": baseline,
        "records": records,
        "summary": summary,
        "limits": [
            "Exploratory effective electronic noise; no explicit nuclei.",
            "Numerical and noise-grid checks concern the first trajectory only.",
            "Coarse and fine grids use the same sampled path.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")

    print("Baseline:", baseline)
    print("Ensemble:", summary)


if __name__ == "__main__":
    main()
