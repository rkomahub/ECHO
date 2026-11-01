import json
from pathlib import Path

import numpy as np
from qutip import Qobj, tensor

from echo_spin.characterisation import longitudinal_characterisation_report
from echo_spin.dynamics.propagators import hermitian_propagator
from echo_spin.electronic_reference import (
    build_shared_drive_rwa,
    compile_shared_selective_events,
    propagate_shared_events,
)
from echo_spin.protocols import ideal_protocol_report
from echo_spin.provenance import software_provenance
from echo_spin.representation import longitudinal_model_report
from echo_spin.scenario import longitudinal_scenario_from_dict


def main():
    """Save a refined whole-register evaluation of physical selective recoupling."""
    root = Path("results/linear/n3")
    source = json.loads(
        (root / 'physical/electronic/level_05_controls/local_addressability.json').read_text()
    )
    scenario = longitudinal_scenario_from_dict(
        source["cases"]["gradient"]["specification"]
    )
    protocol = ideal_protocol_report(
        longitudinal_model_report(
            longitudinal_characterisation_report(scenario)
        )
    )
    model = build_shared_drive_rwa(
        scenario,
        source["field_profile_tesla"],
        source["carrier_rad_per_us"],
    )
    schedule = compile_shared_selective_events(
        protocol, model, source["duration_us"]
    )

    number_of_nv = scenario.register.number_of_nv
    local_embedding = Qobj(
        [[1, 0], [0, 1], [0, 0]], dims=[[3], [2]]
    )
    embedding = tensor([local_embedding] * number_of_nv)
    dimension = embedding.shape[1]
    target_diagonal = (
        np.asarray(protocol["target_diagonal"]["real"])
        + 1j * np.asarray(protocol["target_diagonal"]["imag"])
    )
    target = Qobj(
        np.diag(target_diagonal),
        dims=[[2] * number_of_nv, [2] * number_of_nv],
    )
    frame = hermitian_propagator(
        model["local_rotating_hamiltonian"], -schedule["duration_us"]
    )

    results = []
    for steps in (1024, 2048):
        propagator = propagate_shared_events(model, schedule, steps=steps)
        evolved = frame * propagator * embedding
        logical = embedding.dag() * evolved
        outside = evolved - embedding * logical
        overlap = complex((target.dag() * logical).tr())
        fidelity = (
            logical.norm("fro")**2 + abs(overlap)**2
        ) / (dimension * (dimension + 1))

        result = {
            "steps_per_event": steps,
            "average_gate_fidelity": float(fidelity),
            "average_leakage": float(outside.norm("fro")**2 / dimension),
            "isometry_error": float(
                (evolved.dag() * evolved - Qobj(
                    np.eye(dimension), dims=target.dims
                )).norm("fro") / np.sqrt(dimension)
            ),
            "logical_operation": {
                "real": logical.full().real.tolist(),
                "imag": logical.full().imag.tolist(),
            },
        }
        results.append(result)
        print({key: value for key, value in result.items()
               if key != "logical_operation"})

    change = abs(
        results[1]["average_gate_fidelity"]
        - results[0]["average_gate_fidelity"]
    )
    print("Fidelity refinement change:", change)

    report = {
        "software": software_provenance(),
        "source_local_trial": source,
        "source_protocol_report": protocol,
        "schedule": schedule,
        "pulse_shape": "sine",
        "frame": "Interaction picture of the local rotating Hamiltonian.",
        "results": results,
        "fidelity_refinement_change": change,
        "limits": [
            "Electronic excitation-number RWA.",
            "No nuclei or noise; no physical local corrections.",
            "Nominal timing; no physical-sequence optimisation.",
        ],
    }
    output = root / 'physical/electronic/level_06_evaluation/shared_selective_gate.json'
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
