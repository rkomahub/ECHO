import json
from pathlib import Path

import numpy as np

from echo_spin.control.pulses import sine_envelope
from echo_spin.electronic_reference import shared_pulse_gate_diagnostics
from echo_spin.provenance import software_provenance
from echo_spin.scenario import longitudinal_scenario_from_dict


def main():
    """Save a refined whole-register evaluation of the gradient sine pulse."""
    root = Path("results/linear/n3")
    source = json.loads(
        (root / 'physical/electronic/level_05_controls/local_addressability.json').read_text()
    )
    scenario = longitudinal_scenario_from_dict(
        source["cases"]["gradient"]["specification"]
    )
    duration = source["duration_us"]

    def envelope(time):
        """Return the equal-area sine envelope."""
        return sine_envelope(time, duration, np.pi / 2)

    results = [
        shared_pulse_gate_diagnostics(
            scenario,
            source["field_profile_tesla"],
            source["carrier_rad_per_us"],
            duration,
            envelope=envelope,
            steps=steps,
        )
        for steps in (1024, 2048)
    ]
    change = abs(
        results[1]["average_gate_fidelity"]
        - results[0]["average_gate_fidelity"]
    )
    report = {
        "software": software_provenance(),
        "source_local_trial": source,
        "selected_case": "gradient",
        "envelope": "sine; dimensionless peak pi/2",
        "integration_steps": [1024, 2048],
        "results": results,
        "fidelity_refinement_change": change,
    }
    output = root / 'physical/electronic/level_06_evaluation/shared_sine_gate.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")

    print("Whole-register fidelity:", results[1]["average_gate_fidelity"])
    print("Average leakage:", results[1]["average_leakage"])
    print("Fidelity refinement change:", change)


if __name__ == "__main__":
    main()
