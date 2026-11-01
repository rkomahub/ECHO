import argparse
import json
from pathlib import Path

import numpy as np
from qutip import Qobj

from echo_spin.electronic_reference import (
    build_shared_drive_rwa,
    compile_shared_selective_events,
)
from echo_spin.evaluation import (
    evaluate_ideal_implementation,
    evaluate_shared_electronic_gate,
)
from echo_spin.implementation import ideal_implementation_report
from echo_spin.provenance import software_provenance
from echo_spin.scenario import longitudinal_scenario_from_dict


def main():
    """Compare ideal and physical protocols for the same register and target."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source_gate_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=2048)
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error("Output already exists.")

    source = json.loads(arguments.source_gate_report.read_text())
    trial = source["source_local_trial"]
    protocol = source["source_protocol_report"]
    scenario = longitudinal_scenario_from_dict(
        trial["cases"]["gradient"]["specification"]
    )
    model = build_shared_drive_rwa(
        scenario, trial["field_profile_tesla"], trial["carrier_rad_per_us"]
    )
    ideal = evaluate_ideal_implementation(
        ideal_implementation_report(protocol)
    )
    diagonal = protocol["target_diagonal"]
    number_of_nv = scenario.register.number_of_nv
    target = Qobj(
        np.diag(np.asarray(diagonal["real"]) + 1j*np.asarray(diagonal["imag"])),
        dims=[[2]*number_of_nv, [2]*number_of_nv],
    )

    records = {}
    for name in protocol["protocols"]:
        schedule = compile_shared_selective_events(
            protocol,
            model,
            source["schedule"]["pulse_duration_us"],
            phase_cycle=source["schedule"]["phase_cycle"],
            protocol_name=name,
        )
        physical = evaluate_shared_electronic_gate(
            model,
            schedule,
            target,
            steps=arguments.steps,
            shape=source["pulse_shape"],
            include_operation=True,
        )
        records[name] = {
            "ideal_fidelity": ideal["protocols"][name]["average_gate_fidelity"],
            "physical": physical,
            "duration_us": schedule["duration_us"],
            "tone_pulse_count": sum(
                len(event["tones"]) for event in schedule["events"]
            ),
            "pulse_event_count": len(schedule["events"]),
        }
        print(
            name,
            "ideal:", records[name]["ideal_fidelity"],
            "physical:", physical["average_gate_fidelity"],
            "leakage:", physical["average_leakage"],
            flush=True,
        )

    report = {
        "software": software_provenance(),
        "source_gate_report": source,
        "steps_per_event": arguments.steps,
        "records": records,
        "conventions": {
            "comparison": "Same gradient configuration, duration and target.",
            "frame": "Nominal local-electronic interaction frame.",
            "resources": "Tone pulses and distinct events include closing pulses.",
        },
        "limits": [
            "Physical model uses the electronic RWA; no nuclei or noise.",
            "Current comparison uses the aligned axial configuration.",
            "General physical operations need not be diagonal.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
