import numpy as np

from echo_spin.decoupling.modulation import (
    effective_zz_coupling,
    selective_pair_pulse_times,
)
from echo_spin.representation import build_longitudinal_model


def build_ideal_gate_candidates(model_report) -> dict:
    """Construct requested closed X-pi schedules and their common pair target."""
    model = build_longitudinal_model(model_report)
    specification = model_report["specification"]
    number_of_nv = model.system.number_of_spins
    pair = tuple(specification["task"]["target_pair"])

    record = specification.get("environment", {}).get("control_design", {})
    if record.get("status") != "specified":
        raise ValueError("Specify control_design before constructing candidates.")

    requested = record["parameters"]["candidate_protocols"]
    supported = {"free", "closed_global_hahn", "hadamard_selective"}
    if set(requested) - supported:
        raise ValueError("A requested protocol is not implemented by this builder.")

    coupling = float(
        model_report["couplings_rad_per_us"][pair[0]][pair[1]]
    )
    if coupling == 0.0:
        raise ValueError("The target pair has zero longitudinal coupling.")

    duration = np.pi / (4 * abs(coupling))
    if not np.isfinite(duration):
        raise ValueError("The target-pair gate duration must be finite.")

    z_a = model.pauli_operators["Z"][pair[0]]
    z_b = model.pauli_operators["Z"][pair[1]]
    target = (-1j * np.sign(coupling) * np.pi / 4 * z_a * z_b).expm()

    schedules = {}
    for name in requested:
        if name == "free":
            schedules[name] = [
                np.array([], dtype=float)
                for _ in range(number_of_nv)
            ]
        elif name == "closed_global_hahn":
            schedules[name] = [
                np.array([duration / 2, duration])
                for _ in range(number_of_nv)
            ]
        else:
            schedules[name] = selective_pair_pulse_times(
                number_of_qubits=number_of_nv,
                target_pair=pair,
                total_duration=duration,
            )

    return {
        "model": model,
        "target_pair": pair,
        "duration_us": float(duration),
        "target": target,
        "pulse_axis": "X",
        "pulse_angle_rad": float(np.pi),
        "schedules": schedules,
    }


def ideal_protocol_report(model_report) -> dict:
    """Export candidate schedules, exact sign averages and pulse resources."""
    candidates = build_ideal_gate_candidates(model_report)
    duration = candidates["duration_us"]
    target_diagonal = candidates["target"].diag()
    protocols = {}

    for name, schedules in candidates["schedules"].items():
        number_of_nv = len(schedules)
        local_averages = [
            effective_zz_coupling(1.0, duration, pulses, [])
            for pulses in schedules
        ]
        pair_averages = np.eye(number_of_nv)

        for i in range(number_of_nv):
            for j in range(i + 1, number_of_nv):
                value = effective_zz_coupling(
                    1.0, duration, schedules[i], schedules[j]
                )
                pair_averages[i, j] = value
                pair_averages[j, i] = value

        distinct_times = np.unique(np.concatenate(schedules))
        protocols[name] = {
            "pulse_times_us": [times.tolist() for times in schedules],
            "local_sign_averages": local_averages,
            "pair_sign_averages": pair_averages.tolist(),
            "individual_pulse_count": sum(len(times) for times in schedules),
            "distinct_pulse_times_us": distinct_times.tolist(),
            "distinct_pulse_time_count": int(len(distinct_times)),
            "closed_up_to_global_phase": all(
                len(times) % 2 == 0 for times in schedules
            ),
        }

    return {
        "report_version": 1,
        "source_model": model_report,
        "target_pair": list(candidates["target_pair"]),
        "duration_us": duration,
        "pulse_model": "instantaneous",
        "pulse_axis": candidates["pulse_axis"],
        "pulse_angle_rad": candidates["pulse_angle_rad"],
        "target_diagonal": {
            "real": target_diagonal.real.tolist(),
            "imag": target_diagonal.imag.tolist(),
        },
        "target_convention": (
            "exp(-i*sign(J_ab)*pi*Z_a*Z_b/4), spectator identity, "
            "site zero leftmost; no local corrections."
        ),
        "protocols": protocols,
        "limits": [
            "Sign averages apply to the static longitudinal model.",
            "Schedules include closing pulses.",
            "Ideal independent addressing is assumed.",
            "No finite-pulse, nuclear or leakage dynamics are evaluated.",
            "No gate propagation or fidelity calculation is performed.",
        ],
    }
