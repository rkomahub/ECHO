import numpy as np
from qutip import qeye

from echo_spin.control.rotations import single_qubit_rotation
from echo_spin.core.operators import embed_operator
from echo_spin.decoupling.modulation import modulation_function
from echo_spin.protocols import build_ideal_gate_candidates


def implement_ideal_protocols(report) -> dict:
    """Convert saved X-pi schedules into pulse operators and free intervals."""
    if (
        report["pulse_model"] != "instantaneous"
        or report["pulse_axis"] != "X"
        or report["pulse_angle_rad"] != float(np.pi)
    ):
        raise ValueError("This implementation requires instantaneous X-pi pulses.")

    candidates = build_ideal_gate_candidates(report["source_model"])
    model = candidates["model"]
    duration = report["duration_us"]

    if duration != candidates["duration_us"]:
        raise ValueError("Protocol duration does not match the source model.")

    if tuple(report["target_pair"]) != candidates["target_pair"]:
        raise ValueError("Target pair does not match the source model.")

    saved_target = (
        np.asarray(report["target_diagonal"]["real"], dtype=float)
        + 1j * np.asarray(report["target_diagonal"]["imag"], dtype=float)
    )
    expected_target = candidates["target"].diag()
    if (
        saved_target.shape != expected_target.shape
        or not np.all(np.isfinite(saved_target))
        or not np.allclose(
            saved_target, expected_target, atol=1e-12, rtol=0.0
        )
    ):
        raise ValueError("Saved target does not match the source model.")

    system = model.system
    local_rotation = single_qubit_rotation(np.pi, "x")
    rotations = [
        embed_operator(local_rotation, site, system)
        for site in range(system.number_of_spins)
    ]

    implementations = {}
    for name, protocol in report["protocols"].items():
        schedules = protocol["pulse_times_us"]
        if len(schedules) != system.number_of_spins:
            raise ValueError("Provide one pulse schedule per site.")

        arrays = []
        for times in schedules:
            modulation_function([0.0], times)
            times = np.asarray(times, dtype=float)
            if np.any(times > duration):
                raise ValueError("Pulse times exceed the protocol duration.")
            if len(times) % 2:
                raise ValueError("The saved X-pi schedule is not closed.")
            arrays.append(times)

        event_times = np.unique(np.concatenate(arrays))
        pulses = []
        addressed_sites = []

        for time in event_times:
            pulse = qeye(list(system.dimensions))
            sites = []
            for site, times in enumerate(arrays):
                if np.any(times == time):
                    pulse = rotations[site] * pulse
                    sites.append(site)
            pulses.append(pulse)
            addressed_sites.append(sites)

        intervals = np.diff(
            np.concatenate(([0.0], event_times, [duration]))
        )

        implementations[name] = {
            "event_times_us": event_times,
            "addressed_sites": addressed_sites,
            "free_intervals_us": intervals,
            "pulse_propagators": tuple(pulses),
        }

    return {
        "model": model,
        "target": candidates["target"],
        "duration_us": float(duration),
        "protocols": implementations,
    }


def ideal_implementation_report(protocol_report) -> dict:
    """Export ideal control timelines and their reconstruction conventions."""
    implementation = implement_ideal_protocols(protocol_report)
    timelines = {}

    for name, protocol in implementation["protocols"].items():
        intervals = protocol["free_intervals_us"]
        timelines[name] = {
            "event_times_us": protocol["event_times_us"].tolist(),
            "addressed_sites": protocol["addressed_sites"],
            "free_intervals_us": intervals.tolist(),
            "event_operator_count": len(protocol["pulse_propagators"]),
            "checks": {
                "nonnegative_intervals": bool(np.all(intervals >= 0)),
                "duration_matches": bool(np.isclose(
                    intervals.sum(),
                    implementation["duration_us"],
                    atol=1e-12,
                    rtol=1e-12,
                )),
                "closed_up_to_global_phase": True,
            },
        }

    return {
        "report_version": 1,
        "source_protocol_report": protocol_report,
        "implementation": "instantaneous_local_X_pi",
        "duration_us": implementation["duration_us"],
        "operator_conventions": {
            "local_pulse": "exp(-i*pi*X_i/2) = -i*X_i",
            "simultaneous_pulses": "Product of commuting local pulses.",
            "tensor_order": "site_0_left",
            "interval_order": (
                "Free interval 0, pulse event 0, free interval 1, "
                "and so on, ending with the final free interval."
            ),
        },
        "protocols": timelines,
        "limits": [
            "Closure follows from validated even X-pi pulse counts per site.",
            "No finite pulse duration or shared-drive response is represented.",
            "No gate propagation or fidelity calculation is performed.",
        ],
    }
