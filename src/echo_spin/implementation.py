import numpy as np
from qutip import qeye

from echo_spin.control.pulses import (
    sine_envelope,
    sine_pi_pulse_amplitude,
)
from echo_spin.control.rotations import single_qubit_rotation
from echo_spin.core.operators import embed_operator
from echo_spin.decoupling.modulation import modulation_function
from echo_spin.dynamics.propagators import static_propagator
from echo_spin.dynamics.sequences import finite_pulse_propagator
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


def implement_finite_protocols(
    protocol_report,
    pulse_duration_us,
    *,
    shape="square",
    steps=128,
    amplitude_error=0.0,
    amplitude_error_estimate=0.0,
) -> dict:
    """Implement finite square or sine pulses with a common amplitude error."""
    if shape not in ("square", "sine"):
        raise ValueError("shape must be 'square' or 'sine'.")

    if (
        isinstance(steps, (bool, np.bool_))
        or not isinstance(steps, (int, np.integer))
        or steps < 1
    ):
        raise ValueError("steps must be a positive integer.")

    if (
        isinstance(pulse_duration_us, (bool, np.bool_))
        or not isinstance(
            pulse_duration_us,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(pulse_duration_us)
        or pulse_duration_us <= 0
    ):
        raise ValueError("pulse_duration_us must be finite and positive.")

    if (
        isinstance(amplitude_error, (bool, np.bool_))
        or not isinstance(
            amplitude_error,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(amplitude_error)
        or amplitude_error < -1
    ):
        raise ValueError(
            "amplitude_error must be finite and at least -1."
        )

    amplitude_error = float(amplitude_error)

    if (
        isinstance(amplitude_error_estimate, (bool, np.bool_))
        or not isinstance(
            amplitude_error_estimate,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(amplitude_error_estimate)
        or amplitude_error_estimate <= -1
    ):
        raise ValueError(
            "amplitude_error_estimate must be finite and greater than -1."
        )

    amplitude_error_estimate = float(amplitude_error_estimate)
    pulse_duration = float(pulse_duration_us)
    ideal = implement_ideal_protocols(protocol_report)
    model = ideal["model"]

    peak_amplitude = (
        np.pi / pulse_duration
        if shape == "square"
        else sine_pi_pulse_amplitude(pulse_duration)
    )
    peak_amplitude *= (
        (1 + amplitude_error) / (1 + amplitude_error_estimate)
    )
    implementations = {}

    for name, control in ideal["protocols"].items():
        free_intervals = control["free_intervals_us"].copy()
        free_intervals[:-1] -= pulse_duration

        if np.any(free_intervals < 0):
            raise ValueError(
                f"Finite pulse windows overlap or precede t=0 for {name}."
            )

        pulses = []
        for sites in control["addressed_sites"]:
            control_operator = 0.0 * model.hamiltonian
            for site in sites:
                control_operator += (
                    model.pauli_operators["X"][site] / 2
                )

            if shape == "square":
                pulse = static_propagator(
                    model.hamiltonian
                    + peak_amplitude * control_operator,
                    pulse_duration,
                )
            else:
                pulse = finite_pulse_propagator(
                    free_hamiltonian=model.hamiltonian,
                    control_operator=control_operator,
                    duration=pulse_duration,
                    envelope=lambda time: sine_envelope(
                        time - pulse_duration / (2 * steps),
                        pulse_duration,
                        peak_amplitude,
                    ),
                    steps=int(steps),
                )

            pulses.append(pulse)

        implementations[name] = {
            "event_times_us": control["event_times_us"].copy(),
            "pulse_start_times_us": (
                control["event_times_us"] - pulse_duration
            ),
            "addressed_sites": control["addressed_sites"],
            "free_intervals_us": free_intervals,
            "pulse_propagators": tuple(pulses),
        }

    return {
        "implementation": f"finite_{shape}_local_X_pi",
        "model": model,
        "target": ideal["target"],
        "duration_us": ideal["duration_us"],
        "pulse_duration_us": pulse_duration,
        "pulse_shape": shape,
        "amplitude_error": amplitude_error,
        "amplitude_error_estimate": amplitude_error_estimate,
        "peak_rabi_frequency_rad_per_us": float(peak_amplitude),
        "integration_steps": int(steps) if shape == "sine" else None,
        "protocols": implementations,
    }


def implement_square_protocols(protocol_report, pulse_duration_us) -> dict:
    """Implement square pulses through the general finite-control builder."""
    result = implement_finite_protocols(
        protocol_report,
        pulse_duration_us,
        shape="square",
    )
    result["rabi_frequency_rad_per_us"] = (
        result["peak_rabi_frequency_rad_per_us"]
    )
    return result
