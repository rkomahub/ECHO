import numpy as np

from echo_spin.decoupling.modulation import effective_zz_coupling
from echo_spin.dynamics.sequences import finite_pulse_sequence_propagator
from echo_spin.implementation import (
    implement_finite_protocols,
    implement_ideal_protocols,
)


def ideal_phase_diagnostics(protocol_report, name) -> dict:
    """Calculate unwrapped longitudinal generator phases from saved schedules."""
    model = protocol_report["source_model"]
    schedules = protocol_report["protocols"][name]["pulse_times_us"]
    duration = protocol_report["duration_us"]
    detunings = np.asarray(model["detunings_rad_per_us"], dtype=float)
    couplings = np.asarray(model["couplings_rad_per_us"], dtype=float)
    number_of_nv = len(schedules)

    local_phases = np.array([
        duration * effective_zz_coupling(
            detunings[site] / 2,
            duration,
            schedules[site],
            [],
        )
        for site in range(number_of_nv)
    ])

    pair_phases = np.zeros((number_of_nv, number_of_nv))
    for i in range(number_of_nv):
        for j in range(i + 1, number_of_nv):
            phase = duration * effective_zz_coupling(
                couplings[i, j],
                duration,
                schedules[i],
                schedules[j],
            )
            pair_phases[i, j] = phase
            pair_phases[j, i] = phase

    a, b = protocol_report["target_pair"]
    target_phases = np.zeros_like(pair_phases)
    target_phases[a, b] = np.sign(couplings[a, b]) * np.pi / 4
    target_phases[b, a] = target_phases[a, b]

    return {
        "local_generator_phases_rad": local_phases.tolist(),
        "pair_generator_phases_rad": pair_phases.tolist(),
        "residual_pair_generator_phases_rad": (
            pair_phases - target_phases
        ).tolist(),
    }


def _unitary_metrics(target, propagator) -> dict:
    """Evaluate whole-register unitary errors without local corrections."""
    dimension = target.shape[0]
    overlap = complex((target.dag() * propagator).tr())
    fidelity = (abs(overlap)**2 + dimension) / (
        dimension * (dimension + 1)
    )

    unitary = propagator.full()
    if not np.all(np.isfinite(unitary)):
        raise ValueError("Propagator contains nonfinite entries.")

    gram = np.einsum(
        "ki,kj->ij", unitary.conj(), unitary, optimize=False
    )

    aligned = np.exp(-1j * np.angle(overlap)) * unitary

    return {
        "propagator": propagator,
        "average_gate_fidelity": float(fidelity),
        "phase_aligned_frobenius_error": float(
            np.linalg.norm(
                aligned - target.full(), ord="fro"
            ) / np.sqrt(dimension)
        ),
        "unitarity_error": float(
            np.linalg.norm(
                gram - np.eye(dimension),
                ord="fro",
            ) / np.sqrt(dimension)
        ),
    }


def evaluate_ideal_implementation(report) -> dict:
    """Propagate saved ideal controls and evaluate whole-register unitary fidelity."""
    if report["implementation"] != "instantaneous_local_X_pi":
        raise ValueError("Unsupported control implementation.")

    implemented = implement_ideal_protocols(report["source_protocol_report"])
    if report["duration_us"] != implemented["duration_us"]:
        raise ValueError("Saved duration does not match the reconstructed controls.")

    if set(report["protocols"]) != set(implemented["protocols"]):
        raise ValueError("Saved protocol names do not match the source.")

    model = implemented["model"]
    target = implemented["target"]
    dimension = model.system.dimension
    results = {}

    for name, control in implemented["protocols"].items():
        saved = report["protocols"][name]

        # Do not silently ignore edits to the saved implementation timeline.
        for field in ("event_times_us", "free_intervals_us"):
            expected = control[field]
            actual = np.asarray(saved[field], dtype=float)
            if (
                actual.shape != expected.shape
                or not np.all(np.isfinite(actual))
                or not np.allclose(
                    actual, expected, atol=1e-12, rtol=1e-12
                )
            ):
                raise ValueError(f"Saved {field} does not match the source.")

        if saved["addressed_sites"] != control["addressed_sites"]:
            raise ValueError("Saved addressed sites do not match the source.")

        propagator = finite_pulse_sequence_propagator(
            free_hamiltonian=model.hamiltonian,
            durations=control["free_intervals_us"],
            pulse_propagators=control["pulse_propagators"],
        )

        results[name] = _unitary_metrics(target, propagator)
        unitary = propagator.full()

        results[name]["phases"] = ideal_phase_diagnostics(
            report["source_protocol_report"],
            name,
        )
        results[name]["resources"] = {
            "duration_us": float(implemented["duration_us"]),
            "individual_pulse_count": sum(
                len(sites) for sites in control["addressed_sites"]
            ),
            "distinct_pulse_time_count": len(control["event_times_us"]),
        }

        phases = results[name]["phases"]
        source_model = report["source_protocol_report"]["source_model"]
        generator_diagonal = np.full(
            dimension,
            source_model["constant_rad_per_us"] * implemented["duration_us"],
            dtype=float,
        )
        z_diagonals = [
            operator.diag().real
            for operator in model.pauli_operators["Z"]
        ]

        for i, phase in enumerate(phases["local_generator_phases_rad"]):
            generator_diagonal += phase * z_diagonals[i]

        pair_phases = phases["pair_generator_phases_rad"]
        for i in range(model.system.number_of_spins):
            for j in range(i + 1, model.system.number_of_spins):
                generator_diagonal += (
                    pair_phases[i][j] * z_diagonals[i] * z_diagonals[j]
                )

        pulse_count = results[name]["resources"]["individual_pulse_count"]
        predicted = np.diag(
            (-1j)**pulse_count * np.exp(-1j * generator_diagonal)
        )
        results[name]["phase_prediction_error"] = float(
            np.linalg.norm(unitary - predicted, ord="fro")
            / np.sqrt(dimension)
        )

    return {
        "duration_us": implemented["duration_us"],
        "dimension": dimension,
        "target": target,
        "protocols": results,
        "assessment": dict(model.assessment),
    }


def ideal_evaluation_report(implementation_report) -> dict:
    """Export ideal register operations, performance metrics and phase diagnostics."""
    evaluation = evaluate_ideal_implementation(implementation_report)
    target = evaluation["target"].full()
    protocols = {}

    for name, result in evaluation["protocols"].items():
        propagator = result["propagator"].full()
        protocols[name] = {
            "propagator": {
                "real": propagator.real.tolist(),
                "imag": propagator.imag.tolist(),
            },
            "average_gate_fidelity": result["average_gate_fidelity"],
            "phase_aligned_frobenius_error": (
                result["phase_aligned_frobenius_error"]
            ),
            "unitarity_error": result["unitarity_error"],
            "phase_prediction_error": result["phase_prediction_error"],
            "phases": result["phases"],
            "resources": result["resources"],
        }

    return {
        "report_version": 1,
        "source_implementation": implementation_report,
        "dimension": evaluation["dimension"],
        "duration_us": evaluation["duration_us"],
        "target": {
            "real": target.real.tolist(),
            "imag": target.imag.tolist(),
        },
        "protocols": protocols,
        "assessment": evaluation["assessment"],
        "conventions": {
            "fidelity": "Whole-register average gate fidelity for unitary maps.",
            "operator_error": (
                "Frobenius distance divided by sqrt(d), "
                "minimised over a common global phase."
            ),
            "unitarity_error": (
                "Frobenius norm of U†U-I divided by sqrt(d)."
            ),
            "phases": (
                "Unwrapped generator coefficients in radians; "
                "pair sums use i<j."
            ),
            "local_corrections": "None.",
            "tensor_order": "site_0_left",
            "phase_prediction_error": (
                "Frobenius distance between propagated and analytically predicted "
                "operations divided by sqrt(d), including global phase."
            ),
        },
        "limits": [
            "Results concern the ideal longitudinal qubit model.",
            "Physical leakage and nuclear dynamics are not represented.",
            "Independent instantaneous control is assumed.",
            "Static projection diagnostics do not validate physical gate fidelity.",
            "Fidelity can differ from its mathematical bounds by roundoff.",
        ],
    }


def evaluate_finite_protocols(
    protocol_report,
    pulse_duration_us,
    *,
    shape="square",
    steps=128,
    amplitude_error=0.0,
) -> dict:
    """Evaluate finite-pulse register operations against the ideal target."""
    implemented = implement_finite_protocols(
        protocol_report,
        pulse_duration_us,
        shape=shape,
        steps=steps,
        amplitude_error=amplitude_error,
    )
    model = implemented["model"]
    target = implemented["target"]
    results = {}

    for name, control in implemented["protocols"].items():
        propagator = finite_pulse_sequence_propagator(
            free_hamiltonian=model.hamiltonian,
            durations=control["free_intervals_us"],
            pulse_propagators=control["pulse_propagators"],
        )
        results[name] = _unitary_metrics(target, propagator)
        results[name]["resources"] = {
            "duration_us": float(implemented["duration_us"]),
            "individual_pulse_count": sum(
                len(sites) for sites in control["addressed_sites"]
            ),
            "distinct_pulse_time_count": len(
                control["event_times_us"]
            ),
        }

    return {
        "implementation": implemented["implementation"],
        "duration_us": implemented["duration_us"],
        "dimension": model.system.dimension,
        "target": target,
        "pulse_shape": implemented["pulse_shape"],
        "pulse_duration_us": implemented["pulse_duration_us"],
        "peak_rabi_frequency_rad_per_us": (
            implemented["peak_rabi_frequency_rad_per_us"]
        ),
        "integration_steps": implemented["integration_steps"],
        "protocols": results,
        "assessment": dict(model.assessment),
        "amplitude_error": implemented["amplitude_error"],
    }


def finite_evaluation_report(
    protocol_report,
    pulse_duration_us,
    *,
    shape="square",
    steps=128,
    amplitude_error=0.0,
) -> dict:
    """Export finite-pulse operations and their reconstruction settings."""
    evaluation = evaluate_finite_protocols(
        protocol_report,
        pulse_duration_us,
        shape=shape,
        steps=steps,
        amplitude_error=amplitude_error,
    )
    target = evaluation["target"].full()
    protocols = {}

    for name, result in evaluation["protocols"].items():
        propagator = result["propagator"].full()
        protocols[name] = {
            "propagator": {
                "real": propagator.real.tolist(),
                "imag": propagator.imag.tolist(),
            },
            "average_gate_fidelity": result["average_gate_fidelity"],
            "phase_aligned_frobenius_error": (
                result["phase_aligned_frobenius_error"]
            ),
            "unitarity_error": result["unitarity_error"],
            "resources": dict(result["resources"]),
        }

    return {
        "report_version": 1,
        "source_protocol_report": protocol_report,
        **{
            key: evaluation[key]
            for key in (
                "implementation",
                "duration_us",
                "dimension",
                "pulse_shape",
                "pulse_duration_us",
                "amplitude_error",
                "peak_rabi_frequency_rad_per_us",
                "integration_steps",
                "assessment",
            )
        },
        "target": {
            "real": target.real.tolist(),
            "imag": target.imag.tolist(),
        },
        "protocols": protocols,
        "conventions": {
            "pulse_timing": "Finite windows end at ideal event times.",
            "control_hamiltonian": "Omega(t) * sum_addressed(X_i) / 2",
            "fidelity": "Whole-register average gate fidelity.",
            "local_corrections": "None.",
            "tensor_order": "site_0_left",
        },
        "limits": [
            "Longitudinal qubit drift with independent local controls.",
            "No physical leakage, nuclear dynamics or noise.",
            "Sine propagation requires time-resolution convergence checks.",
            "Instantaneous sign-integrated phase diagnostics do not apply.",
        ],
    }
