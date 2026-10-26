import numpy as np

from echo_spin.dynamics.sequences import finite_pulse_sequence_propagator
from echo_spin.implementation import implement_ideal_protocols


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

        overlap = complex((target.dag() * propagator).tr())
        fidelity = (abs(overlap)**2 + dimension) / (
            dimension * (dimension + 1)
        )

        global_phase = float(np.angle(overlap))
        aligned = np.exp(-1j * global_phase) * propagator.full()
        phase_aligned_error = np.linalg.norm(
            aligned - target.full(), ord="fro"
        ) / np.sqrt(dimension)

        unitary = propagator.full()
        unitarity_error = np.linalg.norm(
            unitary.conj().T @ unitary - np.eye(dimension),
            ord="fro",
        ) / np.sqrt(dimension)

        results[name] = {
            "propagator": propagator,
            "average_gate_fidelity": float(fidelity),
            "phase_aligned_frobenius_error": float(phase_aligned_error),
            "unitarity_error": float(unitarity_error),
        }

    return {
        "duration_us": implemented["duration_us"],
        "dimension": dimension,
        "target": target,
        "protocols": results,
        "assessment": dict(model.assessment),
    }
