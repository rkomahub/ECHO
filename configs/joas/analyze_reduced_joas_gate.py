import argparse
import json

import numpy as np
from qutip import Qobj, basis, tensor

from echo_spin.dynamics.sequences import finite_pulse_intervals
from echo_spin.gates.gates import sqrt_zz_gate
from echo_spin.models.effective import (
    explicit_matrix_product,
    resonant_control_operator,
    second_order_secular_hamiltonian,
    secular_hamiltonian,
)
from echo_spin.models.joas import xy8_gate_schedule


def unitary_step(hamiltonian, duration):
    """Exponentiate a small Hermitian Hamiltonian through explicit contractions."""
    energies, vectors = np.linalg.eigh(hamiltonian)
    return explicit_matrix_product(
        vectors * np.exp(-1j * energies * duration),
        vectors.conj().T,
    )


def analyze_archive(path, static_order=1, electronic_levels=2):
    """Compare a nuclear-memory-preserving RWA gate with the Joas reference."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].item())
        physical_dims = [[3, 3, 3, 3], [3, 3, 3, 3]]
        free = Qobj(
            archive["model_free_hamiltonian"],
            dims=physical_dims,
        )
        frame = Qobj(
            archive["model_frame_generator"],
            dims=physical_dims,
        )
        control = Qobj(
            archive["model_shared_control"],
            dims=physical_dims,
        )

    if (
        metadata["drive_mode"] != "shared"
        or metadata["hyperfine_mode"] != "full"
    ):
        raise ValueError("Provide a shared-drive, full-hyperfine archive.")

    # Ground and addressed excited states come first in each electronic basis.
    electron_one_levels = [1, 0] if electronic_levels == 2 else [1, 0, 2]
    electron_two_levels = [1, 2] if electronic_levels == 2 else [1, 2, 0]

    embeddings = [
        np.column_stack([
            tensor(
                basis(3, e1),
                basis(3, n1),
                basis(3, e2),
                basis(3, n2),
            ).full().ravel()
            for e1 in electron_one_levels
            for e2 in electron_two_levels
        ])
        for n1 in range(3)
        for n2 in range(3)
    ]

    embedding = np.column_stack(embeddings)
    electron_dimension = electronic_levels**2
    joint_dimension = 9 * electron_dimension

    logical_indices = [
        first * electronic_levels + second
        for first in [0, 1]
        for second in [0, 1]
    ]

    logical_embedding = np.eye(
        electron_dimension,
        dtype=complex,
    )[:, logical_indices]

    joint_logical_embedding = np.kron(
        np.eye(9),
        logical_embedding,
    )

    print(
        "Electronic levels per NV =",
        electronic_levels,
        "; joint dimension =",
        joint_dimension,
    )

    if static_order == 2:
        corrected = second_order_secular_hamiltonian(free, frame)
        secular = corrected["hamiltonian"]

        print(
            "Static correction norm / (2*pi) =",
            corrected["correction_norm"] / (2 * np.pi),
            "MHz",
        )
        print(
            "Perturbative transformation norm =",
            corrected["transformation_norm"],
        )
    else:
        secular = secular_hamiltonian(free, frame)

    print("Static approximation order =", static_order)

    drift = explicit_matrix_product(
        embedding.conj().T,
        (secular - frame).full(),
        embedding,
    )

    duration = metadata["pi_pulse_duration"]
    gate_duration = metadata["gate_duration"]
    peak = np.pi**2 / (2 * duration)

    schedule = xy8_gate_schedule(
        tau_1=metadata["tau_1"],
        tau_2=metadata["tau_2"],
        n_pi=8,
    )
    intervals = finite_pulse_intervals(
        pulse_centers=[center for center, _, _ in schedule],
        pulse_duration=duration,
        total_duration=gate_duration,
    )

    controls = {}

    for nv, carrier in enumerate(metadata["carriers"], start=1):
        for axis, phase in [("x", 0.0), ("y", np.pi / 2)]:
            resonant = resonant_control_operator(
                control=control,
                frame_generator=frame,
                carrier=carrier,
                phase=phase,
            )
            controls[nv, axis] = explicit_matrix_product(
                embedding.conj().T,
                resonant.full(),
                embedding,
            )

    def build_gate(steps):
        """Propagate the RWA sequence through explicit midpoint contractions."""
        propagator = np.eye(joint_dimension, dtype=complex)
        dt = duration / steps

        for index, (_, nv, axis) in enumerate(schedule):
            propagator = explicit_matrix_product(
                unitary_step(drift, intervals[index]),
                propagator,
            )

            for step in range(steps):
                time = (step + 0.5) * dt
                envelope = peak * np.sin(np.pi * time / duration)
                propagator = explicit_matrix_product(
                    unitary_step(
                        drift + envelope * controls[nv, axis],
                        dt,
                    ),
                    propagator,
                )

        return explicit_matrix_product(
            unitary_step(drift, intervals[-1]),
            propagator,
        )

    target = sqrt_zz_gate().dag().full()
    plus = np.ones(4, dtype=complex) / 2
    initial = np.outer(plus, plus.conj())
    counts = [0, 1, 2, 4, 8, 16]

    def gate_metrics(propagator):
        """Evaluate logical metrics while allowing discarded electronic population to return."""
        projected = explicit_matrix_product(
            joint_logical_embedding.conj().T,
            propagator,
            joint_logical_embedding,
        )

        blocks = projected.reshape(9, 4, 9, 4)
        survival_weight = 0.0
        overlap_weight = 0.0

        for output in range(9):
            for initial_environment in range(9):
                kraus = blocks[output, :, initial_environment, :] / 3
                survival_weight += np.linalg.norm(kraus, "fro") ** 2
                overlap_weight += abs(
                    np.trace(
                        explicit_matrix_product(target.conj().T, kraus)
                    )
                ) ** 2

        fidelity = (survival_weight + overlap_weight) / 20
        survival = survival_weight / 4

        electronic_initial = explicit_matrix_product(
            logical_embedding,
            initial,
            logical_embedding.conj().T,
        )

        density = np.kron(np.eye(9) / 9, electronic_initial)
        signals = []

        for count in range(max(counts) + 1):
            if count in counts:
                electrons = np.einsum(
                    "aiaj->ij",
                    density.reshape(
                        9, electron_dimension,
                        9, electron_dimension,
                    ),
                )

                logical_density = explicit_matrix_product(
                    logical_embedding.conj().T,
                    electrons,
                    logical_embedding,
                )

                recovery = np.linalg.matrix_power(
                    target.conj().T,
                    count,
                )

                recovered = explicit_matrix_product(
                    recovery,
                    logical_density,
                    recovery.conj().T,
                )

                signals.append(
                    float(
                        np.trace(
                            explicit_matrix_product(initial, recovered)
                        ).real
                    )
                )

            if count < max(counts):
                density = explicit_matrix_product(
                    propagator,
                    density,
                    propagator.conj().T,
                )

        return survival, fidelity, np.asarray(signals)

    results = []

    for steps in [128, 256]:
        result = gate_metrics(build_gate(steps))
        results.append(result)
        print(
            f"\nRWA steps/pulse={steps}: "
            f"survival={result[0]:.8f}, fidelity={result[1]:.8f}"
        )
        print("Repetitions =", counts)
        print("Retained-nuclear RWA return =", result[2])

    reference = np.array([
        1.0, 0.98124782, 0.98527507,
        0.97538142, 0.92666832, 0.81623458,
    ])

    print(
        "\nFidelity convergence difference =",
        abs(results[1][1] - results[0][1]),
    )
    print(
        "Maximum return convergence difference =",
        np.max(np.abs(results[1][2] - results[0][2])),
    )
    print(
        "RWA minus full-reference fidelity =",
        results[1][1] - 0.9829250468336689,
    )
    print("RWA minus full-reference return =", results[1][2] - reference)
    print("Comparison excludes phenomenological dephasing in both models.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument(
        "--static-order",
        type=int,
        choices=[1, 2],
        default=1,
    )
    parser.add_argument(
        "--electronic-levels",
        type=int,
        choices=[2, 3],
        default=2,
    )
    arguments = parser.parse_args()
    analyze_archive(
        arguments.archive,
        arguments.static_order,
        arguments.electronic_levels,
    )

