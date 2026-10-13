import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from qutip import Qobj, qeye
from scipy.optimize import curve_fit

from echo_spin.control.rotations import single_qubit_rotation
from echo_spin.gates.gates import sqrt_zz_gate


def single_qubit_cliffords() -> tuple:
    """Generate the 24 single-qubit Clifford representatives modulo global phase."""
    generators = [
        single_qubit_rotation(np.pi / 2, "x"),
        single_qubit_rotation(np.pi / 2, "y"),
    ]

    cliffords = [qeye(2)]
    next_index = 0

    while next_index < len(cliffords):
        current = cliffords[next_index]
        next_index += 1

        for generator in generators:
            candidate = generator * current

            already_present = any(
                np.isclose(
                    abs((existing.dag() * candidate).tr()),
                    2.0,
                    atol=1e-10,
                    rtol=0.0,
                )
                for existing in cliffords
            )

            if not already_present:
                cliffords.append(candidate)

                if len(cliffords) > 24:
                    raise RuntimeError("Clifford generation did not close.")

    if len(cliffords) != 24:
        raise RuntimeError("Expected 24 single-qubit Cliffords.")

    return tuple(cliffords)


def random_single_qubit_clifford_sequence(
    length: int,
    rng: np.random.Generator,
) -> dict:
    """Sample uniform single-qubit Cliffords and compute their chronological recovery."""
    if (
        isinstance(length, (bool, np.bool_))
        or not isinstance(length, (int, np.integer))
        or length < 0
    ):
        raise ValueError("length must be a non-negative integer.")

    cliffords = single_qubit_cliffords()
    indices = rng.integers(0, len(cliffords), size=length)

    gates = tuple(cliffords[int(index)] for index in indices)
    total = qeye(2)

    for gate in gates:
        total = gate * total

    return {
        "indices": indices,
        "gates": gates,
        "recovery": total.dag(),
    }


def single_qubit_rb_signal(
    initial_density,
    lengths,
    sequences_per_length,
    rng,
    gate_map,
    recovery_map,
    readout,
) -> dict:
    """Average single-qubit Clifford sequences with explicit gate, recovery and readout maps."""
    lengths = np.asarray(lengths)

    if (
        lengths.ndim != 1
        or lengths.size == 0
        or lengths.dtype.kind not in "iu"
        or np.any(lengths < 0)
    ):
        raise ValueError("lengths must be non-negative integers.")

    if (
        isinstance(sequences_per_length, (bool, np.bool_))
        or not isinstance(sequences_per_length, (int, np.integer))
        or sequences_per_length < 2
    ):
        raise ValueError("Use at least two sequences per length.")

    cliffords = single_qubit_cliffords()
    samples = np.empty((len(lengths), sequences_per_length))

    for row, length in enumerate(lengths):
        for column in range(sequences_per_length):
            state = initial_density.copy()
            ideal_total = qeye(2)

            indices = rng.integers(
                0, len(cliffords), size=int(length)
            )

            for index in indices:
                clifford = cliffords[int(index)]
                state = gate_map(state, clifford)
                ideal_total = clifford * ideal_total

                if state.dims != initial_density.dims:
                    raise ValueError("gate_map must preserve state dimensions.")

            state = recovery_map(state, ideal_total.dag())

            if state.dims != initial_density.dims:
                raise ValueError("recovery_map must preserve state dimensions.")

            value = complex(readout(state))

            if not np.isfinite(value) or abs(value.imag) > 1e-10:
                raise ValueError("readout must return a finite real value.")

            samples[row, column] = value.real

    return {
        "lengths": lengths.copy(),
        "samples": samples,
        "mean": samples.mean(axis=1),
        "standard_error": (
            samples.std(axis=1, ddof=1)
            / np.sqrt(sequences_per_length)
        ),
    }


def rb_error_per_clifford(decay_parameter: float, qubits: int) -> float:
    """Convert an RB decay parameter into error per Clifford for the stated dimension."""
    if (
        not np.isfinite(decay_parameter)
        or not 0 <= decay_parameter <= 1
    ):
        raise ValueError("decay_parameter must lie between zero and one.")

    if (
        isinstance(qubits, (bool, np.bool_))
        or not isinstance(qubits, (int, np.integer))
        or qubits < 1
    ):
        raise ValueError("qubits must be a positive integer.")

    return float(
        (1 - 2.0 ** (-int(qubits)))
        * (1 - decay_parameter)
    )


@lru_cache(maxsize=1)
def two_qubit_clifford_catalog() -> dict:
    """Build all two-qubit Cliffords modulo global phase and retain chronological generator words."""
    identity = np.eye(2, dtype=complex)
    rx = single_qubit_rotation(np.pi / 2, "x").full()
    ry = single_qubit_rotation(np.pi / 2, "y").full()

    generators = np.array([
        np.kron(rx, identity),
        np.kron(ry, identity),
        np.kron(identity, rx),
        np.kron(identity, ry),
        sqrt_zz_gate().dag().full(),
    ])

    paulis = [
        identity,
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.diag([1, -1]).astype(complex),
    ]
    pauli_basis = np.array([
        np.kron(first, second)
        for first in paulis
        for second in paulis
    ])

    # X1, Z1, X2, Z2 uniquely determine a Clifford modulo global phase.
    defining_paulis = pauli_basis[[4, 12, 1, 3]]

    def signature(unitary):
        """Identify a Clifford through its four signed generator-Pauli images."""
        transformed = (
            unitary @ defining_paulis
            @ unitary.conj().T
        )
        overlaps = np.einsum(
            "aij,bji->ab",
            pauli_basis,
            transformed,
        ).real / 4

        positions = np.argmax(np.abs(overlaps), axis=0)
        selected = overlaps[positions, np.arange(4)]

        if not np.allclose(
            np.abs(selected), 1.0,
            atol=1e-10, rtol=0.0,
        ):
            raise RuntimeError("Generated operator is not Clifford.")

        return tuple(
            int(position + 1) * (1 if value > 0 else -1)
            for position, value in zip(positions, selected)
        )

    matrices = [np.eye(4, dtype=complex)]
    words = [()]
    known = {signature(matrices[0])}
    next_index = 0

    while next_index < len(matrices):
        current = matrices[next_index]
        word = words[next_index]
        next_index += 1

        for generator_index, generator in enumerate(generators):
            candidate = generator @ current
            key = signature(candidate)

            if key not in known:
                known.add(key)
                matrices.append(candidate)
                words.append(word + (generator_index,))

                if len(matrices) > 11520:
                    raise RuntimeError("Two-qubit Clifford generation did not close.")

    if len(matrices) != 11520:
        raise RuntimeError("Expected 11520 two-qubit Cliffords.")

    matrices = np.array(matrices)
    matrices.setflags(write=False)
    generators.setflags(write=False)

    return {
        "matrices": matrices,
        "words": tuple(words),
        "generators": generators,
        "generator_labels": (
            "RX1_pi_over_2",
            "RY1_pi_over_2",
            "RX2_pi_over_2",
            "RY2_pi_over_2",
            "sqrt_ZZ_dagger",
        ),
    }


def random_two_qubit_clifford_sequence(length, rng) -> dict:
    """Sample uniform two-qubit Cliffords with generator words and ideal recovery."""
    if (
        isinstance(length, (bool, np.bool_))
        or not isinstance(length, (int, np.integer))
        or length < 0
    ):
        raise ValueError("length must be a non-negative integer.")

    catalog = two_qubit_clifford_catalog()
    indices = rng.integers(
        0, len(catalog["matrices"]), size=length
    )

    gates = tuple(
        Qobj(
            catalog["matrices"][int(index)],
            dims=[[2, 2], [2, 2]],
        )
        for index in indices
    )

    total = qeye([2, 2])

    for gate in gates:
        total = gate * total

    recovery_matrix = total.dag().full()

    # Identify recovery modulo global phase in the complete Clifford catalogue.
    overlaps = np.einsum(
        "nij,ij->n",
        catalog["matrices"].conj(),
        recovery_matrix,
    ) / 4

    recovery_index = int(np.argmax(np.abs(overlaps)))

    if not np.isclose(
        abs(overlaps[recovery_index]),
        1.0,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError("Recovery was not found in the Clifford catalogue.")

    return {
        "indices": indices,
        "gates": gates,
        "words": tuple(
            catalog["words"][int(index)]
            for index in indices
        ),
        "recovery": total.dag(),
        "recovery_index": recovery_index,
        "recovery_word": catalog["words"][recovery_index],
    }


def two_qubit_native_gate_counts(word) -> tuple:
    """Count single-qubit and entangling primitives in an unoptimized generator word."""
    word = tuple(word)

    for index in word:
        if (
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            or not 0 <= index < 5
        ):
            raise ValueError("Generator indices must be integers from zero to four.")

    return (
        sum(index < 4 for index in word),
        sum(index == 4 for index in word),
    )


def two_qubit_rb_signal(
    initial_density,
    lengths,
    sequences_per_length,
    rng,
    primitive_map,
    readout,
    primitive_map_factory=None,
) -> dict:
    """Average compiled two-qubit RB circuits with the same primitive map during recovery."""
    lengths = np.asarray(lengths)

    if (
        lengths.ndim != 1
        or lengths.size == 0
        or lengths.dtype.kind not in "iu"
        or np.any(lengths < 0)
    ):
        raise ValueError("lengths must be non-negative integers.")

    if (
        isinstance(sequences_per_length, (bool, np.bool_))
        or not isinstance(sequences_per_length, (int, np.integer))
        or sequences_per_length < 2
    ):
        raise ValueError("Use at least two sequences per length.")

    if (primitive_map is None) == (primitive_map_factory is None):
        raise ValueError(
            "Provide either primitive_map or primitive_map_factory."
        )

    shape = (len(lengths), sequences_per_length)
    samples = np.empty(shape)
    gate_counts = np.zeros(shape + (2,), dtype=int)
    recovery_counts = np.zeros(shape + (2,), dtype=int)
    sampled_indices = []

    for row, length in enumerate(lengths):
        row_indices = []

        for column in range(sequences_per_length):
            sequence = random_two_qubit_clifford_sequence(
                int(length),
                rng,
            )
            row_indices.append(
                tuple(int(index) for index in sequence["indices"])
            )
            state = initial_density.copy()

            active_map = (
                primitive_map_factory()
                if primitive_map_factory is not None
                else primitive_map
            )

            for word in sequence["words"]:
                gate_counts[row, column] += (
                    two_qubit_native_gate_counts(word)
                )

                for generator_index in word:
                    state = active_map(state, generator_index)

                    if state.dims != initial_density.dims:
                        raise ValueError("primitive_map must preserve dimensions.")

            recovery_word = sequence["recovery_word"]
            recovery_counts[row, column] = (
                two_qubit_native_gate_counts(recovery_word)
            )

            for generator_index in recovery_word:
                state = active_map(state, generator_index)

                if state.dims != initial_density.dims:
                    raise ValueError("primitive_map must preserve dimensions.")

            value = complex(readout(state))

            if not np.isfinite(value) or abs(value.imag) > 1e-10:
                raise ValueError("readout must return a finite real value.")

            samples[row, column] = value.real

        sampled_indices.append(tuple(row_indices))

    return {
        "lengths": lengths.copy(),
        "samples": samples,
        "mean": samples.mean(axis=1),
        "standard_error": (
            samples.std(axis=1, ddof=1)
            / np.sqrt(sequences_per_length)
        ),
        "gate_counts": gate_counts,
        "recovery_counts": recovery_counts,
        "clifford_indices": tuple(sampled_indices),
    }


def timed_primitive_map(
    builders,
    frame_generator: Qobj,
    start_time: float = 0.0,
    post_map=None,
):
    """Create a fresh circuit clock using laboratory propagators and one common rotating frame."""
    if (
        not frame_generator.isoper
        or not frame_generator.isherm
        or not np.all(np.isfinite(frame_generator.full()))
    ):
        raise ValueError("frame_generator must be a finite Hermitian operator.")

    if not np.isfinite(start_time):
        raise ValueError("start_time must be finite.")

    builders = dict(builders)
    clock = float(start_time)

    def primitive_map(density_matrix, generator_index):
        """Build a pulse at the current absolute time and advance the circuit clock."""
        nonlocal clock

        lab_propagator, duration = builders[generator_index](clock)

        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Primitive duration must be finite and positive.")

        if (
            lab_propagator.dims != frame_generator.dims
            or density_matrix.dims != frame_generator.dims
        ):
            raise ValueError("Primitive, frame and state dimensions must match.")

        matrix = lab_propagator.full()
        defect = np.linalg.norm(
            matrix.conj().T @ matrix - np.eye(matrix.shape[0]),
            ord="fro",
        )

        if not np.isfinite(defect) or defect > 1e-8:
            raise ValueError(
                f"Primitive is not sufficiently unitary: defect={defect:.3e}."
            )

        end_time = clock + duration

        start_frame = (-1j * frame_generator * clock).expm()
        end_frame = (-1j * frame_generator * end_time).expm()

        rotating_propagator = (
            end_frame.dag()
            * lab_propagator
            * start_frame
        )

        result = (
            rotating_propagator
            * density_matrix
            * rotating_propagator.dag()
        )

        if post_map is not None:
            result = post_map(result, duration)

            if (
                not isinstance(result, Qobj)
                or result.dims != density_matrix.dims
            ):
                raise ValueError(
                    "post_map must preserve the state dimensions."
                )

        clock = end_time
        return result

    return primitive_map


def save_two_qubit_rb_result(
    path,
    result,
    metadata,
    model_arrays=None,
):
    """Archive RB samples, compiler words and model inputs without overwriting existing results."""
    path = Path(path)

    if path.suffix != ".npz":
        raise ValueError("Result path must end in .npz.")

    catalog = two_qubit_clifford_catalog()

    payload = {
        name: np.asarray(result[name])
        for name in [
            "lengths",
            "samples",
            "mean",
            "standard_error",
            "gate_counts",
            "recovery_counts",
        ]
    }

    payload.update({
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
        "clifford_indices_json": np.array(
            json.dumps(result["clifford_indices"])
        ),
        "catalog_words_json": np.array(
            json.dumps(catalog["words"])
        ),
        "generator_labels_json": np.array(
            json.dumps(catalog["generator_labels"])
        ),
        "generators": catalog["generators"],
    })

    if model_arrays is not None:
        for name, value in model_arrays.items():
            array = np.asarray(value)

            if array.dtype.hasobject:
                raise ValueError("Model arrays must not contain Python objects.")

            payload[f"model_{name}"] = array

    path.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation protects previous experimental results.
    with path.open("xb") as output:
        np.savez_compressed(output, **payload)

    return path


def fit_rb_decay(
    lengths,
    signal,
    qubits,
    offset,
    standard_errors=None,
) -> dict:
    """Fit positive-contrast RB return probabilities with an independently fixed offset."""
    dimension_factor = rb_error_per_clifford(0.0, qubits)
    lengths = np.asarray(lengths)

    if (
        lengths.ndim != 1
        or lengths.dtype.kind not in "iu"
        or np.any(lengths < 0)
        or len(np.unique(lengths)) < 4
    ):
        raise ValueError(
            "Provide at least four distinct non-negative integer lengths."
        )

    if np.iscomplexobj(signal):
        raise ValueError("signal must be real.")

    values = np.asarray(signal, dtype=float)

    if (
        values.shape != lengths.shape
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
        or np.any(values > 1)
        or np.ptp(values) == 0
    ):
        raise ValueError("Provide finite, nonconstant return probabilities in [0, 1].")

    if not np.isfinite(offset) or not 0 <= offset < 1:
        raise ValueError("offset must be finite and lie in [0, 1).")

    sigma = None

    if standard_errors is not None:
        if np.iscomplexobj(standard_errors):
            raise ValueError("standard_errors must be real.")

        sigma = np.asarray(standard_errors, dtype=float)

        if (
            sigma.shape != values.shape
            or not np.all(np.isfinite(sigma))
            or np.any(sigma <= 0)
        ):
            raise ValueError("Standard errors must be finite and strictly positive.")

    def decay_model(length, amplitude, decay_parameter):
        """Return the RB probability with its independently specified offset."""
        return offset + amplitude * decay_parameter**length

    first = int(np.argmin(lengths))
    initial_amplitude = np.clip(
        values[first] - offset,
        0.0,
        1 - offset,
    )

    parameters, covariance = curve_fit(
        decay_model,
        lengths,
        values,
        p0=[initial_amplitude, 0.97],
        bounds=([0.0, 0.0], [1 - offset, 1.0]),
        sigma=sigma,
        absolute_sigma=sigma is not None,
        maxfev=20000,
    )

    amplitude, decay_parameter = parameters
    fitted = decay_model(lengths, *parameters)
    residuals = values - fitted
    decay_standard_error = float(np.sqrt(covariance[1, 1]))

    return {
        "offset": float(offset),
        "amplitude": float(amplitude),
        "decay_parameter": float(decay_parameter),
        "decay_parameter_standard_error": decay_standard_error,
        "error_per_clifford": float(
            dimension_factor * (1 - decay_parameter)
        ),
        "error_per_clifford_standard_error": float(
            dimension_factor * decay_standard_error
        ),
        "fitted_signal": fitted,
        "residuals": residuals,
        "residual_rms": float(np.sqrt(np.mean(residuals**2))),
        "reduced_chi_squared": (
            float(np.sum((residuals / sigma) ** 2) / (len(values) - 2))
            if sigma is not None
            else None
        ),
        "parameter_covariance": covariance,
    }


def extract_entangling_gate_error(
    error_per_clifford,
    single_qubit_error,
    single_qubit_gates_per_clifford,
    entangling_gates_per_clifford,
) -> float:
    """Infer entangling error under the multiplicative native-gate error model."""
    epc = error_per_clifford
    single_error = single_qubit_error
    single_count = single_qubit_gates_per_clifford
    entangling_count = entangling_gates_per_clifford

    if not np.isfinite(epc) or not 0 <= epc < 1:
        raise ValueError("error_per_clifford must lie in [0, 1).")

    if not np.isfinite(single_error) or not 0 <= single_error < 1:
        raise ValueError("single_qubit_error must lie in [0, 1).")

    if not np.isfinite(single_count) or single_count < 0:
        raise ValueError("single-qubit gate count must be finite and non-negative.")

    if not np.isfinite(entangling_count) or entangling_count <= 0:
        raise ValueError("Entangling gate count must be finite and positive.")

    log_entangling_survival = (
        np.log1p(-epc)
        - single_count * np.log1p(-single_error)
    ) / entangling_count

    if log_entangling_survival > 1e-12:
        raise ValueError(
            "EPC is below the single-qubit contribution predicted by this model. "
            "Check the calibration, compiler counts and uncertainties."
        )

    # Permit only negligible numerical roundoff at zero inferred error.
    return float(
        -np.expm1(min(log_entangling_survival, 0.0))
    )
