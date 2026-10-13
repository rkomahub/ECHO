import json

import numpy as np
import pytest
from qutip import basis, qeye, sigmax, sigmay, sigmaz, tensor

from echo_spin.benchmarks.randomized import (
    estimate_entangling_gate_error,
    extract_entangling_gate_error,
    fit_rb_decay,
    random_single_qubit_clifford_sequence,
    random_two_qubit_clifford_sequence,
    rb_error_per_clifford,
    save_two_qubit_rb_result,
    single_qubit_cliffords,
    single_qubit_rb_signal,
    timed_primitive_map,
    two_qubit_clifford_catalog,
    two_qubit_native_gate_counts,
    two_qubit_rb_signal,
)
from echo_spin.noise.dephasing import diagonal_dephasing_channel


def test_single_qubit_cliffords_map_pauli_axes():
    """Verify that all 24 distinct Clifford representatives permute signed Pauli axes."""
    cliffords = single_qubit_cliffords()
    paulis = [sigmax(), sigmay(), sigmaz()]

    assert len(cliffords) == 24

    for index, clifford in enumerate(cliffords):
        np.testing.assert_allclose(
            (clifford.dag() * clifford).full(),
            np.eye(2),
            atol=1e-12,
            rtol=0.0,
        )

        for earlier in cliffords[:index]:
            assert abs(
                (earlier.dag() * clifford).tr()
            ) < 2 - 1e-8

        for pauli in paulis:
            transformed = clifford * pauli * clifford.dag()

            overlaps = np.array([
                float(np.real((axis * transformed).tr() / 2))
                for axis in paulis
            ])

            np.testing.assert_allclose(
                np.sort(np.abs(overlaps)),
                [0.0, 0.0, 1.0],
                atol=1e-12,
                rtol=0.0,
            )


@pytest.mark.parametrize("length", [0, 1, 10])
def test_random_clifford_sequence_recovery(length):
    """Verify exact ideal recovery after chronological randomized Clifford evolution."""
    sequence = random_single_qubit_clifford_sequence(
        length=length,
        rng=np.random.default_rng(42),
    )

    total = qeye(2)

    for gate in sequence["gates"]:
        total = gate * total

    recovered = sequence["recovery"] * total

    np.testing.assert_allclose(
        recovered.full(),
        np.eye(2),
        atol=1e-12,
        rtol=0.0,
    )
    assert len(sequence["indices"]) == length


def test_single_qubit_rb_matches_depolarizing_reference():
    """Verify randomized survival against exact depolarizing decay with ideal recovery."""
    decay_parameter = 0.97

    def gate_map(density_matrix, clifford):
        """Apply the Clifford followed by a trace-preserving depolarizing channel."""
        evolved = clifford * density_matrix * clifford.dag()
        return (
            decay_parameter * evolved
            + (1 - decay_parameter)
            * evolved.tr() * qeye(2) / 2
        )

    def recovery_map(density_matrix, recovery):
        """Apply ideal recovery without additional noise."""
        return recovery * density_matrix * recovery.dag()

    def readout(density_matrix):
        """Measure computational ground-state return probability."""
        return (
            basis(2, 0).dag()
            * density_matrix
            * basis(2, 0)
        )

    lengths = np.array([0, 1, 5, 15])

    result = single_qubit_rb_signal(
        initial_density=basis(2, 0).proj(),
        lengths=lengths,
        sequences_per_length=8,
        rng=np.random.default_rng(42),
        gate_map=gate_map,
        recovery_map=recovery_map,
        readout=readout,
    )

    expected = (1 + decay_parameter**lengths) / 2

    np.testing.assert_allclose(
        result["samples"],
        np.broadcast_to(expected[:, None], (4, 8)),
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "qubits, expected_error",
    [(1, 0.015), (2, 0.0225)],
)
def test_rb_error_per_clifford_dimension_factor(qubits, expected_error):
    """Verify the dimension-dependent conversion from RB decay to Clifford error."""
    assert rb_error_per_clifford(
        decay_parameter=0.97,
        qubits=qubits,
    ) == pytest.approx(expected_error, abs=1e-12)


def test_two_qubit_clifford_catalog_size_and_unitarity():
    """Verify full Clifford-group size and numerical unitarity of every representative."""
    catalog = two_qubit_clifford_catalog()
    matrices = catalog["matrices"]

    assert matrices.shape == (11520, 4, 4)
    assert len(catalog["words"]) == 11520

    products = (
        matrices.conj().transpose(0, 2, 1)
        @ matrices
    )

    np.testing.assert_allclose(
        products,
        np.broadcast_to(np.eye(4), products.shape),
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize("length", [0, 1, 10])
def test_two_qubit_random_sequence_words_and_recovery(length):
    """Verify chronological native words and ideal recovery of sampled two-qubit Cliffords."""
    catalog = two_qubit_clifford_catalog()
    sequence = random_two_qubit_clifford_sequence(
        length,
        np.random.default_rng(42),
    )

    total = np.eye(4, dtype=complex)

    for gate, word in zip(sequence["gates"], sequence["words"]):
        reconstructed = np.eye(4, dtype=complex)

        for generator_index in word:
            reconstructed = (
                catalog["generators"][generator_index]
                @ reconstructed
            )

        np.testing.assert_allclose(
            reconstructed,
            gate.full(),
            atol=1e-12,
            rtol=0.0,
        )
        total = gate.full() @ total

    np.testing.assert_allclose(
        sequence["recovery"].full() @ total,
        np.eye(4),
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize("length", [0, 1, 10])
def test_two_qubit_recovery_native_word(length):
    """Verify that compiled recovery cancels the ideal sequence modulo global phase."""
    catalog = two_qubit_clifford_catalog()
    sequence = random_two_qubit_clifford_sequence(
        length,
        np.random.default_rng(123),
    )

    total = np.eye(4, dtype=complex)

    for gate in sequence["gates"]:
        total = gate.full() @ total

    recovery = np.eye(4, dtype=complex)

    for index in sequence["recovery_word"]:
        recovery = catalog["generators"][index] @ recovery

    recovered = recovery @ total
    global_phase = np.trace(recovered) / 4

    assert abs(global_phase) == pytest.approx(1.0, abs=1e-12)

    np.testing.assert_allclose(
        recovered,
        global_phase * np.eye(4),
        atol=1e-12,
        rtol=0.0,
    )


def test_two_qubit_native_gate_counts():
    """Verify explicit primitive counts without assuming article compiler averages."""
    assert two_qubit_native_gate_counts(
        [0, 0, 2, 4, 1, 4]
    ) == (4, 2)

    assert two_qubit_native_gate_counts(()) == (0, 0)


@pytest.mark.parametrize("primitive_decay", [1.0, 0.98])
def test_two_qubit_rb_includes_noisy_compiled_recovery(primitive_decay):
    """Verify compiled RB against exact depolarizing survival for each circuit's primitive count."""
    catalog = two_qubit_clifford_catalog()
    initial_density = tensor(
        basis(2, 0),
        basis(2, 0),
    ).proj()

    def primitive_map(density_matrix, generator_index):
        """Apply a native ideal primitive followed by two-qubit depolarization."""
        matrix = catalog["generators"][generator_index]

        evolved_matrix = (
            matrix
            @ density_matrix.full()
            @ matrix.conj().T
        )
        evolved = density_matrix.copy()
        # Construct through the existing Qobj class, preserving subsystem dimensions.
        evolved = type(density_matrix)(
            evolved_matrix,
            dims=density_matrix.dims,
        )

        return (
            primitive_decay * evolved
            + (1 - primitive_decay)
            * evolved.tr() * qeye([2, 2]) / 4
        )

    def readout(density_matrix):
        """Measure computational two-qubit ground-state return."""
        return (initial_density * density_matrix).tr()

    result = two_qubit_rb_signal(
        initial_density=initial_density,
        lengths=[0, 1, 4],
        sequences_per_length=8,
        rng=np.random.default_rng(123),
        primitive_map=primitive_map,
        readout=readout,
    )

    total_primitives = (
        result["gate_counts"] + result["recovery_counts"]
    ).sum(axis=2)

    expected = (
        0.25 + 0.75 * primitive_decay**total_primitives
    )

    np.testing.assert_allclose(
        result["samples"],
        expected,
        atol=1e-12,
        rtol=0.0,
    )

    assert np.any(result["recovery_counts"][1:] > 0)


def test_rb_archive_preserves_data_without_pickle(tmp_path):
    """Verify numeric results and circuit provenance survive a pickle-free archive round trip."""
    result = {
        "lengths": np.array([0, 1]),
        "samples": np.array([[1.0, 1.0], [0.9, 0.8]]),
        "mean": np.array([1.0, 0.85]),
        "standard_error": np.array([0.0, 0.05]),
        "gate_counts": np.zeros((2, 2, 2), dtype=int),
        "recovery_counts": np.zeros((2, 2, 2), dtype=int),
        "clifford_indices": (((), ()), ((0,), (1,))),
    }

    path = save_two_qubit_rb_result(
        tmp_path / "rb.npz",
        result,
        metadata={"seed": 42},
        model_arrays={"example": np.eye(4, dtype=complex)},
    )

    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["samples"], result["samples"]
        )
        assert json.loads(
            archive["metadata_json"].item()
        ) == {"seed": 42}
        assert json.loads(
            archive["clifford_indices_json"].item()
        ) == [[[], []], [[0], [1]]]

    with pytest.raises(FileExistsError):
        save_two_qubit_rb_result(
            path, result, metadata={"seed": 42}
        )


def test_timed_primitive_dephasing_uses_elapsed_duration():
    """Verify per-primitive damping, accumulated duration and preserved input loss."""
    plus = (basis(2, 0) + basis(2, 1)).unit()
    initial_density = 0.73 * plus.proj()
    recorded_durations = []

    def first_builder(start_time):
        """Return an identity primitive lasting the first interval."""
        return qeye(2), 0.2

    def second_builder(start_time):
        """Return an identity primitive lasting the second interval."""
        return qeye(2), 0.3

    def post_map(density_matrix, duration):
        """Record duration and apply longitudinal dephasing without renormalizing."""
        recorded_durations.append(duration)
        return diagonal_dephasing_channel(
            density_matrix=density_matrix,
            time=duration,
            operators=[sigmaz() / 2],
            coherence_times=[2.0],
        )

    active_map = timed_primitive_map(
        builders={0: first_builder, 1: second_builder},
        frame_generator=0 * sigmaz(),
        post_map=post_map,
    )

    actual = active_map(initial_density, 0)
    actual = active_map(actual, 1)

    coherence = np.exp(-0.5 / 2.0)
    expected = 0.73 / 2 * np.array([
        [1.0, coherence],
        [coherence, 1.0],
    ])

    np.testing.assert_allclose(
        actual.full(), expected,
        atol=1e-12, rtol=0.0,
    )
    np.testing.assert_allclose(
        recorded_durations, [0.2, 0.3],
        atol=1e-12,
    )
    assert actual.tr() == pytest.approx(0.73, abs=1e-12)


def test_rb_decay_fit_and_uncertainty_scaling():
    """Verify exact RB parameter recovery and absolute-error covariance scaling."""
    lengths = np.array([0, 1, 2, 4, 8, 16, 32, 64])
    signal = 0.25 + 0.70 * 0.97**lengths
    errors = np.full(len(lengths), 0.002)

    result = fit_rb_decay(
        lengths, signal,
        qubits=2,
        offset=0.25,
        standard_errors=errors,
    )
    doubled_errors = fit_rb_decay(
        lengths, signal,
        qubits=2,
        offset=0.25,
        standard_errors=2 * errors,
    )

    assert result["amplitude"] == pytest.approx(0.70, abs=1e-7)
    assert result["decay_parameter"] == pytest.approx(0.97, abs=1e-7)
    assert result["error_per_clifford"] == pytest.approx(0.0225, abs=1e-7)

    assert result["error_per_clifford_standard_error"] == pytest.approx(
        0.75 * result["decay_parameter_standard_error"],
        rel=1e-12,
    )
    assert doubled_errors["error_per_clifford_standard_error"] == pytest.approx(
        2 * result["error_per_clifford_standard_error"],
        rel=1e-4,
    )


@pytest.mark.parametrize(
    "single_count, entangling_count",
    [
        (10.5, 1.8),
        (5.8727430555555555, 1.8458333333333334),
    ],
)
def test_entangling_error_extraction_round_trip(
    single_count,
    entangling_count,
):
    """Recover a known entangling error using the matching compiler's gate counts."""
    single_error = 0.0077
    entangling_error = 0.04

    epc = 1 - (
        (1 - single_error) ** single_count
        * (1 - entangling_error) ** entangling_count
    )

    actual = extract_entangling_gate_error(
        error_per_clifford=epc,
        single_qubit_error=single_error,
        single_qubit_gates_per_clifford=single_count,
        entangling_gates_per_clifford=entangling_count,
    )

    assert actual == pytest.approx(entangling_error, abs=1e-12)


def test_entangling_error_extraction_rejects_inconsistent_inputs():
    """Reject an EPC smaller than the assumed single-qubit error contribution."""
    with pytest.raises(ValueError, match="single-qubit contribution"):
        extract_entangling_gate_error(
            error_per_clifford=0.01,
            single_qubit_error=0.0077,
            single_qubit_gates_per_clifford=10.5,
            entangling_gates_per_clifford=1.8,
        )


def test_entangling_error_uncertainty_matches_finite_difference():
    """Verify analytical uncertainty derivatives against independent numerical differences."""
    epc = 0.149
    single_error = 0.0077
    single_count = 10.5
    entangling_count = 1.8
    step = 1e-6

    def extracted_error(clifford_error, native_single_error):
        """Evaluate the point estimate at perturbed calibration inputs."""
        return extract_entangling_gate_error(
            error_per_clifford=clifford_error,
            single_qubit_error=native_single_error,
            single_qubit_gates_per_clifford=single_count,
            entangling_gates_per_clifford=entangling_count,
        )

    derivative_epc = (
        extracted_error(epc + step, single_error)
        - extracted_error(epc - step, single_error)
    ) / (2 * step)

    derivative_single = (
        extracted_error(epc, single_error + step)
        - extracted_error(epc, single_error - step)
    ) / (2 * step)

    expected_uncertainty = np.sqrt(
        (derivative_epc * 0.027) ** 2
        + (derivative_single * 0.0012) ** 2
    )

    result = estimate_entangling_gate_error(
        error_per_clifford=epc,
        single_qubit_error=single_error,
        single_qubit_gates_per_clifford=single_count,
        entangling_gates_per_clifford=entangling_count,
        epc_standard_error=0.027,
        single_qubit_error_standard_error=0.0012,
    )

    assert result["standard_error"] == pytest.approx(
        expected_uncertainty,
        rel=1e-6,
    )
    assert result["entangling_fidelity"] == pytest.approx(
        1 - extracted_error(epc, single_error),
        abs=1e-12,
    )
