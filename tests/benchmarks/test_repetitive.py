import numpy as np
import pytest
from qutip import basis, qeye, sigmax, sigmaz, tensor

from echo_spin.benchmarks.repetitive import (
    fit_repetitive_decay,
    logical_repetitive_return_signal,
    repetitive_benchmark_signal,
)
from echo_spin.gates.gates import sqrt_zz_gate


@pytest.mark.parametrize("survival_per_gate", [1.0, 0.97])
def test_repetitive_benchmark_ideal_recovery_and_loss(survival_per_gate):
    """Verify ideal recovery and retained loss for unsorted and repeated counts."""
    plus = (basis(2, 0) + basis(2, 1)).unit()
    initial_density = tensor(plus, plus).proj()
    gate = sqrt_zz_gate()

    def gate_map(density_matrix):
        """Apply the ideal entangler with a specified uniform survival factor."""
        return (
            survival_per_gate
            * gate * density_matrix * gate.dag()
        )

    def readout(density_matrix, count):
        """Undo the ideal repeated gate and measure initial-state return."""
        recovery = (gate.dag()) ** count
        recovered = (
            recovery * density_matrix * recovery.dag()
        )
        return (initial_density * recovered).tr()

    counts = np.array([4, 0, 2, 4, 1])

    actual = repetitive_benchmark_signal(
        initial_density=initial_density,
        gate_map=gate_map,
        repetitions=counts,
        readout=readout,
    )

    np.testing.assert_allclose(
        actual,
        survival_per_gate ** counts,
        atol=1e-12,
        rtol=0.0,
    )


def test_repetitive_benchmark_retains_environment_memory():
    """Verify that repeated joint evolution retains a persistent nuclear-like environment."""
    plus = (basis(2, 0) + basis(2, 1)).unit()
    electron_density = plus.proj()
    initial_density = tensor(
        electron_density,
        qeye(2) / 2,
    )

    phase_per_gate = 0.4
    gate = (
        -1j * phase_per_gate / 2
        * tensor(sigmaz(), sigmaz())
    ).expm()

    def gate_map(density_matrix):
        """Evolve the same joint electron-environment state through another gate."""
        return gate * density_matrix * gate.dag()

    def readout(density_matrix, count):
        """Measure electron return while tracing the persistent environment."""
        return (
            electron_density * density_matrix.ptrace(0)
        ).tr()

    counts = np.arange(6)

    actual = repetitive_benchmark_signal(
        initial_density=initial_density,
        gate_map=gate_map,
        repetitions=counts,
        readout=readout,
    )

    expected = (
        1 + np.cos(counts * phase_per_gate)
    ) / 2

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-12,
        rtol=0.0,
    )

    reset_environment_prediction = (
        1 + np.cos(phase_per_gate) ** counts
    ) / 2

    assert abs(
        actual[-1] - reset_environment_prediction[-1]
    ) > 1e-2


@pytest.mark.parametrize("amplitude", [0.73, -0.15])
def test_repetitive_decay_fit_recovers_known_parameters(amplitude):
    """Recover an exact exponential decay with positive or negative readout contrast."""
    counts = np.arange(0, 81, 4)
    offset = 0.2
    decay_parameter = 0.96

    signal = (
        offset + amplitude * decay_parameter**counts
    )

    result = fit_repetitive_decay(
        repetitions=counts,
        signal=signal,
    )

    assert result["offset"] == pytest.approx(offset, abs=1e-7)
    assert result["amplitude"] == pytest.approx(amplitude, abs=1e-7)
    assert result["decay_parameter"] == pytest.approx(
        decay_parameter, abs=1e-7
    )
    assert result["pseudo_error_per_gate"] == pytest.approx(
        0.04, abs=1e-7
    )
    assert result["residual_rms"] < 1e-7


def test_logical_repetitive_return_preserves_three_level_nuclear_memory():
    """Verify logical return against an exact persistent spin-1 nuclear reference."""
    plus = (basis(2, 0) + basis(2, 1)).unit()
    initial_density = tensor(plus, basis(2, 0)).proj()

    logical_basis = [
        tensor(basis(3, 1), basis(3, 1)),
        tensor(basis(3, 1), basis(3, 0)),
        tensor(basis(3, 0), basis(3, 1)),
        tensor(basis(3, 0), basis(3, 0)),
    ]

    # Diagonal operator with eigenvalues (+1, 0, -1).
    sz = basis(3, 0).proj() - basis(3, 2).proj()
    phase_per_gate = 0.4

    propagator = (
        -1j * phase_per_gate
        * tensor(sz, sz, qeye(3), qeye(3))
    ).expm()

    counts = np.arange(6)

    actual = logical_repetitive_return_signal(
        initial_density=initial_density,
        propagator=propagator,
        logical_basis=logical_basis,
        nuclear_states=[qeye(3) / 3, qeye(3) / 3],
        target=qeye([2, 2]),
        repetitions=counts,
    )

    coherence = (
        1 + 2 * np.cos(counts * phase_per_gate)
    ) / 3

    np.testing.assert_allclose(
        actual,
        (1 + coherence) / 2,
        atol=1e-12,
        rtol=0.0,
    )


def test_repetitive_local_recovery_for_even_gate_counts():
    """Verify local recovery of ideal even entangler powers without resetting nuclei."""
    plus = (basis(2, 0) + basis(2, 1)).unit()
    initial_density = tensor(plus, plus).proj()
    target = sqrt_zz_gate()

    logical_basis = [
        tensor(basis(2, i), basis(2, j))
        for i in range(2)
        for j in range(2)
    ]

    joint_gate = tensor(
        target,
        qeye(2),
        qeye(2),
    ).permute([0, 2, 1, 3])

    def local_readout(density_matrix, count):
        """Measure recovered logical Z through its equivalent signed X observable."""
        if count % 2:
            raise ValueError("This local recovery requires even gate counts.")

        return (
            (-1) ** (count // 2)
            * (tensor(sigmax(), qeye(2)) * density_matrix).tr()
        )

    signal = logical_repetitive_return_signal(
        initial_density=initial_density,
        propagator=joint_gate,
        logical_basis=logical_basis,
        nuclear_states=[qeye(2) / 2, qeye(2) / 2],
        target=target,
        repetitions=np.arange(0, 17, 2),
        logical_readout=local_readout,
    )

    np.testing.assert_allclose(
        signal,
        1.0,
        atol=1e-12,
        rtol=0.0,
    )
