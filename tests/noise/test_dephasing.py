import numpy as np
import pytest
from qutip import basis, tensor

from echo_spin.noise.dephasing import (
    dephasing_channel,
    dephasing_hamiltonian,
    exponential_coherence,
    ornstein_uhlenbeck_noise,
    stochastic_dephasing_propagator,
    two_qubit_dephasing_channel,
)


def test_exponential_coherence():
    """Test that exponential coherence has the correct limiting behaviour."""
    assert np.isclose(
        exponential_coherence(
            time=0.0,
            coherence_time=465.0,
        ),
        1.0,
    )

    assert np.isclose(
        exponential_coherence(
            time=465.0,
            coherence_time=465.0,
        ),
        np.exp(-1),
    )


def test_exponential_coherence_rejects_invalid_parameters():
    """Test that invalid time and coherence-time parameters are rejected."""
    with pytest.raises(ValueError):
        exponential_coherence(
            time=-1.0,
            coherence_time=465.0,
        )

    with pytest.raises(ValueError):
        exponential_coherence(
            time=1.0,
            coherence_time=0.0,
        )


def test_joas_xy8_coherence_limit():
    """Reproduce the Joas XY8 coherence-limited gate error of approximately 1.4%."""
    t2_nv1 = 454.0
    t2_nv2 = 476.0

    average_t2 = (
        t2_nv1 + t2_nv2
    ) / 2

    gate_duration = 6.4

    fidelity = exponential_coherence(
        time=gate_duration,
        coherence_time=average_t2,
    )

    error = 1 - fidelity

    print("Joas XY8 coherence fidelity =", fidelity)
    print("Joas XY8 coherence error =", error)

    assert np.isclose(fidelity, 0.986, atol=1e-3)
    assert np.isclose(error, 0.014, atol=1e-3)


def test_dephasing_channel_preserves_populations():
    """Check that pure dephasing leaves basis-state populations unchanged."""
    zero = basis(2, 0)
    one = basis(2, 1)

    density_matrix = (
        0.3 * zero.proj()
        + 0.7 * one.proj()
    )

    result = dephasing_channel(
        density_matrix=density_matrix,
        time=10.0,
        coherence_time=20.0,
    )

    assert np.allclose(
        np.diag(result.full()),
        [0.3, 0.7],
    )


def test_dephasing_channel_decays_coherence():
    """Check exponential decay of the off-diagonal density-matrix elements."""
    plus = (
        basis(2, 0)
        + basis(2, 1)
    ).unit()

    density_matrix = plus.proj()

    time = 10.0
    coherence_time = 20.0

    result = dephasing_channel(
        density_matrix=density_matrix,
        time=time,
        coherence_time=coherence_time,
    )

    expected_coherence = (
        0.5
        * np.exp(-time / coherence_time)
    )

    assert np.isclose(
        result[0, 1],
        expected_coherence,
    )

    assert np.isclose(
        result[1, 0],
        expected_coherence,
    )


def test_dephasing_channel_preserves_trace():
    """Check that the dephasing channel preserves the density-matrix trace."""
    plus = (
        basis(2, 0)
        + basis(2, 1)
    ).unit()

    result = dephasing_channel(
        density_matrix=plus.proj(),
        time=10.0,
        coherence_time=20.0,
    )

    assert np.isclose(
        result.tr(),
        1.0,
    )


def test_two_qubit_dephasing_channel():
    """Check independent exponential dephasing of two qubits."""
    zero = basis(2, 0)
    one = basis(2, 1)

    zero_zero = tensor(zero, zero)
    one_zero = tensor(one, zero)
    zero_one = tensor(zero, one)

    time = 10.0
    coherence_times = (20.0, 40.0)

    state_1 = (
        zero_zero + one_zero
    ).unit()

    result_1 = two_qubit_dephasing_channel(
        density_matrix=state_1.proj(),
        time=time,
        coherence_times=coherence_times,
    )

    assert np.isclose(
        result_1[0, 2],
        0.5 * np.exp(-time / coherence_times[0]),
    )

    state_2 = (
        zero_zero + zero_one
    ).unit()

    result_2 = two_qubit_dephasing_channel(
        density_matrix=state_2.proj(),
        time=time,
        coherence_times=coherence_times,
    )

    assert np.isclose(
        result_2[0, 1],
        0.5 * np.exp(-time / coherence_times[1]),
    )


def test_ornstein_uhlenbeck_zero_noise():
    """Check that zero noise amplitude produces an identically zero trajectory."""
    times = np.linspace(
        0.0,
        10.0,
        101,
    )

    noise = ornstein_uhlenbeck_noise(
        times=times,
        sigma=0.0,
        correlation_time=2.0,
        rng=np.random.default_rng(42),
    )

    assert np.allclose(
        noise,
        0.0,
    )


def test_ornstein_uhlenbeck_rejects_invalid_parameters():
    """Check that invalid OU-process parameters are rejected."""
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        ornstein_uhlenbeck_noise(
            times=np.array([0.0, 1.0]),
            sigma=-1.0,
            correlation_time=2.0,
            rng=rng,
        )

    with pytest.raises(ValueError):
        ornstein_uhlenbeck_noise(
            times=np.array([0.0, 1.0]),
            sigma=1.0,
            correlation_time=0.0,
            rng=rng,
        )

    with pytest.raises(ValueError):
        ornstein_uhlenbeck_noise(
            times=np.array([0.0, 1.0, 0.5]),
            sigma=1.0,
            correlation_time=2.0,
            rng=rng,
        )


def test_ornstein_uhlenbeck_stationary_variance():
    """Check that a long OU trajectory approaches the stationary variance."""
    sigma = 2.0

    times = np.linspace(
        0.0,
        1000.0,
        100001,
    )

    noise = ornstein_uhlenbeck_noise(
        times=times,
        sigma=sigma,
        correlation_time=1.0,
        rng=np.random.default_rng(42),
    )

    assert np.isclose(
        np.var(noise),
        sigma**2,
        rtol=0.1,
    )


def test_dephasing_hamiltonian():
    """Check the longitudinal Hamiltonian generated by a noise realization."""
    noise_value = 2.0

    hamiltonian = dephasing_hamiltonian(
        noise_value=noise_value,
    )

    expected = np.array([
        [1.0, 0.0],
        [0.0, -1.0],
    ])

    assert np.allclose(
        hamiltonian.full(),
        expected,
    )


def test_dephasing_hamiltonian_preserves_populations():
    """Check that longitudinal noise changes phase but not populations."""
    plus = (
        basis(2, 0)
        + basis(2, 1)
    ).unit()

    hamiltonian = dephasing_hamiltonian(
        noise_value=2.0,
    )

    time = 1.0

    propagator = (
        -1j * hamiltonian * time
    ).expm()

    final_density = (
        propagator
        * plus.proj()
        * propagator.dag()
    )

    assert np.allclose(
        np.diag(final_density.full()),
        [0.5, 0.5],
    )

    assert not np.isclose(
        final_density[0, 1],
        0.5,
    )


def test_stochastic_dephasing_propagator_constant_noise():
    """Check phase accumulation for a constant longitudinal noise trajectory."""
    times = np.linspace(
        0.0,
        2.0,
        201,
    )

    noise_value = 3.0
    noise = np.full(
        len(times),
        noise_value,
    )

    propagator = stochastic_dephasing_propagator(
        times=times,
        noise=noise,
    )

    total_time = times[-1] - times[0]

    expected = (
        -1j
        * dephasing_hamiltonian(noise_value)
        * total_time
    ).expm()

    assert np.allclose(
        propagator.full(),
        expected.full(),
    )


def test_stochastic_dephasing_propagator_preserves_norm():
    """Check that stochastic longitudinal evolution remains unitary."""
    times = np.linspace(
        0.0,
        10.0,
        1001,
    )

    noise = ornstein_uhlenbeck_noise(
        times=times,
        sigma=2.0,
        correlation_time=1.0,
        rng=np.random.default_rng(42),
    )

    propagator = stochastic_dephasing_propagator(
        times=times,
        noise=noise,
    )

    identity = np.eye(2)

    assert np.allclose(
        (propagator.dag() * propagator).full(),
        identity,
    )

