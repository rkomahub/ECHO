import numpy as np
import pytest
from qutip import basis, tensor

from echo_spin.control.rotations import single_qubit_rotation
from echo_spin.gates.gates import average_gate_fidelity, sqrt_zz_gate
from echo_spin.noise.dephasing import (
    dephasing_channel,
    dephasing_hamiltonian,
    ensemble_dephasing_map,
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


def test_joas_xy8_coherence_limited_pseudo_error():
    """Reproduce the Joas coherence-limited pseudo-error of about 1.4%."""
    average_t2 = (454.0 + 476.0) / 2
    gate_duration = 6.4

    coherence = exponential_coherence(
        time=gate_duration,
        coherence_time=average_t2,
    )
    pseudo_error = 1 - coherence

    assert np.isclose(coherence, 0.986, atol=1e-3)
    assert np.isclose(pseudo_error, 0.014, atol=1e-3)


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


def test_ensemble_dephasing_map_preserves_populations():
    """Check that ensemble-averaged longitudinal noise preserves populations."""
    plus = (
        basis(2, 0)
        + basis(2, 1)
    ).unit()

    times = np.linspace(
        0.0,
        10.0,
        1001,
    )

    result = ensemble_dephasing_map(
        density_matrix=plus.proj(),
        times=times,
        sigma=1.0,
        correlation_time=1.0,
        realizations=200,
        rng=np.random.default_rng(42),
    )

    assert np.allclose(
        np.diag(result.full()),
        [0.5, 0.5],
    )


def test_ensemble_dephasing_map_reduces_coherence():
    """Check that averaging OU realizations suppresses qubit coherence."""
    plus = (
        basis(2, 0)
        + basis(2, 1)
    ).unit()

    times = np.linspace(
        0.0,
        10.0,
        1001,
    )

    result = ensemble_dephasing_map(
        density_matrix=plus.proj(),
        times=times,
        sigma=1.0,
        correlation_time=1.0,
        realizations=500,
        rng=np.random.default_rng(42),
    )

    assert abs(result[0, 1]) < 0.5
    assert np.isclose(
        result.tr(),
        1.0,
    )


@pytest.mark.parametrize(
    "time, coherence_times",
    [
        (0.0, (20.0, 40.0)),
        (10.0, (20.0, 40.0)),
        (6.4, (454.0, 476.0)),
        (1000.0, (1.0, 2.0)),
    ],
)
def test_two_qubit_dephasing_average_gate_fidelity(
    time,
    coherence_times,
):
    """Check noisy sqrt(ZZ) fidelity against the analytic channel result."""
    target = sqrt_zz_gate()
    basis_states = [
        tensor(basis(2, i), basis(2, j))
        for i in range(2)
        for j in range(2)
    ]

    def dynamical_map(density_matrix):
        ideal_output = target * density_matrix * target.dag()

        return two_qubit_dephasing_channel(
            density_matrix=ideal_output,
            time=time,
            coherence_times=coherence_times,
        )

    actual = average_gate_fidelity(
        dynamical_map=dynamical_map,
        target=target,
        basis_states=basis_states,
    )

    p1 = np.exp(-time / coherence_times[0])
    p2 = np.exp(-time / coherence_times[1])
    expected = ((1 + p1) * (1 + p2) + 1) / 5

    assert actual == pytest.approx(expected, abs=1e-12)


def test_ensemble_ou_dephasing_matches_analytic_coherence():
    """Validate stationary OU free induction, including sampling errors."""
    sigma = 0.9
    correlation_time = 0.8
    total_time = 2.0
    realizations = 1000
    times = np.linspace(0.0, total_time, 41)

    plus = (basis(2, 0) + basis(2, 1)).unit()

    result = ensemble_dephasing_map(
        density_matrix=plus.proj(),
        times=times,
        sigma=sigma,
        correlation_time=correlation_time,
        realizations=realizations,
        rng=np.random.default_rng(42),
    )

    # The propagator uses noise at each interval's left endpoint.
    sample_times = times[:-1]
    durations = np.diff(times)
    covariance = sigma**2 * np.exp(
        -np.abs(sample_times[:, None] - sample_times[None, :])
        / correlation_time
    )
    sampled_phase_variance = float(
        durations @ covariance @ durations
    )
    sampled_coherence = np.exp(-sampled_phase_variance / 2)

    # Exact continuous-time OU free-induction coherence.
    ratio = total_time / correlation_time
    analytic_coherence = np.exp(
        -sigma**2
        * correlation_time**2
        * (ratio + np.expm1(-ratio))
    )

    # Check that the chosen time grid resolves the continuous prediction.
    assert sampled_coherence == pytest.approx(
        analytic_coherence,
        rel=1e-3,
    )

    # Gaussian phase statistics give the Monte Carlo standard errors.
    real_variance = (
        (1 + np.exp(-2 * sampled_phase_variance)) / 2
        - sampled_coherence**2
    )
    imag_variance = (
        1 - np.exp(-2 * sampled_phase_variance)
    ) / 2

    real_standard_error = np.sqrt(real_variance / realizations)
    imag_standard_error = np.sqrt(imag_variance / realizations)

    normalized_coherence = complex(result[0, 1] / plus.proj()[0, 1])

    assert abs(
        normalized_coherence.real - sampled_coherence
    ) < 5 * real_standard_error

    assert abs(
        normalized_coherence.imag
    ) < 5 * imag_standard_error


def test_ou_hahn_echo_matches_analytic_coherence():
    """Validate Hahn coherence using one continuous OU trajectory per run."""
    sigma = 0.9
    correlation_time = 0.8
    total_time = 2.0
    realizations = 1000
    times = np.linspace(0.0, total_time, 41)
    midpoint = (len(times) - 1) // 2

    plus = (basis(2, 0) + basis(2, 1)).unit()
    initial_density = plus.proj()
    pi_x = single_qubit_rotation(angle=np.pi, axis="x")
    rng = np.random.default_rng(42)
    average_density = 0 * initial_density

    for _ in range(realizations):
        noise = ornstein_uhlenbeck_noise(
            times=times,
            sigma=sigma,
            correlation_time=correlation_time,
            rng=rng,
        )

        first = stochastic_dephasing_propagator(
            times=times[:midpoint + 1],
            noise=noise[:midpoint + 1],
        )
        second = stochastic_dephasing_propagator(
            times=times[midpoint:],
            noise=noise[midpoint:],
        )

        # Remove the final control transformation to obtain the echo frame.
        echo = pi_x.dag() * second * pi_x * first
        average_density += echo * initial_density * echo.dag()

    average_density /= realizations

    sample_times = times[:-1]
    weights = np.diff(times)
    weights[midpoint:] *= -1

    covariance = sigma**2 * np.exp(
        -np.abs(sample_times[:, None] - sample_times[None, :])
        / correlation_time
    )
    phase_variance = float(weights @ covariance @ weights)
    sampled_coherence = np.exp(-phase_variance / 2)

    ratio = total_time / correlation_time
    analytic_coherence = np.exp(
        -sigma**2
        * correlation_time**2
        * (
            ratio - 3
            + 4 * np.exp(-ratio / 2)
            - np.exp(-ratio)
        )
    )

    assert sampled_coherence == pytest.approx(
        analytic_coherence,
        rel=2e-3,
    )

    real_variance = (
        (1 + np.exp(-2 * phase_variance)) / 2
        - sampled_coherence**2
    )
    imag_variance = (1 - np.exp(-2 * phase_variance)) / 2

    coherence = complex(
        average_density[0, 1] / initial_density[0, 1]
    )

    assert abs(coherence.real - sampled_coherence) < (
        5 * np.sqrt(real_variance / realizations)
    )
    assert abs(coherence.imag) < (
        5 * np.sqrt(imag_variance / realizations)
    )

    fid_coherence = np.exp(
        -sigma**2
        * correlation_time**2
        * (ratio - 1 + np.exp(-ratio))
    )
    assert analytic_coherence > fid_coherence
