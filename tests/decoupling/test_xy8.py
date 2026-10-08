import numpy as np
import pytest
from qutip import basis, qeye, tensor

from echo_spin.control.rotations import two_qubit_rotation
from echo_spin.decoupling.modulation import modulation_function
from echo_spin.decoupling.xy8 import xy8_pulses
from echo_spin.noise.dephasing import (
    ornstein_uhlenbeck_noise,
    stochastic_dephasing_propagator,
)


def test_xy8_contains_eight_pulses():
    pulses = xy8_pulses(qubit=0)

    assert len(pulses) == 8


def test_xy8_phase_pattern():
    expected = [
        two_qubit_rotation(0, np.pi, "x"),
        two_qubit_rotation(0, np.pi, "y"),
        two_qubit_rotation(0, np.pi, "x"),
        two_qubit_rotation(0, np.pi, "y"),
        two_qubit_rotation(0, np.pi, "y"),
        two_qubit_rotation(0, np.pi, "x"),
        two_qubit_rotation(0, np.pi, "y"),
        two_qubit_rotation(0, np.pi, "x"),
    ]

    actual = xy8_pulses(qubit=0)

    for actual_pulse, expected_pulse in zip(actual, expected):
        assert np.allclose(
            actual_pulse.full(),
            expected_pulse.full(),
        )


def test_ideal_xy8_under_ou_noise():
    """Validate ideal XY8 coherence against stationary OU statistics."""
    sigma = 0.9
    correlation_time = 0.8
    total_time = 2.0
    realizations = 1000

    times = np.linspace(0.0, total_time, 81)
    pulses = xy8_pulses(qubit=0)

    # Symmetric XY8: half-spacing before the first and after the last pulse.
    pulse_indices = np.arange(5, 80, 10)
    boundaries = np.concatenate(([0], pulse_indices, [80]))

    plus = (basis(2, 0) + basis(2, 1)).unit()
    initial_density = tensor(plus, basis(2, 0)).proj()
    average_density = 0 * initial_density

    cumulative = qeye([2, 2])
    for pulse in pulses:
        cumulative = pulse * cumulative

    rng = np.random.default_rng(42)

    for _ in range(realizations):
        noise = ornstein_uhlenbeck_noise(
            times=times,
            sigma=sigma,
            correlation_time=correlation_time,
            rng=rng,
        )

        propagator = qeye([2, 2])

        for interval, (start, stop) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        ):
            free = stochastic_dephasing_propagator(
                times=times[start:stop + 1],
                noise=noise[start:stop + 1],
            )
            propagator = tensor(free, qeye(2)) * propagator

            if interval < len(pulses):
                propagator = pulses[interval] * propagator

        echo = cumulative.dag() * propagator
        average_density += echo * initial_density * echo.dag()

    average_density /= realizations
    coherence = complex(
        average_density[0, 2] / initial_density[0, 2]
    )

    # Exact Gaussian prediction for the left-endpoint sampled noise.
    signs = (-1.0) ** np.arange(len(boundaries) - 1)
    interval_midpoints = (times[:-1] + times[1:]) / 2
    modulation = modulation_function(
        times=interval_midpoints,
        pulse_times=times[pulse_indices],
    )
    weights = modulation * np.diff(times)
    sample_times = times[:-1]

    covariance = sigma**2 * np.exp(
        -np.abs(sample_times[:, None] - sample_times[None, :])
        / correlation_time
    )
    phase_variance = float(weights @ covariance @ weights)
    sampled_coherence = np.exp(-phase_variance / 2)

    # Continuous-time double integral of the OU correlation function.
    edges = times[boundaries]
    starts = edges[:-1]
    stops = edges[1:]

    def primitive(value):
        return (
            correlation_time * np.abs(value)
            + correlation_time**2
            * np.exp(-np.abs(value) / correlation_time)
        )

    interval_covariance = sigma**2 * (
        primitive(stops[:, None] - starts[None, :])
        - primitive(starts[:, None] - starts[None, :])
        - primitive(stops[:, None] - stops[None, :])
        + primitive(starts[:, None] - stops[None, :])
    )
    continuous_variance = float(
        signs @ interval_covariance @ signs
    )
    analytic_coherence = np.exp(-continuous_variance / 2)

    assert sampled_coherence == pytest.approx(
        analytic_coherence,
        abs=2e-3,
    )

    real_variance = (
        (1 + np.exp(-2 * phase_variance)) / 2
        - sampled_coherence**2
    )
    imag_variance = (1 - np.exp(-2 * phase_variance)) / 2

    assert abs(coherence.real - sampled_coherence) < (
        5 * np.sqrt(real_variance / realizations)
    )
    assert abs(coherence.imag) < (
        5 * np.sqrt(imag_variance / realizations)
    )

    ratio = total_time / correlation_time
    hahn_coherence = np.exp(
        -sigma**2
        * correlation_time**2
        * (
            ratio - 3
            + 4 * np.exp(-ratio / 2)
            - np.exp(-ratio)
        )
    )
    assert analytic_coherence > hahn_coherence
