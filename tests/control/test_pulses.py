import numpy as np
import pytest
from qutip import basis, jmat, qeye, sigmax, sigmaz

from echo_spin.control.pulses import (
    centered_sine_envelope,
    sine_envelope,
    sine_pi_pulse_amplitude,
)
from echo_spin.control.rotations import single_qubit_rotation
from echo_spin.core.basis import project_operator
from echo_spin.dynamics.propagators import time_dependent_propagator
from echo_spin.dynamics.sequences import finite_pulse_propagator


def test_sine_envelope_boundaries():
    assert np.isclose(sine_envelope(0.0, 1.0, 2.0), 0.0)
    assert np.isclose(sine_envelope(1.0, 1.0, 2.0), 0.0)


def test_sine_envelope_peak():
    assert np.isclose(sine_envelope(0.5, 1.0, 2.0), 2.0)


def test_sine_envelope_outside_pulse():
    assert sine_envelope(-0.1, 1.0, 2.0) == 0.0
    assert sine_envelope(1.1, 1.0, 2.0) == 0.0


def test_sine_envelope_rejects_invalid_duration():
    with pytest.raises(ValueError):
        sine_envelope(0.0, 0.0, 2.0)


def test_sine_pi_pulse_amplitude():
    duration = 2.0

    expected = np.pi**2 / (2 * duration)

    assert np.isclose(
        sine_pi_pulse_amplitude(duration),
        expected,
    )


def test_sine_pi_pulse_amplitude_rejects_invalid_duration():
    with pytest.raises(ValueError):
        sine_pi_pulse_amplitude(0.0)


def test_sine_pi_pulse_generates_pi_rotation():
    """A calibrated sine pulse generates an X pi rotation."""
    duration = 1.0
    steps = 1000

    peak_amplitude = sine_pi_pulse_amplitude(duration)

    x = sigmax()

    def hamiltonian(time):
        omega = sine_envelope(
            time=time,
            duration=duration,
            peak_amplitude=peak_amplitude,
        )

        return 0.5 * omega * x

    propagator = time_dependent_propagator(
        hamiltonian=hamiltonian,
        t0=0.0,
        t1=duration,
        steps=steps,
    )

    expected = single_qubit_rotation(
        angle=np.pi,
        axis="x",
    )

    assert np.allclose(
        propagator.full(),
        expected.full(),
        atol=1e-5,
    )


def test_finite_sine_pi_pulse_is_affected_by_detuning():
    """A finite-duration pi pulse is imperfect in the presence of detuning."""
    duration = 1.0
    steps = 1000
    detuning = 0.5

    peak_amplitude = sine_pi_pulse_amplitude(duration)

    x = sigmax()
    z = sigmaz()

    def hamiltonian(time):
        omega = sine_envelope(
            time=time,
            duration=duration,
            peak_amplitude=peak_amplitude,
        )

        return 0.5 * detuning * z + 0.5 * omega * x

    propagator = time_dependent_propagator(
        hamiltonian=hamiltonian,
        t0=0.0,
        t1=duration,
        steps=steps,
    )

    ideal = single_qubit_rotation(
        angle=np.pi,
        axis="x",
    )

    assert not np.allclose(
        propagator.full(),
        ideal.full(),
        atol=1e-5,
    )


def test_sine_pi_pulse_converges_to_ideal_rotation_when_shortened():
    """Finite-pulse error decreases as the pulse becomes short relative to detuning."""
    detuning = 0.5
    steps = 1000

    x = sigmax()
    z = sigmaz()

    ideal = single_qubit_rotation(
        angle=np.pi,
        axis="x",
    )

    errors = []

    for duration in [1.0, 0.5, 0.1, 0.05]:
        peak_amplitude = sine_pi_pulse_amplitude(duration)

        def hamiltonian(time):
            omega = sine_envelope(
                time=time,
                duration=duration,
                peak_amplitude=peak_amplitude,
            )

            return 0.5 * detuning * z + 0.5 * omega * x

        propagator = time_dependent_propagator(
            hamiltonian=hamiltonian,
            t0=0.0,
            t1=duration,
            steps=steps,
        )

        error = np.linalg.norm(
            propagator.full() - ideal.full()
        )

        errors.append(error)

    assert all(
        later < earlier
        for earlier, later in zip(errors, errors[1:])
    )

    assert errors[-1] < 0.01


def test_centered_sine_envelope():
    duration = 0.2
    center = 1.0
    amplitude = 3.0

    assert centered_sine_envelope(
        0.9, center, duration, amplitude
    ) == pytest.approx(0.0)

    assert centered_sine_envelope(
        1.0, center, duration, amplitude
    ) == pytest.approx(amplitude)

    assert centered_sine_envelope(
        1.1, center, duration, amplitude
    ) == pytest.approx(0.0)

    assert centered_sine_envelope(
        0.8, center, duration, amplitude
    ) == pytest.approx(0.0)


@pytest.mark.parametrize("duration", [0.05, 0.2])
@pytest.mark.parametrize("start_time", [0.0, 0.7])
def test_finite_sine_pulse_calibration_converges(duration, start_time):
    """Check calibrated sine pulses at different durations and absolute times."""
    center = start_time + duration / 2
    amplitude = sine_pi_pulse_amplitude(duration)
    ideal = single_qubit_rotation(np.pi, "x")

    def envelope(time):
        return centered_sine_envelope(
            time=time,
            center=center,
            duration=duration,
            peak_amplitude=amplitude,
        )

    errors = []

    for steps in [100, 200]:
        actual = finite_pulse_propagator(
            free_hamiltonian=0 * sigmaz(),
            control_operator=0.5 * sigmax(),
            duration=duration,
            envelope=envelope,
            steps=steps,
            start_time=start_time,
        )

        errors.append(
            np.linalg.norm(actual.full() - ideal.full())
        )

    assert errors[1] < 1e-4

    # For this commuting sine pulse with zero endpoint amplitudes,
    # the integrated pulse-area error decreases quadratically.
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.02)


@pytest.mark.parametrize("detuning", [0.0, 0.5, 2.0])
def test_detuned_rectangular_pulse_matches_exact_solution(detuning):
    """Validate finite evolution with both drive and detuning active."""
    duration = 0.4
    omega = np.pi / duration
    effective_frequency = np.sqrt(omega**2 + detuning**2)

    actual = finite_pulse_propagator(
        free_hamiltonian=0.5 * detuning * sigmaz(),
        control_operator=0.5 * sigmax(),
        duration=duration,
        envelope=lambda time: omega,
        steps=10,
        start_time=0.7,
    )

    expected = (
        np.cos(effective_frequency * duration / 2) * qeye(2)
        - 1j
        * np.sin(effective_frequency * duration / 2)
        * (omega * sigmax() + detuning * sigmaz())
        / effective_frequency
    )

    assert np.allclose(
        actual.full(),
        expected.full(),
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize("angle", [0.0, 0.7, np.pi / 2, np.pi])
def test_physical_spin_one_control_exposes_logical_leakage(angle):
    """Validate spin-1 populations and leakage from the selected logical space."""
    sx = jmat(1, "x")
    sz = jmat(1, "z")

    # QuTiP spin-1 ordering: |+1>, |0>, |-1>.
    spectator = basis(3, 0)
    logical_zero = basis(3, 1)
    logical_one = basis(3, 2)

    duration = 1.0
    propagator = finite_pulse_propagator(
        free_hamiltonian=0 * sz,
        control_operator=sx,
        duration=duration,
        envelope=lambda time: angle / duration,
        steps=20,
    )

    final_state = propagator * logical_zero

    assert propagator.isunitary
    assert abs(logical_zero.overlap(final_state))**2 == pytest.approx(
        np.cos(angle)**2,
        abs=1e-12,
    )
    assert abs(logical_one.overlap(final_state))**2 == pytest.approx(
        np.sin(angle)**2 / 2,
        abs=1e-12,
    )
    assert abs(spectator.overlap(final_state))**2 == pytest.approx(
        np.sin(angle)**2 / 2,
        abs=1e-12,
    )

    projected = project_operator(
        propagator,
        [logical_zero, logical_one],
    )
    expected = np.array([
        [np.cos(angle), -1j * np.sin(angle) / np.sqrt(2)],
        [-1j * np.sin(angle) / np.sqrt(2), (1 + np.cos(angle)) / 2],
    ])

    assert np.allclose(
        projected.full(),
        expected,
        atol=1e-12,
        rtol=0.0,
    )

    retained_probability = np.linalg.norm(
        projected.full()[:, 0]
    )**2

    assert retained_probability == pytest.approx(
        1 - np.sin(angle)**2 / 2,
        abs=1e-12,
    )
