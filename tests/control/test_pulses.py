import numpy as np
import pytest

from qutip import sigmax, sigmaz

from echo_spin.control.pulses import centered_sine_envelope, sine_envelope, sine_pi_pulse_amplitude
from echo_spin.control.rotations import single_qubit_rotation
from echo_spin.dynamics.propagators import time_dependent_propagator

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