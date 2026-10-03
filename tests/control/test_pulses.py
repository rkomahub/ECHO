import numpy as np
import pytest
from echo_spin.control.pulses import sine_envelope


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