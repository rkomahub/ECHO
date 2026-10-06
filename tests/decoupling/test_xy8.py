import numpy as np

from echo_spin.control.rotations import two_qubit_rotation
from echo_spin.decoupling.xy8 import xy8_pulses


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
