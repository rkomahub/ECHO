import numpy as np
import pytest
from qutip import qeye, sigmaz

from echo_spin.control.rotations import single_qubit_rotation
from echo_spin.decoupling.modulation import modulation_function


def test_no_pulses_gives_constant_modulation():
    actual = modulation_function([0.0, 0.5, 1.0], [])
    assert np.array_equal(actual, [1, 1, 1])


def test_hahn_modulation_and_pulse_boundary():
    actual = modulation_function(
        [0.0, 0.49, 0.5, 0.51, 1.0],
        [0.5],
    )
    assert np.array_equal(actual, [1, 1, -1, -1, -1])


def test_xy8_modulation_matches_operator_toggling():
    """Check scalar signs against explicit cumulative X/Y rotations."""
    pulse_times = (np.arange(8) + 0.5) / 8
    sample_times = np.concatenate(([0.0], pulse_times, [1.0]))
    actual = modulation_function(sample_times, pulse_times)

    cumulative = qeye(2)
    assert actual[0] == 1

    for index, axis in enumerate(
        ["x", "y", "x", "y", "y", "x", "y", "x"],
        start=1,
    ):
        pulse = single_qubit_rotation(np.pi, axis)
        cumulative = pulse * cumulative
        toggled = cumulative.dag() * sigmaz() * cumulative

        assert np.allclose(
            toggled.full(),
            (actual[index] * sigmaz()).full(),
        )

    assert actual[-1] == 1


def test_hahn_modulation_cancels_static_phase():
    """Equal intervals with opposite signs cancel static detuning."""
    boundaries = np.array([0.0, 0.5, 1.0])
    midpoints = (boundaries[:-1] + boundaries[1:]) / 2
    signs = modulation_function(midpoints, [0.5])

    integrated_modulation = signs @ np.diff(boundaries)
    assert integrated_modulation == pytest.approx(0.0)


@pytest.mark.parametrize(
    "times, pulse_times",
    [
        ([-0.1], []),
        ([np.nan], []),
        ([[0.0, 1.0]], []),
        ([0.0], [-0.1]),
        ([0.0], [np.inf]),
        ([0.0], [0.5, 0.25]),
        ([0.0], [0.5, 0.5]),
        ([0.0], [[0.5]]),
        ([0.0], [0.5j]),
    ],
)
def test_modulation_rejects_invalid_times(times, pulse_times):
    with pytest.raises(ValueError):
        modulation_function(times, pulse_times)
