import numpy as np
import pytest
from qutip import basis, qeye, sigmax, tensor

from echo_spin.control.rotations import (
    selective_rotation,
    single_qubit_rotation,
    two_qubit_rotation,
)


def test_pi_x_rotation():
    """Test that a pi rotation about the x-axis is implemented correctly."""
    rotation = single_qubit_rotation(np.pi, "x")

    expected = -1j * sigmax()

    assert np.allclose(rotation.full(), expected.full())


def test_two_qubit_rotation_on_first_qubit():
    """Test that a two-qubit rotation on the first qubit is implemented correctly."""
    rotation = two_qubit_rotation(0, np.pi, "x")

    expected = tensor(-1j * sigmax(), qeye(2))

    assert np.allclose(rotation.full(), expected.full())


def test_selective_x_pi_rotation():
    """Test that a selective pi rotation about the x-axis is implemented correctly."""
    state_0 = basis(3, 0)
    state_1 = basis(3, 1)
    spectator = basis(3, 2)

    rotation = selective_rotation(
        state_0=state_0,
        state_1=state_1,
        angle=np.pi,
        axis="x",
    )

    rotated_0 = rotation * state_0
    rotated_1 = rotation * state_1
    rotated_spectator = rotation * spectator

    assert abs(state_1.overlap(rotated_0)) == pytest.approx(1.0)
    assert abs(state_0.overlap(rotated_1)) == pytest.approx(1.0)
    assert abs(spectator.overlap(rotated_spectator)) == pytest.approx(1.0)


def test_selective_y_pi_rotation():
    """Test that a selective pi rotation about the y-axis is implemented correctly."""
    state_0 = basis(3, 0)
    state_1 = basis(3, 1)
    spectator = basis(3, 2)

    rotation = selective_rotation(
        state_0=state_0,
        state_1=state_1,
        angle=np.pi,
        axis="y",
    )

    rotated_0 = rotation * state_0
    rotated_1 = rotation * state_1
    rotated_spectator = rotation * spectator

    assert abs(state_1.overlap(rotated_0)) == pytest.approx(1.0)
    assert abs(state_0.overlap(rotated_1)) == pytest.approx(1.0)
    assert abs(spectator.overlap(rotated_spectator)) == pytest.approx(1.0)


def test_selective_rotation_invalid_axis():
    """Test that an invalid axis raises a ValueError."""
    state_0 = basis(3, 0)
    state_1 = basis(3, 1)

    with pytest.raises(ValueError):
        selective_rotation(
            state_0=state_0,
            state_1=state_1,
            angle=np.pi,
            axis="z",
        )
