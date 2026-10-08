import numpy as np
import pytest
from qutip import basis, qeye, sigmax, sigmay, sigmaz, tensor

from echo_spin.control.rotations import (
    selective_rotation,
    single_qubit_rotation,
    two_qubit_rotation,
)


@pytest.mark.parametrize(
    "axis, pauli",
    [
        ("x", sigmax()),
        ("y", sigmay()),
        ("z", sigmaz()),
    ],
)
def test_cartesian_rotation_matches_closed_form(axis, pauli):
    """Check Cartesian rotations at a non-pi angle."""
    angle = 0.7
    expected = (
        np.cos(angle / 2) * qeye(2)
        - 1j * np.sin(angle / 2) * pauli
    )

    actual = single_qubit_rotation(angle, axis)

    assert np.allclose(actual.full(), expected.full())


def test_arbitrary_axis_rotation_matches_closed_form():
    """Check an oblique rotation and automatic axis normalization."""
    angle = 0.7
    generator = (sigmax() + 2 * sigmay() + 3 * sigmaz()) / np.sqrt(14)
    expected = (
        np.cos(angle / 2) * qeye(2)
        - 1j * np.sin(angle / 2) * generator
    )

    actual = single_qubit_rotation(angle, [1.0, 2.0, 3.0])
    scaled = single_qubit_rotation(angle, [5.0, 10.0, 15.0])

    assert np.allclose(actual.full(), expected.full())
    assert np.allclose(scaled.full(), expected.full())


def test_arbitrary_axis_rotation_inverse():
    """Opposite angles about the same axis undo each other."""
    axis = [1.0, -2.0, 3.0]
    forward = single_qubit_rotation(0.7, axis)
    backward = single_qubit_rotation(-0.7, axis)

    assert np.allclose(
        (backward * forward).full(),
        qeye(2).full(),
    )


def test_two_qubit_arbitrary_axis_rotation():
    """Check that vector axes pass through the embedding wrapper."""
    axis = [1.0, 2.0, 3.0]
    local = single_qubit_rotation(0.7, axis)
    actual = two_qubit_rotation(1, 0.7, axis)

    assert np.allclose(
        actual.full(),
        tensor(qeye(2), local).full(),
    )


@pytest.mark.parametrize(
    "axis",
    [
        "invalid",
        [0.0, 0.0, 0.0],
        [1.0, 2.0],
        [np.nan, 0.0, 1.0],
        [np.inf, 0.0, 1.0],
        [1j, 0.0, 1.0],
    ],
)
def test_single_qubit_rotation_rejects_invalid_axis(axis):
    with pytest.raises(ValueError):
        single_qubit_rotation(0.7, axis)


@pytest.mark.parametrize("angle", [np.nan, np.inf])
def test_single_qubit_rotation_rejects_nonfinite_angle(angle):
    with pytest.raises(ValueError):
        single_qubit_rotation(angle, "x")

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
            axis="invalid",
        )


@pytest.mark.parametrize(
    "axis",
    ["x", "y", "z", [1.0, -2.0, 3.0]],
)
def test_selective_rotation_in_nontrivial_basis(axis):
    """Check logical rotation and exact spectator preservation."""
    state_0 = (basis(3, 0) + 1j * basis(3, 2)).unit()
    state_1 = basis(3, 1)
    spectator = (basis(3, 0) - 1j * basis(3, 2)).unit()
    states = [state_0, state_1]

    actual = selective_rotation(
        state_0=state_0,
        state_1=state_1,
        angle=0.7,
        axis=axis,
    )
    logical = single_qubit_rotation(0.7, axis)

    for j, state in enumerate(states):
        expected = (
            logical[0, j] * state_0
            + logical[1, j] * state_1
        )
        assert np.allclose(
            (actual * state).full(),
            expected.full(),
        )

    assert np.allclose(
        (actual * spectator).full(),
        spectator.full(),
    )
    assert np.allclose(
        (actual.dag() * actual).full(),
        qeye(3).full(),
    )


@pytest.mark.parametrize(
    "state_0, state_1",
    [
        (2 * basis(3, 0), basis(3, 1)),
        (basis(3, 0), (basis(3, 0) + basis(3, 1)).unit()),
        (basis(3, 0), basis(2, 1)),
        (basis(3, 0).proj(), basis(3, 1)),
    ],
)
def test_selective_rotation_rejects_invalid_states(state_0, state_1):
    """Reject inputs that do not define an orthonormal two-level basis."""
    with pytest.raises(ValueError):
        selective_rotation(state_0, state_1, 0.7, "x")
