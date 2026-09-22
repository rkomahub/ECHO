import numpy as np
import pytest

from echo_spin.frames import (
    rotate_to_local_frame, 
    rotation_matrix, 
    rotate_vector,
)


def test_zero_angle_returns_identity():
    """A zero-angle rotation should leave all vectors unchanged."""
    rotation = rotation_matrix("y", 0.0)

    assert np.allclose(
        rotation,
        np.eye(3),
    )


def test_y_rotation_by_pi_over_two():
    """A pi/2 rotation around y should map z onto x."""
    rotation = rotation_matrix("y", np.pi / 2)

    vector = np.array([0.0, 0.0, 1.0])

    rotated = rotate_vector(vector, rotation)

    assert np.allclose(
        rotated,
        [1.0, 0.0, 0.0],
    )


def test_rotation_preserves_vector_norm():
    """A proper rotation should preserve Euclidean vector length."""
    rotation = rotation_matrix("y", 0.37)

    vector = np.array([1.0, 2.0, 3.0])

    rotated = rotate_vector(vector, rotation)

    assert np.isclose(
        np.linalg.norm(rotated),
        np.linalg.norm(vector),
    )


def test_rotation_matrix_is_orthogonal():
    """A rotation matrix should satisfy R.T @ R = I."""
    rotation = rotation_matrix("y", 0.63)

    assert np.allclose(
        rotation.T @ rotation,
        np.eye(3),
    )


def test_invalid_axis_raises_error():
    """Only Cartesian rotation axes should be accepted."""
    with pytest.raises(ValueError):
        rotation_matrix("a", 0.5)


def test_invalid_vector_shape_raises_error():
    """Only three-dimensional vectors should be rotated."""
    rotation = rotation_matrix("y", 0.5)

    with pytest.raises(ValueError):
        rotate_vector([1.0, 2.0], rotation)


def test_rotate_to_local_frame():
    """The inverse rotation should express a vector in the local frame."""
    vector = [0.0, 0.0, 1.0]

    local = rotate_to_local_frame(
        vector=vector,
        axis="y",
        angle=np.pi / 2,
    )

    assert np.allclose(
        local,
        [-1.0, 0.0, 0.0],
    )