import numpy as np
import pytest

from echo_spin.nv.frames import (
    crystallographic_nv_axes,
    local_frame_from_axis,
    rotate_to_local_frame,
    rotate_vector,
    rotation_matrix,
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


def test_crystallographic_axes_have_tetrahedral_geometry():
    """Four unit representatives have pairwise dot products -1/3."""
    axes = crystallographic_nv_axes()

    assert axes.shape == (4, 3)

    expected = np.full((4, 4), -1 / 3)
    np.fill_diagonal(expected, 1.0)

    assert np.allclose(
        axes @ axes.T,
        expected,
        atol=1e-12,
        rtol=0.0,
    )


def test_crystal_field_projections():
    """Check characteristic [001] and [111] field projections."""
    axes = crystallographic_nv_axes()

    field_001 = np.array([0.0, 0.0, 1.0])
    assert np.allclose(
        np.abs(axes @ field_001),
        np.full(4, 1 / np.sqrt(3)),
    )

    field_111 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    assert np.allclose(
        axes @ field_111,
        [1.0, -1 / 3, -1 / 3, -1 / 3],
    )


@pytest.mark.parametrize("index", range(4))
def test_local_frames_for_crystallographic_axes(index):
    axis = crystallographic_nv_axes()[index]
    frame = local_frame_from_axis(axis, [0.0, 0.0, 1.0])

    assert np.allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame) == pytest.approx(1.0)
    assert np.allclose(frame[:, 2], axis)

    # A vector along the NV axis has only a local z component.
    assert np.allclose(
        rotate_vector(axis, frame.T),
        [0.0, 0.0, 1.0],
        atol=1e-12,
    )

    vector = np.array([0.2, -0.3, 0.7])
    local = rotate_vector(vector, frame.T)
    recovered = rotate_vector(local, frame)

    assert np.allclose(recovered, vector, atol=1e-12)


def test_transverse_reference_fixes_local_xy_axes():
    frame = local_frame_from_axis(
        nv_axis=[0.0, 0.0, 2.0],
        transverse_reference=[0.0, 3.0, 4.0],
    )

    # Local x follows global y; local y follows minus global x.
    expected = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    assert np.allclose(frame, expected, atol=1e-12)
    assert np.allclose(
        rotate_vector([1.0, 0.0, 0.0], frame.T),
        [0.0, -1.0, 0.0],
    )


@pytest.mark.parametrize(
    "axis, reference",
    [
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, 1.0], [0.0, 0.0, 2.0]),
        ([0.0, 0.0, 1.0], [0.0, 0.0, -2.0]),
        ([0.0, np.nan, 1.0], [1.0, 0.0, 0.0]),
        ([0.0, 1.0], [1.0, 0.0, 0.0]),
        ([0.0, 0.0, 1.0], [1j, 0.0, 0.0]),
    ],
)
def test_local_frame_rejects_invalid_vectors(axis, reference):
    with pytest.raises(ValueError):
        local_frame_from_axis(axis, reference)
