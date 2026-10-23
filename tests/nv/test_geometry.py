import numpy as np
import pytest

from echo_spin.nv.frames import crystallographic_nv_axes, rotation_matrix
from echo_spin.nv.geometry import (
    lattice_positions,
    register_local_frames,
    validate_positions,
)


@pytest.mark.parametrize(
    "shape, expected_indices",
    [
        ((3, 1, 1), [[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
        ((2, 2, 1), [[0, 0, 0], [0, 1, 0],
                     [1, 0, 0], [1, 1, 0]]),
        ((1, 2, 2), [[0, 0, 0], [0, 0, 1],
                     [0, 1, 0], [0, 1, 1]]),
    ],
)
def test_lattice_layout_and_site_order(shape, expected_indices):
    """Generate chains, planes and spatial layouts in deterministic order."""
    spacing = 15e-9
    origin = np.array([2e-9, -3e-9, 5e-9])
    actual = lattice_positions(shape, spacing * np.eye(3), origin)
    expected = origin + spacing * np.array(expected_indices)

    np.testing.assert_allclose(actual, expected, atol=1e-22, rtol=0.0)


def test_lattice_vectors_are_rows():
    """Use arbitrary laboratory vectors without assuming Cartesian alignment."""
    vectors = 1e-9 * np.array([[2, 1, 0], [0, 3, 1], [1, 0, 4]])
    actual = lattice_positions((2, 2, 2), vectors)

    np.testing.assert_allclose(
        actual[[1, 2, 4, 7]],
        [vectors[2], vectors[1], vectors[0], vectors.sum(axis=0)],
        atol=1e-22,
        rtol=0.0,
    )


def test_explicit_positions_are_copied():
    """Preserve arbitrary coordinates without sharing the input array."""
    positions = np.array([[0.0, 0.0, 0.0], [1e-9, 2e-9, 3e-9]])
    actual = validate_positions(positions)
    actual[0, 0] = 1.0

    assert positions[0, 0] == 0.0


@pytest.mark.parametrize(
    "positions",
    [
        [],
        [[0, 0]],
        [[0, 0, np.nan]],
        [[0, 0, 1j]],
        [[0, 0, 0], [0, 0, 0]],
    ],
)
def test_invalid_positions_are_rejected(positions):
    """Reject malformed, nonfinite, complex or coincident positions."""
    with pytest.raises(ValueError):
        validate_positions(positions)


@pytest.mark.parametrize(
    "shape, vectors",
    [
        ((0, 1, 1), np.eye(3)),
        ((2.0, 1.0, 1.0), np.eye(3)),
        ((2, 2), np.eye(3)),
        ((2, 1, 1), np.zeros((3, 3))),
        ((2, 1, 1), np.full((3, 3), np.inf)),
    ],
)
def test_invalid_lattices_are_rejected(shape, vectors):
    """Reject invalid lattice shapes, vectors and coincident generated sites."""
    with pytest.raises(ValueError):
        lattice_positions(shape, vectors)


def test_register_frames_preserve_site_orientations():
    """Associate each crystallographic orientation with its position index."""
    positions = lattice_positions((4, 1, 1), 15e-9 * np.eye(3))
    axes = crystallographic_nv_axes()
    references = np.tile([1.0, 0.0, 0.0], (4, 1))

    frames = register_local_frames(positions, axes, references)

    assert frames.shape == (4, 3, 3)
    np.testing.assert_allclose(frames[:, :, 2], axes, atol=1e-12)
    for frame in frames:
        gram = np.einsum("ia,ib->ab", frame, frame, optimize=False)
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-12)
        assert np.linalg.det(frame) == pytest.approx(1.0)


def test_register_frames_follow_laboratory_rotation():
    """Rotate positions and orientation vectors consistently in the laboratory."""
    positions = lattice_positions((3, 1, 1), 15e-9 * np.eye(3))
    axes = np.tile(crystallographic_nv_axes()[0], (3, 1))
    references = np.tile([1.0, 0.0, 0.0], (3, 1))
    rotation = rotation_matrix("y", 0.37)

    original = register_local_frames(positions, axes, references)

    rotated = register_local_frames(
        np.einsum("ab,nb->na", rotation, positions),
        np.einsum("ab,nb->na", rotation, axes),
        np.einsum("ab,nb->na", rotation, references),
    )
    expected = np.einsum("ab,nbc->nac", rotation, original)

    np.testing.assert_allclose(rotated, expected, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    "axes, references",
    [
        ([[0, 0, 1]], [[1, 0, 0], [1, 0, 0]]),
        ([[0, 0, 1], [0, 0, 1]], [[1, 0, 0]]),
        ([[0, 0, 1], [0, 0, 1]], [[0, 0, 1], [1, 0, 0]]),
    ],
)
def test_register_frames_reject_inconsistent_inputs(axes, references):
    """Reject missing site vectors and references parallel to an NV axis."""
    positions = lattice_positions((2, 1, 1), 15e-9 * np.eye(3))

    with pytest.raises(ValueError):
        register_local_frames(positions, axes, references)
