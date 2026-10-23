import numpy as np
import pytest

from echo_spin.nv.geometry import lattice_positions, validate_positions


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
