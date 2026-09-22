import numpy as np


def rotation_matrix(axis: str, angle: float) -> np.ndarray:
    """Return the 3D rotation matrix for a Cartesian-axis rotation.

    Parameters
    ----------
    axis : str
        Rotation axis: "x", "y", or "z".
    angle : float
        Rotation angle in radians.

    Returns
    -------
    ndarray
        3x3 rotation matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)

    if axis == "x":
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ])

    if axis == "y":
        return np.array([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ])

    if axis == "z":
        return np.array([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ])

    raise ValueError("axis must be 'x', 'y', or 'z'.")


def rotate_vector(
    vector,
    rotation: np.ndarray,
) -> np.ndarray:
    """Rotate a three-dimensional vector."""
    vector = np.asarray(vector, dtype=float)
    rotation = np.asarray(rotation, dtype=float)

    if vector.shape != (3,):
        raise ValueError("vector must contain three Cartesian components.")

    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3).")

    return rotation @ vector