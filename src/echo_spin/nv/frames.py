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


def rotate_to_local_frame(
    vector,
    axis: str,
    angle: float,
) -> np.ndarray:
    """Express a laboratory-frame vector in a rotated local frame."""
    rotation = rotation_matrix(axis, -angle)

    return rotate_vector(vector, rotation)


def crystallographic_nv_axes() -> np.ndarray:
    """Return four unit NV-axis representatives in cubic crystal coordinates.

    Coordinate axes are [100], [010], [001].
    Rows represent [111], [1,-1,-1], [-1,1,-1], [-1,-1,1].
    These identify axis classes; defect polarity is not modeled separately.
    """
    return np.array([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]) / np.sqrt(3.0)


def local_frame_from_axis(
    nv_axis,
    transverse_reference,
) -> np.ndarray:
    """Return a right-handed frame with local axes stored as columns.

    Local z follows nv_axis. Local x follows the transverse projection
    of transverse_reference, and local y = z cross x.

    Both inputs must be expressed in the same coordinate system.
    The returned matrix R maps local vectors to that system;
    R.T maps vectors into the local frame.
    """
    vectors = []

    for value in (nv_axis, transverse_reference):
        if np.iscomplexobj(value):
            raise ValueError("Frame vectors must be real.")

        vector = np.asarray(value, dtype=float)

        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("Frame vectors must have three finite components.")

        scale = np.max(np.abs(vector))
        if scale == 0:
            raise ValueError("Frame vectors must be nonzero.")

        vector = vector / scale
        vectors.append(vector / np.linalg.norm(vector))

    z_axis, reference = vectors
    x_axis = reference - np.dot(reference, z_axis) * z_axis
    transverse_norm = np.linalg.norm(x_axis)

    if transverse_norm <= 1e-12:
        raise ValueError("The transverse reference must not parallel the NV axis.")

    x_axis /= transverse_norm
    y_axis = np.cross(z_axis, x_axis)

    return np.column_stack((x_axis, y_axis, z_axis))
