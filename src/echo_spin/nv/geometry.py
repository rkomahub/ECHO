from itertools import product

import numpy as np


def validate_positions(positions) -> np.ndarray:
    """Return a copy of distinct finite laboratory positions in metres."""
    if np.iscomplexobj(positions):
        raise ValueError("positions must be real.")

    try:
        array = np.asarray(positions, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("positions must contain real numbers.") from error

    if (
        array.ndim != 2
        or array.shape[1] != 3
        or len(array) == 0
        or not np.all(np.isfinite(array))
    ):
        raise ValueError("positions must be a finite nonempty (N, 3) array.")

    if len(np.unique(array, axis=0)) != len(array):
        raise ValueError("Sites must have distinct positions.")

    return array.copy()


def lattice_positions(
    shape,
    lattice_vectors,
    origin=(0.0, 0.0, 0.0),
) -> np.ndarray:
    """Generate laboratory positions in metres with the last index fastest."""
    counts = np.asarray(shape)
    if (
        counts.shape != (3,)
        or counts.dtype.kind not in "iu"
        or np.any(counts <= 0)
    ):
        raise ValueError("shape must contain three positive integers.")

    if np.iscomplexobj(lattice_vectors) or np.iscomplexobj(origin):
        raise ValueError("lattice_vectors and origin must be real.")

    vectors = np.asarray(lattice_vectors, dtype=float)
    offset = np.asarray(origin, dtype=float)
    if (
        vectors.shape != (3, 3)
        or offset.shape != (3,)
        or not np.all(np.isfinite(vectors))
        or not np.all(np.isfinite(offset))
    ):
        raise ValueError(
            "Provide finite lattice_vectors of shape (3, 3) "
            "and origin of shape (3,)."
        )

    indices = np.array(
        list(product(*(range(int(count)) for count in counts))),
        dtype=float,
    )

    # Each row of vectors is one laboratory-frame lattice vector.
    with np.errstate(over="ignore", invalid="ignore"):
        positions = offset + np.einsum(
            "nk,kj->nj", indices, vectors, optimize=False
        )

    return validate_positions(positions)
