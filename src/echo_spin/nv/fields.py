import numpy as np

from echo_spin.nv.geometry import validate_positions


def _finite_real_array(value, shape, name):
    """Validate and return a finite real array with the required shape."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real.")

    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain real numbers.") from error

    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must have shape {shape} and contain finite values."
        )

    return array


def linear_magnetic_field(
    position,
    field_at_reference,
    gradient=None,
    reference_position=(0.0, 0.0, 0.0),
    source_free=True,
) -> np.ndarray:
    """Evaluate a static affine laboratory field in tesla at a position in metres."""
    position = _finite_real_array(position, (3,), "position")
    reference = _finite_real_array(
        reference_position, (3,), "reference_position"
    )
    field = _finite_real_array(
        field_at_reference, (3,), "field_at_reference"
    )

    if not isinstance(source_free, (bool, np.bool_)):
        raise ValueError("source_free must be boolean.")

    if gradient is None:
        gradient = np.zeros((3, 3))
    else:
        gradient = _finite_real_array(gradient, (3, 3), "gradient")

    # G[a, b] = derivative of B_a with respect to position_b, in T/m.
    scale = float(np.max(np.abs(gradient)))
    scaled = gradient / scale if scale > 0.0 else gradient

    # Magnetic fields must always satisfy div(B) = 0.
    if abs(np.trace(scaled)) > 1e-10:
        raise ValueError("gradient must be traceless: div(B) must vanish.")

    # In a static current-free region, curl(B) also vanishes.
    if source_free and not np.allclose(
        scaled,
        scaled.T,
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError(
            "gradient must be symmetric in a source-free region."
        )

    result = field + np.einsum(
        "ab,b->a",
        gradient,
        position - reference,
        optimize=False,
    )

    if not np.all(np.isfinite(result)):
        raise ValueError("The evaluated field must be finite.")

    return result


def local_zeeman_vector(
    field_lab,
    local_frame,
    gyromagnetic_ratio,
) -> np.ndarray:
    """Convert a laboratory field to local angular-frequency Hamiltonian coefficients."""
    field = _finite_real_array(field_lab, (3,), "field_lab")
    frame = _finite_real_array(local_frame, (3, 3), "local_frame")
    gamma = _finite_real_array(
        gyromagnetic_ratio, (), "gyromagnetic_ratio"
    )

    gram = np.einsum("ia,ib->ab", frame, frame, optimize=False)

    if (
        not np.allclose(gram, np.eye(3), atol=1e-10, rtol=0.0)
        or not np.isclose(
            np.linalg.det(frame), 1.0, atol=1e-10, rtol=0.0
        )
    ):
        raise ValueError("local_frame must be a right-handed orthonormal frame.")

    # Frame columns are local axes expressed in laboratory coordinates.
    field_local = np.einsum(
        "ia,i->a",
        frame,
        field,
        optimize=False,
    )
    result = float(gamma) * field_local

    if not np.all(np.isfinite(result)):
        raise ValueError("The local Zeeman vector must be finite.")

    return result


def register_magnetic_fields(
    positions,
    field_at_reference,
    gradient=None,
    reference_position=(0.0, 0.0, 0.0),
    source_free=True,
) -> np.ndarray:
    """Return laboratory magnetic fields in tesla for ordered sites in metres."""
    positions = validate_positions(positions)

    return np.stack([
        linear_magnetic_field(
            position=position,
            field_at_reference=field_at_reference,
            gradient=gradient,
            reference_position=reference_position,
            source_free=source_free,
        )
        for position in positions
    ])
