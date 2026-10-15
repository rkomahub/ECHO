import numpy as np
import pytest

from echo_spin.core.operators import spin_operators
from echo_spin.core.system import SpinSystem
from echo_spin.nv.fields import (
    linear_magnetic_field,
    local_zeeman_vector,
)
from echo_spin.nv.frames import rotation_matrix
from echo_spin.nv.hamiltonians import electronic_nv_hamiltonian


def test_uniform_field_is_position_independent():
    """Recover the same laboratory field at every position without a gradient."""
    field = [0.001, -0.002, 0.01]

    for position in ([0, 0, 0], [1e-8, 2e-8, -3e-8]):
        np.testing.assert_allclose(
            linear_magnetic_field(position, field),
            field,
            atol=1e-15,
            rtol=0.0,
        )


def test_gradient_generates_expected_field_difference():
    """Produce the expected field difference between two register positions."""
    gradient = np.diag([-500.0, -500.0, 1000.0])
    reference = np.array([1e-6, 2e-6, 3e-6])
    field = np.array([0.0, 0.0, 0.01])

    at_reference = linear_magnetic_field(
        reference, field, gradient, reference
    )
    displaced = linear_magnetic_field(
        reference + [0.0, 0.0, 20e-9],
        field,
        gradient,
        reference,
    )

    np.testing.assert_allclose(
        at_reference, field, atol=1e-15, rtol=0.0
    )
    np.testing.assert_allclose(
        displaced - at_reference,
        [0.0, 0.0, 20e-6],
        atol=1e-15,
        rtol=0.0,
    )


def test_field_and_frame_rotate_consistently():
    """Preserve local Zeeman coefficients under a common laboratory rotation."""
    field = np.array([0.003, -0.002, 0.01])
    frame = rotation_matrix("y", 0.7)
    laboratory_rotation = rotation_matrix("z", 0.4)
    gamma = 7.3

    original = local_zeeman_vector(field, frame, gamma)
    rotated = local_zeeman_vector(
        laboratory_rotation @ field,
        laboratory_rotation @ frame,
        gamma,
    )

    np.testing.assert_allclose(
        rotated, original, atol=1e-12, rtol=0.0
    )


def test_gradient_field_enters_existing_electronic_hamiltonian():
    """Convert a field difference into the expected electronic Zeeman difference."""
    gradient = np.diag([-500.0, -500.0, 1000.0])
    field = [0.0, 0.0, 0.01]
    gamma = 100.0  # Synthetic angular-frequency coefficient for this test.
    system = SpinSystem([1])

    hamiltonians = []
    for position in ([0.0, 0.0, 0.0], [0.0, 0.0, 20e-9]):
        local_field = linear_magnetic_field(position, field, gradient)
        omega = local_zeeman_vector(local_field, np.eye(3), gamma)

        hamiltonians.append(
            electronic_nv_hamiltonian(
                system=system,
                electron_site=0,
                D=10.0,
                omega_e=omega,
            )
        )

    _, _, sz = spin_operators(1)
    expected = gamma * 20e-6 * sz

    np.testing.assert_allclose(
        (hamiltonians[1] - hamiltonians[0]).full(),
        expected.full(),
        atol=1e-12,
        rtol=0.0,
    )


def test_gradient_rejects_nonzero_divergence():
    """Reject a magnetic gradient with nonzero divergence even when currents are allowed."""
    with pytest.raises(ValueError, match="traceless"):
        linear_magnetic_field(
            [0, 0, 0],
            [0, 0, 0],
            np.eye(3),
            source_free=False,
        )


def test_current_free_gradient_requires_zero_curl():
    """Allow an antisymmetric gradient only when the source-free restriction is disabled."""
    gradient = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])

    with pytest.raises(ValueError, match="symmetric"):
        linear_magnetic_field([1, 0, 0], [0, 0, 0], gradient)

    actual = linear_magnetic_field(
        [1, 0, 0],
        [0, 0, 0],
        gradient,
        source_free=False,
    )

    np.testing.assert_allclose(actual, [0, 1, 0])


@pytest.mark.parametrize(
    "frame",
    [np.diag([1.0, 1.0, -1.0]), 2 * np.eye(3)],
)
def test_local_zeeman_vector_rejects_invalid_frames(frame):
    """Reject reflected or nonorthonormal coordinate frames."""
    with pytest.raises(ValueError):
        local_zeeman_vector([0, 0, 0.01], frame, 1.0)


@pytest.mark.parametrize(
    "position",
    [[0, 0], [0, np.nan, 0], [0, 0, 1j]],
)
def test_linear_field_rejects_invalid_positions(position):
    """Reject malformed, nonfinite or complex spatial coordinates."""
    with pytest.raises(ValueError):
        linear_magnetic_field(position, [0, 0, 0.01])
