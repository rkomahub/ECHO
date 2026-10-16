import numpy as np
import pytest
from qutip import qeye, tensor

from echo_spin.core.operators import spin_operators
from echo_spin.core.system import SpinSystem
from echo_spin.nv.frames import rotation_matrix
from echo_spin.nv.interactions import dipolar_interaction
from echo_spin.nv.registers import nv_register_hamiltonian


def test_identity_frames_preserve_dipolar_interaction():
    """Recover the original Cartesian dipolar expression with identity frames."""
    system = SpinSystem([1, 1])
    direction = np.array([1.0, 2.0, -0.5])
    direction /= np.linalg.norm(direction)
    coupling = 0.17

    spins = spin_operators(1)
    scalar = sum(tensor(operator, operator) for operator in spins)
    projection = sum(
        component * operator
        for component, operator in zip(direction, spins)
    )
    expected = coupling * (scalar - 3 * tensor(projection, projection))

    actual = dipolar_interaction(
        system, 0, 1, coupling, direction,
        frame_i=np.eye(3),
        frame_j=np.eye(3),
    )
    default = dipolar_interaction(system, 0, 1, coupling, direction)

    np.testing.assert_allclose(actual.full(), expected.full(), atol=1e-12)
    np.testing.assert_allclose(default.full(), expected.full(), atol=1e-12)


def test_local_frames_match_quantum_basis_transformation():
    """Agree with independently rotating the two spin-one quantum bases."""
    system = SpinSystem([1, 1])
    angle_i = 0.37
    angle_j = -0.61
    direction = [1.0, -0.4, 0.8]

    laboratory = dipolar_interaction(system, 0, 1, 0.23, direction)

    _, sy, _ = spin_operators(1)
    quantum_rotation = tensor(
        (-1j * angle_i * sy).expm(),
        (-1j * angle_j * sy).expm(),
    )
    expected = quantum_rotation.dag() * laboratory * quantum_rotation

    actual = dipolar_interaction(
        system, 0, 1, 0.23, direction,
        frame_i=rotation_matrix("y", angle_i),
        frame_j=rotation_matrix("y", angle_j),
    )

    np.testing.assert_allclose(
        actual.full(), expected.full(), atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        actual.full(), actual.full().conj().T,
        atol=1e-12, rtol=0.0,
    )


def test_common_laboratory_rotation_preserves_local_interaction():
    """Preserve the local matrix when geometry and both frames rotate together."""
    system = SpinSystem([1, 1])
    frame_i = rotation_matrix("x", 0.3)
    frame_j = rotation_matrix("y", -0.7)
    common_rotation = rotation_matrix("z", 0.6)
    direction = np.array([0.4, 0.7, -0.2])

    original = dipolar_interaction(
        system, 0, 1, 0.13, direction,
        frame_i=frame_i,
        frame_j=frame_j,
    )
    rotated = dipolar_interaction(
        system, 0, 1, 0.13, common_rotation @ direction,
        frame_i=common_rotation @ frame_i,
        frame_j=common_rotation @ frame_j,
    )

    np.testing.assert_allclose(
        rotated.full(), original.full(), atol=1e-12, rtol=0.0
    )


def test_exchanging_pair_labels_preserves_interaction():
    """Preserve the embedded interaction after exchanging sites and their frames."""
    system = SpinSystem([1, 1])
    frame_i = rotation_matrix("x", 0.3)
    frame_j = rotation_matrix("y", -0.7)
    direction = np.array([0.4, 0.7, -0.2])

    forward = dipolar_interaction(
        system, 0, 1, 0.13, direction,
        frame_i=frame_i,
        frame_j=frame_j,
    )
    backward = dipolar_interaction(
        system, 1, 0, 0.13, -direction,
        frame_i=frame_j,
        frame_j=frame_i,
    )

    np.testing.assert_allclose(
        backward.full(), forward.full(), atol=1e-12, rtol=0.0
    )


def test_register_positions_pass_local_frames_to_pair_interaction():
    """Assemble a geometry-derived pair with the supplied local NV frames."""
    system = SpinSystem([1, 1, 1, 1])
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.5]])
    frames = [
        rotation_matrix("x", 0.3),
        rotation_matrix("y", -0.7),
    ]
    parameters = [
        {
            "D": 0.0,
            "omega_e": [0.0, 0.0, 0.0],
            "Q": 0.0,
            "omega_n": [0.0, 0.0, 0.0],
            "A": np.zeros((3, 3)),
        }
        for _ in range(2)
    ]
    displacement = positions[1] - positions[0]
    prefactor = 0.2

    actual = nv_register_hamiltonian(
        system=system,
        nv_parameters=parameters,
        positions=positions,
        coupling_prefactor=prefactor,
        local_frames=frames,
    )
    expected = dipolar_interaction(
        system=system,
        site_i=0,
        site_j=2,
        coupling=prefactor / np.linalg.norm(displacement)**3,
        direction=displacement,
        frame_i=frames[0],
        frame_j=frames[1],
    )

    assert actual.dims == qeye([3, 3, 3, 3]).dims
    np.testing.assert_allclose(
        actual.full(), expected.full(), atol=1e-12, rtol=0.0
    )


@pytest.mark.parametrize(
    "frame",
    [
        np.diag([1.0, 1.0, -1.0]),
        2 * np.eye(3),
        np.full((3, 3), np.nan),
        np.eye(2),
    ],
)
def test_dipolar_interaction_rejects_invalid_local_frames(frame):
    """Reject frames that do not define finite proper spatial rotations."""
    with pytest.raises(ValueError):
        dipolar_interaction(
            SpinSystem([1, 1]), 0, 1, 0.2, [0, 0, 1],
            frame_i=frame,
        )
