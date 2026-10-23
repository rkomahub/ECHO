import numpy as np
import pytest

from echo_spin.core.operators import embed_operator, spin_operators
from echo_spin.core.system import SpinSystem
from echo_spin.nv.frames import local_frame_from_axis
from echo_spin.nv.interactions import (
    dipolar_interaction,
    register_dipolar_tensors,
)


def test_register_tensors_have_expected_scaling_and_angular_factors():
    """Recover axial dipolar factors, inverse-cube scaling and zero self terms."""
    positions = [[0, 0, 0], [0, 0, 10e-9], [0, 0, 20e-9]]
    frames = [np.eye(3)] * 3
    tensors = register_dipolar_tensors(positions, frames, 1e-24)

    np.testing.assert_allclose(
        tensors[0, 1], np.diag([1, 1, -2]), atol=1e-12
    )
    np.testing.assert_allclose(
        tensors[0, 2], tensors[0, 1] / 8, atol=1e-12
    )
    for i in range(3):
        np.testing.assert_array_equal(tensors[i, i], np.zeros((3, 3)))
        for j in range(3):
            np.testing.assert_allclose(tensors[j, i], tensors[i, j].T)


def test_register_tensor_matches_existing_spin_one_interaction():
    """Reconstruct the existing physical pair interaction from its local tensor."""
    displacement = np.array([8e-9, -6e-9, 4e-9])
    frames = [
        local_frame_from_axis([1, 1, 1], [1, 0, 0]),
        local_frame_from_axis([1, -1, -1], [1, 0, 0]),
    ]
    prefactor = 1e-24
    tensors = register_dipolar_tensors(
        [np.zeros(3), displacement], frames, prefactor
    )

    system = SpinSystem([1, 1])
    operators = spin_operators(1)
    actual = 0 * embed_operator(operators[0], 0, system)

    for a, operator_i in enumerate(operators):
        for b, operator_j in enumerate(operators):
            actual += (
                tensors[0, 1, a, b]
                * embed_operator(operator_i, 0, system)
                * embed_operator(operator_j, 1, system)
            )

    expected = dipolar_interaction(
        system=system,
        site_i=0,
        site_j=1,
        coupling=prefactor / np.linalg.norm(displacement)**3,
        direction=displacement,
        frame_i=frames[0],
        frame_j=frames[1],
    )

    np.testing.assert_allclose(
        actual.full(), expected.full(), atol=1e-12, rtol=0.0
    )


@pytest.mark.parametrize(
    "frames, prefactor",
    [
        ([np.eye(3)], 1e-24),
        ([np.eye(3), 2 * np.eye(3)], 1e-24),
        ([np.eye(3), np.eye(3)], np.nan),
        ([np.eye(3), np.eye(3)], 1j),
    ],
)
def test_register_tensors_reject_invalid_inputs(frames, prefactor):
    """Reject missing or invalid frames and nonfinite or complex prefactors."""
    with pytest.raises(ValueError):
        register_dipolar_tensors(
            [[0, 0, 0], [10e-9, 0, 0]], frames, prefactor
        )
