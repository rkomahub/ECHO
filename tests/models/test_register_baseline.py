import numpy as np
import pytest
from scipy.constants import hbar, mu_0, physical_constants

from echo_spin.models.longitudinal import (
    longitudinal_qubit_parameters,
    longitudinal_register_hamiltonian,
)
from echo_spin.nv.fields import (
    local_zeeman_vector,
    register_magnetic_fields,
)
from echo_spin.nv.frames import crystallographic_nv_axes
from echo_spin.nv.geometry import lattice_positions, register_local_frames
from echo_spin.nv.interactions import register_dipolar_tensors


def test_aligned_three_nv_geometry_builds_longitudinal_model():
    """Connect SI geometry and fields to all physical pair coefficients."""
    number_of_nv = 3
    spacing = 15e-9
    axis = crystallographic_nv_axes()[0]
    chain_direction = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)

    # Only the first lattice vector is active for this chain.
    vectors = spacing * np.array([
        chain_direction,
        axis,
        np.cross(axis, chain_direction),
    ])
    positions = lattice_positions((number_of_nv, 1, 1), vectors)
    axes = np.tile(axis, (number_of_nv, 1))
    references = np.tile(chain_direction, (number_of_nv, 1))
    frames = register_local_frames(positions, axes, references)

    fields = register_magnetic_fields(positions, 0.01 * axis)

    gamma_si = abs(physical_constants["electron gyromag. ratio"][0])
    gamma_us = gamma_si * 1e-6
    coefficients = np.array([
        local_zeeman_vector(field, frame, gamma_us)
        for field, frame in zip(fields, frames)
    ])

    np.testing.assert_allclose(
        coefficients,
        np.tile([0.0, 0.0, 0.01 * gamma_us], (number_of_nv, 1)),
        atol=1e-12,
        rtol=1e-12,
    )

    # Convert rad/s to rad/us; positions remain in metres.
    prefactor = mu_0 * hbar * gamma_si**2 / (4 * np.pi) * 1e-6
    tensors = register_dipolar_tensors(positions, frames, prefactor)
    detunings, couplings, constant = longitudinal_qubit_parameters(
        np.zeros(number_of_nv), tensors
    )

    nearest = prefactor / (4 * spacing**3)
    expected = nearest * np.array([
        [0.0, 1.0, 1 / 8],
        [1.0, 0.0, 1.0],
        [1 / 8, 1.0, 0.0],
    ])
    np.testing.assert_allclose(
        couplings, expected, atol=1e-14, rtol=1e-12
    )

    gate_duration = np.pi / (4 * abs(couplings[0, 1]))
    assert gate_duration == pytest.approx(
        np.pi * spacing**3 / prefactor
    )

    hamiltonian = longitudinal_register_hamiltonian(
        detunings, couplings, constant
    )
    assert hamiltonian.shape == (8, 8)
    assert hamiltonian.isherm
