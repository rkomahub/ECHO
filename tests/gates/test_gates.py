import numpy as np
import pytest
from qutip import Qobj, qeye

from echo_spin.gates.gates import (
    sqrt_zz_gate,
    conditional_phase,
    remove_local_z_phases,
)


def test_sqrt_zz_gate_matrix():
    """Check the matrix representation of the sqrt(ZZ) gate."""
    gate = sqrt_zz_gate()

    expected = np.diag([1.0, 1.0j, 1.0j, 1.0])

    assert np.allclose(gate.full(), expected)


def test_sqrt_zz_gate_is_unitary():
    """Check that the sqrt(ZZ) gate is unitary."""
    gate = sqrt_zz_gate()

    identity = gate.dag() * gate

    assert np.allclose(identity.full(), qeye([2, 2]).full())


def test_sqrt_zz_conditional_phase():
    """Check the conditional phase of the sqrt(ZZ) gate."""
    phase = conditional_phase(sqrt_zz_gate())

    assert np.exp(1j * phase) == pytest.approx(-1.0)


def test_inverse_sqrt_zz_conditional_phase():
    """Check the conditional phase of the inverse sqrt(ZZ) gate."""
    phase = conditional_phase(sqrt_zz_gate().dag())

    assert np.exp(1j * phase) == pytest.approx(-1.0)


def test_remove_local_z_phases_preserves_conditional_phase():
    """Check that the conditional phase is unchanged after removing local Z phases."""
    operator = Qobj(
        np.diag([
            np.exp(1j * 0.3),
            np.exp(1j * 0.7),
            np.exp(-1j * 0.4),
            np.exp(1j * 1.2),
        ]),
        dims=[[2, 2], [2, 2]],
    )

    corrected = remove_local_z_phases(operator)

    assert conditional_phase(corrected) == pytest.approx(
        conditional_phase(operator)
    )


def test_remove_local_z_phases_removes_local_phases():
    """Check that local Z phases are removed correctly."""
    local_phases = Qobj(
        np.diag([
            1.0,
            np.exp(1j * 0.4),
            np.exp(-1j * 0.7),
            np.exp(-1j * 0.3),
        ]),
        dims=[[2, 2], [2, 2]],
    )

    corrected = remove_local_z_phases(local_phases)

    assert np.allclose(
        corrected.full(),
        np.eye(4),
    )