import numpy as np
import pytest

from qutip import qeye

from echo_spin.controls import (
    control_amplitude,
    control_operator,
    microwave_coefficient,
    microwave_hamiltonian,
)

def test_constant_control_amplitude():
    """A constant control amplitude should be returned unchanged."""
    assert np.isclose(
        control_amplitude(2.0, time=5.0),
        2.0,
    )


def test_time_dependent_control_amplitude():
    """A callable control amplitude should be evaluated at the requested time."""
    amplitude = lambda t: 2.0 * t

    assert np.isclose(
        control_amplitude(amplitude, time=3.0),
        6.0,
    )


def test_microwave_coefficient_at_zero_time():
    """At zero time and zero phase, only the x-quadrature contributes."""
    coefficient = microwave_coefficient(
        time=0.0,
        omega=1.0,
        phase=0.0,
        omega_x=2.0,
        omega_y=3.0,
    )

    assert np.isclose(
        coefficient,
        np.sqrt(2.0) * 2.0,
    )


def test_control_operator_includes_nuclear_contribution():
    """The nuclear control term should be weighted by gamma_ratio."""
    electronic = [qeye(2)]
    nuclear = [2.0 * qeye(2)]

    control = control_operator(
        electronic_x_operators=electronic,
        nuclear_x_operators=nuclear,
        gamma_ratio=0.5,
    )

    expected = 2.0 * qeye(2)

    assert np.allclose(
        control.full(),
        expected.full(),
    )


def test_microwave_hamiltonian_sums_multiple_drives():
    """The microwave Hamiltonian should sum all carrier contributions."""
    control = qeye(2)

    drives = [
        {
            "omega": 1.0,
            "phase": 0.0,
            "omega_x": 1.0,
            "omega_y": 0.0,
        },
        {
            "omega": 2.0,
            "phase": 0.0,
            "omega_x": 2.0,
            "omega_y": 0.0,
        },
    ]

    hamiltonian = microwave_hamiltonian(
        time=0.0,
        control=control,
        drives=drives,
    )

    expected = 3.0 * np.sqrt(2.0) * control

    assert np.allclose(
        hamiltonian.full(),
        expected.full(),
    )