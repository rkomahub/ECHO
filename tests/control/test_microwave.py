import numpy as np
import pytest
from qutip import jmat, qeye, sigmax, sigmay, sigmaz

from echo_spin.control.microwave import (
    control_amplitude,
    control_hamiltonian,
    control_operator,
    linear_drive_qubit_hamiltonian,
    microwave_coefficient,
    microwave_hamiltonian,
)
from echo_spin.control.rotations import single_qubit_rotation
from echo_spin.dynamics.propagators import (
    rotating_frame_hamiltonian,
    time_dependent_propagator,
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


def test_generic_control_evaluates_independent_channels():
    """Generic control Hamiltonian evaluates independent channels."""
    sampled_times = []

    def y_amplitude(time):
        sampled_times.append(time)
        return 2 * time

    actual = control_hamiltonian(
        time=0.7,
        operators=[sigmax() / 2, sigmay() / 2, sigmaz() / 2],
        amplitudes=[0.3, y_amplitude, -0.2],
    )
    expected = (
        0.3 * sigmax() + 1.4 * sigmay() - 0.2 * sigmaz()
    ) / 2

    assert sampled_times == [0.7]
    assert (actual - expected).norm() < 1e-12


def test_generic_control_generates_arbitrary_axis_rotation():
    """Generic control Hamiltonian generating arbitrary axis rotation."""
    amplitudes = np.array([0.3, -0.4, 0.2])
    duration = 0.7

    def hamiltonian(time):
        return control_hamiltonian(
            time=time,
            operators=[sigmax() / 2, sigmay() / 2, sigmaz() / 2],
            amplitudes=amplitudes,
        )

    actual = time_dependent_propagator(
        hamiltonian,
        t0=0.0,
        t1=duration,
        steps=20,
    )
    expected = single_qubit_rotation(
        angle=np.linalg.norm(amplitudes) * duration,
        axis=amplitudes,
    )

    assert np.allclose(
        actual.full(),
        expected.full(),
        atol=1e-12,
        rtol=0.0,
    )


def test_generic_control_supports_spin_one():
    """Generic control Hamiltonian supports spin 1 operator."""
    sy = jmat(1, "y")
    actual = control_hamiltonian(0.4, [sy], [0.7])

    assert actual.dims == sy.dims
    assert (actual - 0.7 * sy).norm() < 1e-12


@pytest.mark.parametrize(
    "operators, amplitudes",
    [
        ([], []),
        ([sigmax()], []),
        ([sigmax(), qeye(3)], [1.0, 1.0]),
        ([sigmax() + 1j * sigmay()], [1.0]),
        ([sigmax()], [np.nan]),
        ([sigmax()], [lambda time: 1j]),
    ],
)
def test_generic_control_rejects_invalid_terms(operators, amplitudes):
    """Generic control Hamiltonian rejects invalid terms."""
    with pytest.raises(ValueError):
        control_hamiltonian(0.0, operators, amplitudes)


@pytest.mark.parametrize("phase", [0.0, 0.7, np.pi / 2])
@pytest.mark.parametrize("time", [0.0, 0.137])
def test_linear_drive_exact_frame_selection(time, phase):
    """Test linear drive Hamiltonian selects exact frame."""
    parameters = dict(
        omega_0=5.3,
        carrier=5.0,
        rabi_frequency=0.8,
        phase=phase,
    )

    def lab_hamiltonian(t):
        return linear_drive_qubit_hamiltonian(
            time=t, frame="lab", **parameters
        )

    transformed = rotating_frame_hamiltonian(
        time=time,
        driven_hamiltonian=lab_hamiltonian,
        generator=5.0 * sigmaz() / 2,
    )
    selected = linear_drive_qubit_hamiltonian(
        time=time,
        frame="rotating",
        rwa=False,
        **parameters,
    )

    assert (transformed - selected).norm() < 1e-12


@pytest.mark.parametrize("phase", [0.0, 0.7, np.pi / 2])
def test_linear_drive_rwa_preserves_detuning_and_phase(phase):
    """Test linear drive Hamiltonian preserves detuning and phase after RWA."""
    expected = (
        0.3 * sigmaz() / 2
        + 0.8
        * (np.cos(phase) * sigmax() + np.sin(phase) * sigmay())
        / 2
    )

    for time in [0.0, 0.137]:
        actual = linear_drive_qubit_hamiltonian(
            time=time,
            omega_0=5.3,
            carrier=5.0,
            rabi_frequency=0.8,
            phase=phase,
            frame="rotating",
            rwa=True,
        )

        assert (actual - expected).norm() < 1e-12


@pytest.mark.parametrize(
    "options",
    [
        {"frame": "unknown"},
        {"frame": "lab", "rwa": True},
        {"carrier": 0.0},
        {"phase": np.nan},
    ],
)
def test_linear_drive_rejects_invalid_options(options):
    """Test linear drive Hamiltonian rejects invalid options."""
    parameters = dict(
        time=0.0,
        omega_0=5.3,
        carrier=5.0,
        rabi_frequency=0.8,
    )
    parameters.update(options)

    with pytest.raises(ValueError):
        linear_drive_qubit_hamiltonian(**parameters)
