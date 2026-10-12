import numpy as np
import pytest
from qutip import basis, jmat, qeye, sigmax, sigmay, sigmaz, tensor

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
    static_propagator,
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
    def amplitude(time):
        """Return a linearly increasing control amplitude."""
        return 2.0 * time

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


@pytest.mark.parametrize("detuning_ratio", [0.0, 1.0, 4.0, 20.0])
def test_shared_carrier_detuned_spectator_excitation(detuning_ratio):
    """Verify spectator excitation under a shared carrier against the detuned Rabi solution."""
    rabi_frequency = 1.0
    carrier = 100.0
    detuning = detuning_ratio * rabi_frequency
    pulse_duration = np.pi / rabi_frequency

    addressed_hamiltonian = linear_drive_qubit_hamiltonian(
        time=0.0,
        omega_0=carrier,
        carrier=carrier,
        rabi_frequency=rabi_frequency,
        frame="rotating",
        rwa=True,
    )

    spectator_hamiltonian = linear_drive_qubit_hamiltonian(
        time=0.0,
        omega_0=carrier + detuning,
        carrier=carrier,
        rabi_frequency=rabi_frequency,
        frame="rotating",
        rwa=True,
    )

    addressed_propagator = static_propagator(
        hamiltonian=addressed_hamiltonian,
        time=pulse_duration,
    )
    spectator_propagator = static_propagator(
        hamiltonian=spectator_hamiltonian,
        time=pulse_duration,
    )

    final_state = tensor(
        addressed_propagator,
        spectator_propagator,
    ) * tensor(basis(2, 0), basis(2, 0))

    addressed_density = final_state.ptrace(0)
    spectator_density = final_state.ptrace(1)

    addressed_excitation = float(np.real(
        basis(2, 1).dag()
        * addressed_density
        * basis(2, 1)
    ))
    spectator_excitation = float(np.real(
        basis(2, 1).dag()
        * spectator_density
        * basis(2, 1)
    ))

    generalized_rabi = np.sqrt(
        rabi_frequency**2 + detuning**2
    )
    excitation_bound = (
        rabi_frequency**2 / generalized_rabi**2
    )
    expected_excitation = (
        excitation_bound
        * np.sin(
            generalized_rabi * pulse_duration / 2
        ) ** 2
    )

    assert addressed_excitation == pytest.approx(
        1.0, abs=1e-12
    )
    assert spectator_excitation == pytest.approx(
        expected_excitation, abs=1e-12
    )
    assert spectator_excitation <= excitation_bound + 1e-12


@pytest.mark.parametrize(
    "fractional_error",
    [-0.1, -0.02, 0.0, 0.02, 0.1],
)
def test_resonant_pi_pulse_amplitude_error(fractional_error):
    """Verify population and gate-fidelity errors from a miscalibrated pi pulse."""
    nominal_rabi = 1.0
    carrier = 100.0
    pulse_duration = np.pi / nominal_rabi

    hamiltonian = linear_drive_qubit_hamiltonian(
        time=0.0,
        omega_0=carrier,
        carrier=carrier,
        rabi_frequency=nominal_rabi * (1 + fractional_error),
        frame="rotating",
        rwa=True,
    )

    propagator = static_propagator(
        hamiltonian=hamiltonian,
        time=pulse_duration,
    )

    initial = basis(2, 0)
    final = propagator * initial

    transfer = abs(basis(2, 1).overlap(final)) ** 2

    angle_error = np.pi * fractional_error
    expected_transfer = np.cos(angle_error / 2) ** 2

    ideal_pi = -1j * np.array(
        [[0.0, 1.0], [1.0, 0.0]],
        dtype=complex,
    )
    actual = propagator.full()

    overlap = np.trace(
        ideal_pi.conj().T @ actual
    )
    average_fidelity = (
        2 + abs(overlap) ** 2
    ) / 6

    expected_fidelity = (
        1 + 2 * np.cos(angle_error / 2) ** 2
    ) / 3

    assert transfer == pytest.approx(
        expected_transfer,
        abs=1e-12,
    )
    assert average_fidelity == pytest.approx(
        expected_fidelity,
        abs=1e-12,
    )
    assert propagator.isunitary


@pytest.mark.parametrize(
    "detuning_ratio",
    [-1.0, -0.1, 0.0, 0.1, 1.0],
)
def test_pi_pulse_detuning_gate_fidelity(detuning_ratio):
    """Verify detuned pi-pulse fidelity against the exact rotating-frame solution."""
    rabi_frequency = 1.0
    carrier = 100.0
    detuning = detuning_ratio * rabi_frequency
    pulse_duration = np.pi / rabi_frequency

    hamiltonian = linear_drive_qubit_hamiltonian(
        time=0.0,
        omega_0=carrier + detuning,
        carrier=carrier,
        rabi_frequency=rabi_frequency,
        frame="rotating",
        rwa=True,
    )

    propagator = static_propagator(
        hamiltonian=hamiltonian,
        time=pulse_duration,
    )

    ideal_pi = -1j * np.array(
        [[0.0, 1.0], [1.0, 0.0]],
        dtype=complex,
    )

    overlap = np.trace(
        ideal_pi.conj().T @ propagator.full()
    )
    average_fidelity = (
        2 + abs(overlap) ** 2
    ) / 6

    generalized_rabi = np.sqrt(
        rabi_frequency**2 + detuning**2
    )
    expected_transfer = (
        rabi_frequency**2
        / generalized_rabi**2
        * np.sin(
            generalized_rabi * pulse_duration / 2
        ) ** 2
    )
    expected_fidelity = (
        1 + 2 * expected_transfer
    ) / 3

    final_state = propagator * basis(2, 0)
    transfer = abs(
        basis(2, 1).overlap(final_state)
    ) ** 2

    assert transfer == pytest.approx(
        expected_transfer,
        abs=1e-12,
    )
    assert average_fidelity == pytest.approx(
        expected_fidelity,
        abs=1e-12,
    )
