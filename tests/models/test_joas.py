import numpy as np
import pytest

from qutip import Qobj, qeye, basis, tensor, sigmax, sigmay

from echo_spin.core.system import SpinSystem
from echo_spin.core.operators import embed_operator, spin_operators
from echo_spin.core.basis import (
    dressed_spin_one_basis,
    dressed_spin_operators,
    operator_in_basis,
    project_operator,
)
from echo_spin.control.rotations import selective_rotation, two_qubit_rotation
from echo_spin.nv.registers import nv_register_hamiltonian
from echo_spin.nv.frames import rotate_to_local_frame
from echo_spin.nv.hamiltonians import electronic_nv_hamiltonian, nv_hamiltonian
from echo_spin.dynamics.sequences import (
    toggling_hamiltonians,
    toggling_propagator,
    finite_pulse_propagator,
    finite_pulse_sequence_propagator
)
from echo_spin.models.joas import (
    driven_hamiltonian,
    electron_logical_states,
    free_hamiltonian,
    interaction_time,
    reduced_free_hamiltonian,
    sequence_duration,
    two_electron_hamiltonian,
    two_electron_logical_basis,
    two_pulse_gate_durations,
    xy8_gate_intervals,
    xy8_gate_pulses,
    xy8_gate_schedule,
    xy8_gate_times
)
from echo_spin.gates.gates import (
    sqrt_zz_gate,
    conditional_phase,
    remove_local_z_phases,
    average_gate_fidelity,
)
from echo_spin.dynamics.propagators import(
     electron_nuclear_dynamical_map,
     static_propagator,
     time_dependent_propagator,
     unitary_dynamical_map,
)
from echo_spin.control.pulses import sine_envelope, sine_pi_pulse_amplitude, centered_sine_envelope
from echo_spin.control.microwave import microwave_hamiltonian
from echo_spin.spectroscopy.transitions import allowed_transitions

def test_free_hamiltonian_sums_nv_terms_and_interaction():
    """The free Hamiltonian should sum all NV and interaction terms."""
    h_nv_1 = 1.0 * qeye(3)
    h_nv_2 = 2.0 * qeye(3)
    interaction = 0.5 * qeye(3)

    h_free = free_hamiltonian(
        nv_hamiltonians=[h_nv_1, h_nv_2],
        interaction=interaction,
    )

    expected = 3.5 * qeye(3)

    assert np.allclose(
        h_free.full(),
        expected.full(),
    )


def test_free_hamiltonian_requires_nv_system():
    """The free Hamiltonian requires at least one NV contribution."""
    import pytest

    with pytest.raises(ValueError):
        free_hamiltonian(
            nv_hamiltonians=[],
            interaction=qeye(3),
        )


def test_driven_hamiltonian_adds_microwave_term():
    """The driven Hamiltonian should equal H_free + H_mw at each time."""
    free = 2.0 * qeye(2)
    control = qeye(2)

    drives = [
        {
            "omega": 1.0,
            "phase": 0.0,
            "omega_x": 1.0,
            "omega_y": 0.0,
        }
    ]

    hamiltonian = driven_hamiltonian(
        time=0.0,
        free=free,
        control=control,
        drives=drives,
    )

    expected = (
        2.0 + np.sqrt(2.0)
    ) * qeye(2)

    assert np.allclose(
        hamiltonian.full(),
        expected.full(),
    )


def test_joas_two_pulse_toggling_hamiltonians():
    """The toggling Hamiltonians for the Joas two-pulse sequence. Periods 1 -- 5"""
    delta_1 = 0.31
    delta_2 = -0.17
    g = 0.08

    free = reduced_free_hamiltonian(
        delta_1=delta_1,
        delta_2=delta_2,
        coupling=g,
    )

    x1 = two_qubit_rotation(
    qubit=1,
    angle=np.pi,
    axis="x",
    )

    x2 = two_qubit_rotation(
        qubit=0,
        angle=np.pi,
        axis="x",
    )

    hamiltonians = toggling_hamiltonians(
        free_hamiltonian=free,
        pulses=[x1, x2, x1, x2],
    )

    expected = [
        np.diag([
            delta_1,
            0,
            delta_1 + delta_2 - g,
            delta_2,
        ]),
        np.diag([
            0,
            delta_1,
            delta_2,
            delta_1 + delta_2 - g,
        ]),
        np.diag([
            delta_2,
            delta_1 + delta_2 - g,
            0,
            delta_1,
        ]),
        np.diag([
            delta_1 + delta_2 - g,
            delta_2,
            delta_1,
            0,
        ]),
        np.diag([
            delta_1,
            0,
            delta_1 + delta_2 - g,
            delta_2,
        ]),
    ]

    assert len(hamiltonians) == 5

    for actual, target in zip(hamiltonians, expected):
        assert np.allclose(actual.full(), target)


def test_two_pulse_gate_durations():
    """The two-pulse gate durations are consistent with the Joas timing convention."""
    tau_1 = 0.8
    tau_2 = 0.2

    durations = two_pulse_gate_durations(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    expected = [
        0.4,
        0.2,
        0.6,
        0.2,
        0.2,
    ]

    assert np.allclose(durations, expected)


def test_two_pulse_gate_durations_rejects_large_tau_2():
    """The two-pulse gate durations reject tau_2 > tau_1 / 2."""
    with pytest.raises(ValueError):
        two_pulse_gate_durations(
            tau_1=0.8,
            tau_2=0.5,
        )


def test_joas_two_pulse_gate_refocuses_detunings():
    """The Joas sequence refocuses detunings while retaining the ZZ phase."""
    delta_1 = 0.31
    delta_2 = -0.17
    g = 0.08

    tau_1 = 1.0
    tau_2 = 0.2

    free = reduced_free_hamiltonian(
        delta_1=delta_1,
        delta_2=delta_2,
        coupling=g,
    )

    # Joas reduced-basis convention:
    # NV1 -> ECHO qubit 1
    # NV2 -> ECHO qubit 0
    x1 = two_qubit_rotation(1, np.pi, "x")
    x2 = two_qubit_rotation(0, np.pi, "x")

    hamiltonians = toggling_hamiltonians(
        free_hamiltonian=free,
        pulses=[x1, x2, x1, x2],
    )

    durations = two_pulse_gate_durations(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    propagator = toggling_propagator(
        hamiltonians=hamiltonians,
        durations=durations,
    )

    # Remove the physically irrelevant global phase.
    matrix = propagator.full()
    matrix = matrix * np.exp(-1j * np.angle(matrix[0, 0]))

    expected_phase = 2 * g * tau_2

    expected = np.diag([
        1.0,
        np.exp(1j * expected_phase),
        np.exp(1j * expected_phase),
        1.0,
    ])

    assert np.allclose(matrix, expected)


def test_joas_two_pulse_gate_is_independent_of_detunings():
    """Different static detunings give the same gate up to global phase."""
    g = 0.08
    tau_1 = 1.0
    tau_2 = 0.2

    x1 = two_qubit_rotation(1, np.pi, "x")
    x2 = two_qubit_rotation(0, np.pi, "x")

    durations = two_pulse_gate_durations(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    def gate(delta_1, delta_2):
        free = reduced_free_hamiltonian(
            delta_1=delta_1,
            delta_2=delta_2,
            coupling=g,
        )

        hamiltonians = toggling_hamiltonians(
            free_hamiltonian=free,
            pulses=[x1, x2, x1, x2],
        )

        propagator = toggling_propagator(
            hamiltonians=hamiltonians,
            durations=durations,
        )

        matrix = propagator.full()

        # Remove global phase relative to the first diagonal element.
        return matrix * np.exp(-1j * np.angle(matrix[0, 0]))

    gate_a = gate(
        delta_1=0.31,
        delta_2=-0.17,
    )

    gate_b = gate(
        delta_1=2.7,
        delta_2=-1.4,
    )

    assert np.allclose(gate_a, gate_b)


def test_joas_two_pulse_sequence_generates_sqrt_zz():
    """The ideal Joas sequence generates sqrt(ZZ) up to global phase."""
    delta_1 = 0.31
    delta_2 = -0.17
    g = 0.08

    # For N_pi = 2, require 2 * g * tau_2 = pi / 2.
    tau_2 = np.pi / (4 * g)

    # tau_1 must satisfy tau_2 <= tau_1 / 2.
    tau_1 = 2.5 * tau_2

    free = reduced_free_hamiltonian(
        delta_1=delta_1,
        delta_2=delta_2,
        coupling=g,
    )

    # Joas reduced-basis convention:
    # NV1 -> ECHO qubit 1
    # NV2 -> ECHO qubit 0
    x1 = two_qubit_rotation(1, np.pi, "x")
    x2 = two_qubit_rotation(0, np.pi, "x")

    hamiltonians = toggling_hamiltonians(
        free_hamiltonian=free,
        pulses=[x1, x2, x1, x2],
    )

    durations = two_pulse_gate_durations(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    propagator = toggling_propagator(
        hamiltonians=hamiltonians,
        durations=durations,
    )

    # Remove global phase.
    matrix = propagator.full()
    matrix *= np.exp(-1j * np.angle(matrix[0, 0]))

    expected = sqrt_zz_gate().full()

    assert np.allclose(matrix, expected)


def test_sqrt_zz_gate_time_scales_inverse_with_coupling():
    """The sqrt(ZZ) interaction time scales as 1/g."""
    g_1 = 0.08
    g_2 = 2 * g_1

    tau_2_1 = np.pi / (4 * g_1)
    tau_2_2 = np.pi / (4 * g_2)

    assert np.isclose(tau_2_2, tau_2_1 / 2)


def test_finite_pi_pulse_on_nv1_converges_to_ideal_rotation():
    """A short finite pulse on NV1 approaches the ideal X pi rotation."""
    delta_1 = 0.31
    delta_2 = -0.17
    g = 0.08

    duration = 0.01
    steps = 1000

    free = reduced_free_hamiltonian(
        delta_1=delta_1,
        delta_2=delta_2,
        coupling=g,
    )

    x1 = two_qubit_rotation(
        qubit=1,  # NV1 in the Joas reduced-basis convention
        angle=np.pi,
        axis="x",
    )

    x1_operator = tensor(qeye(2), sigmax())

    peak_amplitude = sine_pi_pulse_amplitude(duration)

    def hamiltonian(time):
        omega = sine_envelope(
            time=time,
            duration=duration,
            peak_amplitude=peak_amplitude,
        )

        return free + 0.5 * omega * x1_operator

    propagator = time_dependent_propagator(
        hamiltonian=hamiltonian,
        t0=0.0,
        t1=duration,
        steps=steps,
    )

    assert np.allclose(
        propagator.full(),
        x1.full(),
        atol=1e-2,
    )


def test_finite_pi_pulse_on_nv2_converges_to_ideal_rotation():
    """A short finite pulse on NV2 approaches the ideal X pi rotation."""
    delta_1 = 0.31
    delta_2 = -0.17
    g = 0.08

    duration = 0.01
    steps = 1000

    free = reduced_free_hamiltonian(
        delta_1=delta_1,
        delta_2=delta_2,
        coupling=g,
    )

    x2 = two_qubit_rotation(
        qubit=0,  # NV2 in the Joas reduced-basis convention
        angle=np.pi,
        axis="x",
    )

    x2_operator = tensor(sigmax(), qeye(2))

    peak_amplitude = sine_pi_pulse_amplitude(duration)

    def hamiltonian(time):
        omega = sine_envelope(
            time=time,
            duration=duration,
            peak_amplitude=peak_amplitude,
        )

        return free + 0.5 * omega * x2_operator

    propagator = time_dependent_propagator(
        hamiltonian=hamiltonian,
        t0=0.0,
        t1=duration,
        steps=steps,
    )

    assert np.allclose(
        propagator.full(),
        x2.full(),
        atol=1e-2,
    )


def test_finite_joas_sequence_converges_to_sqrt_zz():
    """The finite-pulse Joas gate approaches sqrt(ZZ) for short pulses."""
    delta_1 = 0.31
    delta_2 = -0.17
    g = 0.08

    tau_2 = np.pi / (4 * g)
    tau_1 = 2.5 * tau_2

    pulse_duration = 1e-3
    steps = 500

    free = reduced_free_hamiltonian(
        delta_1=delta_1,
        delta_2=delta_2,
        coupling=g,
    )

    durations = two_pulse_gate_durations(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    # Joas reduced-basis convention.
    x1_operator = tensor(qeye(2), sigmax())
    x2_operator = tensor(sigmax(), qeye(2))

    peak_amplitude = sine_pi_pulse_amplitude(
        pulse_duration
    )

    def envelope(time):
        return sine_envelope(
            time=time,
            duration=pulse_duration,
            peak_amplitude=peak_amplitude,
        )

    pulse_1 = finite_pulse_propagator(
        free_hamiltonian=free,
        control_operator=0.5 * x1_operator,
        duration=pulse_duration,
        envelope=envelope,
        steps=steps,
    )

    pulse_2 = finite_pulse_propagator(
        free_hamiltonian=free,
        control_operator=0.5 * x2_operator,
        duration=pulse_duration,
        envelope=envelope,
        steps=steps,
    )

    propagator = finite_pulse_sequence_propagator(
        free_hamiltonian=free,
        durations=durations,
        pulse_propagators=[
            pulse_1,
            pulse_2,
            pulse_1,
            pulse_2,
        ],
    )

    matrix = propagator.full()

    # Remove global phase.
    matrix *= np.exp(
        -1j * np.angle(matrix[0, 0])
    )

    expected = sqrt_zz_gate().full()

    assert np.allclose(
        matrix,
        expected,
        atol=1e-2,
    )


def test_joas_sequence_timing():
    """The Joas sequence timing functions return the expected values."""
    tau_1 = 3000e-9
    tau_2 = 200e-9
    n_pi = 8

    assert np.isclose(
        interaction_time(tau_2, n_pi),  
        1600e-9,
    )

    assert np.isclose(
        sequence_duration(tau_1, n_pi),
        24e-6,
    )


def test_xy8_gate_times_rejects_invalid_tau_2():
    """The XY8 gate times reject tau_2 > tau_1 / 2."""
    with pytest.raises(ValueError):
        xy8_gate_times(
            tau_1=800e-9,
            tau_2=500e-9,
        )


def test_xy8_gate_schedule_contains_sixteen_pulses():
    """The XY8-1 Joas gate schedule contains 16 ideal pi pulses."""
    schedule = xy8_gate_schedule(
        tau_1=800e-9,
        tau_2=200e-9,
    )

    assert len(schedule) == 16


def test_xy8_gate_schedule_is_chronological():
    """The XY8 gate schedule is chronological for valid tau_1 and tau_2."""
    schedule = xy8_gate_schedule(
        tau_1=800e-9,
        tau_2=200e-9,
    )

    times = [event[0] for event in schedule]

    assert all(
        t_next > t
        for t, t_next in zip(times, times[1:])
    )


def test_xy8_gate_schedule_preserves_xy8_cycles():
    """The XY8 gate schedule preserves the XY8-1 pulse cycles on both NVs."""
    schedule = xy8_gate_schedule(
        tau_1=800e-9,
        tau_2=200e-9,
    )

    expected_phases = ["x", "y", "x", "y", "y", "x", "y", "x"]

    nv1_phases = [
        phase for _, nv, phase in schedule if nv == 1
    ]

    nv2_phases = [
        phase for _, nv, phase in schedule if nv == 2
    ]

    assert nv1_phases == expected_phases
    assert nv2_phases == expected_phases


def test_xy8_gate_intervals_contains_seventeen_intervals():
    """The XY8 gate intervals contain 17 free-evolution periods."""
    intervals = xy8_gate_intervals(
        tau_1=800e-9,
        tau_2=200e-9,
    )

    assert len(intervals) == 17


def test_xy8_gate_intervals_sum_to_gate_duration():
    """The XY8 gate intervals sum to the total gate duration."""
    tau_1 = 800e-9

    intervals = xy8_gate_intervals(
        tau_1=tau_1,
        tau_2=200e-9,
    )

    assert np.isclose(
        sum(intervals),
        8 * tau_1,
    )


def test_xy8_gate_intervals_are_positive():
    """The XY8 gate intervals are positive for valid tau_1 and tau_2."""
    intervals = xy8_gate_intervals(
        tau_1=800e-9,
        tau_2=200e-9,
    )

    assert all(interval > 0 for interval in intervals)


def test_xy8_gate_intervals_tau_2_zero():
    """The XY8 gate intervals are positive when tau_2 is zero."""
    tau_1 = 800e-9

    intervals = xy8_gate_intervals(
        tau_1=tau_1,
        tau_2=0.0,
    )

    assert np.isclose(sum(intervals), 8 * tau_1)


def test_xy8_gate_pulses_contains_sixteen_pulses():
    """The XY8-1 Joas gate contains 16 ideal pi pulses."""
    pulses = xy8_gate_pulses(
        tau_1=800e-9,
        tau_2=200e-9,
    )

    assert len(pulses) == 16


def test_xy8_gate_accumulates_expected_conditional_phase():
    """The ideal XY8-1 Joas sequence accumulates the expected conditional phase."""
    delta_1 = 0.31
    delta_2 = -0.17
    coupling = 0.08

    tau_1 = 1.0
    tau_2 = 0.2

    free = reduced_free_hamiltonian(
        delta_1=delta_1,
        delta_2=delta_2,
        coupling=coupling,
    )

    intervals = xy8_gate_intervals(tau_1, tau_2)
    pulses = xy8_gate_pulses(tau_1, tau_2)

    propagator = finite_pulse_sequence_propagator(
        free_hamiltonian=free,
        durations=intervals,
        pulse_propagators=pulses,
    )

    phase = 8 * coupling * tau_2

    expected = Qobj(
        np.diag([
            1.0,
            np.exp(1j * phase),
            np.exp(1j * phase),
            1.0,
        ]),
        dims=[[2, 2], [2, 2]],
    )

    # Remove global phase before comparison.
    global_phase = propagator[0, 0] / expected[0, 0]

    assert (
        propagator - global_phase * expected
    ).norm() < 1e-10


def test_xy8_gate_times_follow_joas_timing():
    """The XY8 gate times follow the Joas two-pulse timing convention."""
    tau_1 = 800e-9
    tau_2 = 200e-9

    nv1_times, nv2_times = xy8_gate_times(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    assert len(nv1_times) == 8
    assert len(nv2_times) == 8

    for k in range(8):
        assert np.isclose(
            nv1_times[k],
            (k + 0.5) * tau_1,
        )

        assert np.isclose(
            nv2_times[k],
            (k + 1.0) * tau_1 - tau_2,
        )


def test_xy8_joas_sequence_generates_sqrt_zz():
    """The ideal XY8-1 Joas sequence generates sqrt(ZZ)."""
    delta_1 = 0.31
    delta_2 = -0.17

    nu_dip = 0.08
    coupling = 2 * np.pi * nu_dip

    n_pi = 8

    tau_2 = 1 / (4 * n_pi * nu_dip)

    # Must satisfy tau_2 <= tau_1 / 2.
    tau_1 = 2.5 * tau_2

    free = reduced_free_hamiltonian(
        delta_1=delta_1,
        delta_2=delta_2,
        coupling=coupling,
    )

    intervals = xy8_gate_intervals(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    pulses = xy8_gate_pulses(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    propagator = finite_pulse_sequence_propagator(
        free_hamiltonian=free,
        durations=intervals,
        pulse_propagators=pulses,
    )

    matrix = propagator.full()

    # Remove global phase.
    matrix *= np.exp(
        -1j * np.angle(matrix[0, 0])
    )

    assert np.allclose(
        matrix,
        sqrt_zz_gate().full(),
    )


def test_finite_xy8_joas_gate_converges_to_sqrt_zz():
    """Finite sine pulses approach the ideal XY8-1 sqrt(ZZ) gate."""
    delta_1 = 0.31
    delta_2 = -0.17

    nu_dip = 0.08
    coupling = 2 * np.pi * nu_dip

    n_pi = 8

    tau_2 = 1 / (4 * n_pi * nu_dip)
    tau_1 = 2.5 * tau_2

    pulse_duration = 1e-3
    steps = 500

    free = reduced_free_hamiltonian(
        delta_1=delta_1,
        delta_2=delta_2,
        coupling=coupling,
    )

    intervals = xy8_gate_intervals(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    schedule = xy8_gate_schedule(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    peak_amplitude = sine_pi_pulse_amplitude(
        pulse_duration
    )

    def envelope(time):
        return sine_envelope(
            time=time,
            duration=pulse_duration,
            peak_amplitude=peak_amplitude,
        )

    pulse_propagators = []

    for _, nv, phase in schedule:
        if nv == 1:
            qubit = 1
        else:
            qubit = 0

        if phase == "x":
            pauli = sigmax()
        else:
            pauli = sigmay()

        if qubit == 0:
            control = tensor(pauli, qeye(2))
        else:
            control = tensor(qeye(2), pauli)

        pulse = finite_pulse_propagator(
            free_hamiltonian=free,
            control_operator=0.5 * control,
            duration=pulse_duration,
            envelope=envelope,
            steps=steps,
        )

        pulse_propagators.append(pulse)

    propagator = finite_pulse_sequence_propagator(
        free_hamiltonian=free,
        durations=intervals,
        pulse_propagators=pulse_propagators,
    )

    matrix = propagator.full()

    # Remove global phase.
    matrix *= np.exp(
        -1j * np.angle(matrix[0, 0])
    )

    assert np.allclose(
        matrix,
        sqrt_zz_gate().full(),
        atol=1e-2,
    )


def test_two_electron_hamiltonian_has_dimension_nine():
    """The two-electron Hamiltonian is a 9x9 operator."""
    h_nv1 = Qobj(np.diag([1.0, 0.0, 2.0]))
    h_nv2 = Qobj(np.diag([3.0, 0.0, 4.0]))

    interaction = 0.1 * qeye([3, 3])

    hamiltonian = two_electron_hamiltonian(
        h_nv1=h_nv1,
        h_nv2=h_nv2,
        interaction=interaction,
    )

    assert hamiltonian.shape == (9, 9)
    assert hamiltonian.dims == [[3, 3], [3, 3]]


def test_two_electron_hamiltonian_sums_local_terms():
    """The two-electron Hamiltonian should sum the local NV terms."""
    h_nv1 = Qobj(np.diag([1.0, 0.0, 2.0]))
    h_nv2 = Qobj(np.diag([3.0, 0.0, 4.0]))

    interaction = 0.1 * qeye([3, 3])

    actual = two_electron_hamiltonian(
        h_nv1=h_nv1,
        h_nv2=h_nv2,
        interaction=interaction,
    )

    expected = (
        tensor(h_nv1, qeye(3))
        + tensor(qeye(3), h_nv2)
        + interaction
    )

    assert np.allclose(
        actual.full(),
        expected.full(),
    )


def test_two_electron_logical_basis():
    """The two-electron logical basis is the tensor product of the NV logical states."""
    nv1_0 = basis(3, 1)
    nv1_1 = basis(3, 0)

    nv2_0 = basis(3, 1)
    nv2_1 = basis(3, 0)

    logical_basis = two_electron_logical_basis(
        nv1_states=(nv1_0, nv1_1),
        nv2_states=(nv2_0, nv2_1),
    )

    expected = [
        tensor(nv1_0, nv2_0),
        tensor(nv1_0, nv2_1),
        tensor(nv1_1, nv2_0),
        tensor(nv1_1, nv2_1),
    ]

    assert len(logical_basis) == 4

    for actual, target in zip(logical_basis, expected):
        assert np.allclose(
            actual.full(),
            target.full(),
        )


def test_two_electron_hamiltonian_projects_to_logical_subspace():
    """The two-electron Hamiltonian projects to a 4x4 operator in the logical subspace."""
    h_nv1 = Qobj(np.diag([1.0, 0.0, 2.0]))
    h_nv2 = Qobj(np.diag([3.0, 0.0, 4.0]))

    interaction = 0.1 * qeye([3, 3])

    full = two_electron_hamiltonian(
        h_nv1=h_nv1,
        h_nv2=h_nv2,
        interaction=interaction,
    )

    logical_basis = two_electron_logical_basis(
        nv1_states=(basis(3, 1), basis(3, 0)),
        nv2_states=(basis(3, 1), basis(3, 0)),
    )

    projected = project_operator(
        operator=full,
        basis_states=logical_basis,
    )

    assert projected.shape == (4, 4)


def test_electron_logical_states_axial_field():
    """The electron logical states are the m_s = 0 and m_s = -1 states for an axial field."""
    system = SpinSystem([1])

    hamiltonian = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2.87,
        omega_e=[0.0, 0.0, 0.10],
    )

    zero, one = electron_logical_states(hamiltonian)

    expected_zero = basis(3, 1)
    expected_one = basis(3, 0)

    assert abs(expected_zero.overlap(zero)) ** 2 == pytest.approx(1.0)
    assert abs(expected_one.overlap(one)) ** 2 == pytest.approx(1.0)


def test_electron_logical_states_tilted_field_are_orthonormal():
    """The electron logical states are orthonormal for a tilted field."""
    system = SpinSystem([1])

    hamiltonian = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2.87,
        omega_e=[0.05, 0.0, 0.10],
    )

    zero, one = electron_logical_states(hamiltonian)

    assert zero.norm() == pytest.approx(1.0)
    assert one.norm() == pytest.approx(1.0)
    assert abs(zero.overlap(one)) == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_joas_nv1_local_frame_rotation():
    """The Joas NV1 local frame rotation is consistent with the rotation matrix."""
    beta = np.deg2rad(70.53)

    vector = np.array([1.0, 2.0, 3.0])

    actual = rotate_to_local_frame(
        vector=vector,
        axis="y",
        angle=beta,
    )

    expected = np.array([
        np.cos(beta) * vector[0] - np.sin(beta) * vector[2],
        vector[1],
        np.sin(beta) * vector[0] + np.cos(beta) * vector[2],
    ])

    assert np.allclose(actual, expected)


def test_joas_setting_2_two_electron_spectrum():
    """The two-electron Hamiltonian spectrum matches the Joas setting 2 spectrum."""
    system = SpinSystem([1])

    omega = 2 * np.pi * 295.18

    # NV1 = center A = target, strongly misaligned
    theta_nv1 = np.deg2rad(74.08)
    omega_nv1 = omega * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv1,
    )

    # NV2 = center B = control, approximately aligned
    theta_nv2 = np.deg2rad(3.58)
    omega_nv2 = omega * np.array([
        np.sin(theta_nv2),
        0.0,
        np.cos(theta_nv2),
    ])

    h_nv2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv2,
    )

    interaction = Qobj(
        np.zeros((9, 9)),
        dims=[[3, 3], [3, 3]],
    )

    h_two = two_electron_hamiltonian(
        h_nv1=h_nv1,
        h_nv2=h_nv2,
        interaction=interaction,
    )

    energies_nv1 = np.sort(h_nv1.eigenenergies())
    energies_nv2 = np.sort(h_nv2.eigenenergies())

    expected = np.sort(
        np.array([
            e1 + e2
            for e1 in energies_nv1
            for e2 in energies_nv2
        ])
    )

    actual = np.sort(h_two.eigenenergies())

    assert h_two.shape == (9, 9)
    assert np.allclose(actual, expected)

    # Ground-state energy of the two-electron system
    ground = actual[0]

    # Excite NV1 while NV2 remains in its ground state
    nv1_transitions = (
        energies_nv1[1:] - energies_nv1[0]
    ) / (2 * np.pi)

    # Excite NV2 while NV1 remains in its ground state
    nv2_transitions = (
        energies_nv2[1:] - energies_nv2[0]
    ) / (2 * np.pi)

    assert np.allclose(
        nv1_transitions,
        [2825.45481206, 2988.99810054],
        atol=1e-6,
    )

    assert np.allclose(
        nv2_transitions,
        [2572.85177892, 3162.04745120],
        atol=1e-6,
    )


def test_joas_setting_2_dressed_spin_operators():
    """The dressed spin operators are consistent with the spin-one operators."""
    system = SpinSystem([1])

    omega = 2 * np.pi * 295.18

    # NV1 = A = target, misaligned
    theta_nv1 = np.deg2rad(74.08)
    omega_nv1 = omega * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv1,
    )

    # NV2 = B = control, aligned
    theta_nv2 = np.deg2rad(3.58)
    omega_nv2 = omega * np.array([
        np.sin(theta_nv2),
        0.0,
        np.cos(theta_nv2),
    ])

    h_nv2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv2,
    )

    basis_nv1 = dressed_spin_one_basis(h_nv1)
    basis_nv2 = dressed_spin_one_basis(h_nv2)

    _, _, sz_nv1 = dressed_spin_operators(
        basis_states=basis_nv1,
        spin=1,
    )

    _, _, sz_nv2 = dressed_spin_operators(
        basis_states=basis_nv2,
        spin=1,
    )

    _, _, sz = spin_operators(1)

    assert np.allclose(
        operator_in_basis(sz_nv1, basis_nv1).full(),
        sz.full(),
        atol=1e-12,
    )

    assert np.allclose(
        operator_in_basis(sz_nv2, basis_nv2).full(),
        sz.full(),
        atol=1e-12,
    )


def test_joas_setting_2_logical_effective_interaction():
    """The two-electron interaction projects to the expected logical subspace operator."""
    system = SpinSystem([1])

    omega = 2 * np.pi * 295.18

    # NV1 = A = target, misaligned
    theta_nv1 = np.deg2rad(74.08)
    omega_nv1 = omega * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv1,
    )

    # NV2 = B = control, aligned
    theta_nv2 = np.deg2rad(3.58)
    omega_nv2 = omega * np.array([
        np.sin(theta_nv2),
        0.0,
        np.cos(theta_nv2),
    ])

    h_nv2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv2,
    )

    basis_nv1 = dressed_spin_one_basis(h_nv1)
    basis_nv2 = dressed_spin_one_basis(h_nv2)

    _, _, sz_nv1 = dressed_spin_operators(
        basis_states=basis_nv1,
        spin=1,
    )

    _, _, sz_nv2 = dressed_spin_operators(
        basis_states=basis_nv2,
        spin=1,
    )

    coupling = 2 * np.pi * 0.11261

    interaction = coupling * tensor(
        sz_nv1,
        sz_nv2,
    )

    # Joas setting 2:
    # NV1 addresses the +1-like dressed state.
    # NV2 addresses the -1-like dressed state.
    logical_nv1 = electron_logical_states(
        h_nv1,
        excited_state="+1",
    )

    logical_nv2 = electron_logical_states(
        h_nv2,
        excited_state="-1",
    )

    logical_basis = two_electron_logical_basis(
        nv1_states=logical_nv1,
        nv2_states=logical_nv2,
    )

    logical_interaction = project_operator(
        operator=interaction,
        basis_states=logical_basis,
    )

    expected = Qobj(
        np.diag([
            0.0,
            0.0,
            0.0,
            -coupling,
        ])
    )

    assert np.allclose(
        logical_interaction.full(),
        expected.full(),
        atol=1e-12,
    )


def test_joas_setting_2_physical_to_logical_hamiltonian():
    """The two-electron Hamiltonian projects to the expected logical subspace operator."""
    system = SpinSystem([1])

    omega = 2 * np.pi * 295.18

    # NV1 = A = target, misaligned
    theta_nv1 = np.deg2rad(74.08)
    omega_nv1 = omega * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv1,
    )

    # NV2 = B = control, aligned
    theta_nv2 = np.deg2rad(3.58)
    omega_nv2 = omega * np.array([
        np.sin(theta_nv2),
        0.0,
        np.cos(theta_nv2),
    ])

    h_nv2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv2,
    )

    basis_nv1 = dressed_spin_one_basis(h_nv1)
    basis_nv2 = dressed_spin_one_basis(h_nv2)

    _, _, sz_nv1 = dressed_spin_operators(
        basis_states=basis_nv1,
        spin=1,
    )

    _, _, sz_nv2 = dressed_spin_operators(
        basis_states=basis_nv2,
        spin=1,
    )

    coupling = 2 * np.pi * 0.11261

    interaction = coupling * tensor(
        sz_nv1,
        sz_nv2,
    )

    h_two = two_electron_hamiltonian(
        h_nv1=h_nv1,
        h_nv2=h_nv2,
        interaction=interaction,
    )

    # Actual addressed transitions in Joas setting 2.
    logical_nv1 = electron_logical_states(
        h_nv1,
        excited_state="+1",
    )

    logical_nv2 = electron_logical_states(
        h_nv2,
        excited_state="-1",
    )

    logical_basis = two_electron_logical_basis(
        nv1_states=logical_nv1,
        nv2_states=logical_nv2,
    )

    h_logical = project_operator(
        operator=h_two,
        basis_states=logical_basis,
    )

    e0_nv1 = (
        logical_nv1[0].dag()
        * h_nv1
        * logical_nv1[0]
    )

    e1_nv1 = (
        logical_nv1[1].dag()
        * h_nv1
        * logical_nv1[1]
    )

    e0_nv2 = (
        logical_nv2[0].dag()
        * h_nv2
        * logical_nv2[0]
    )

    e1_nv2 = (
        logical_nv2[1].dag()
        * h_nv2
        * logical_nv2[1]
    )

    expected = Qobj(
        np.diag([
            e0_nv1 + e0_nv2,
            e0_nv1 + e1_nv2,
            e1_nv1 + e0_nv2,
            e1_nv1 + e1_nv2 - coupling,
        ])
    )

    assert np.allclose(
        h_logical.full(),
        expected.full(),
        atol=1e-10,
    )


def test_electron_logical_states_select_excited_branch():
    """The electron logical states select the correct excited branch."""
    system = SpinSystem([1])

    hamiltonian = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2.87,
        omega_e=[0.0, 0.0, 0.10],
    )

    plus_one, zero, minus_one = dressed_spin_one_basis(
        hamiltonian
    )

    logical_zero_plus, logical_one_plus = electron_logical_states(
        hamiltonian,
        excited_state="+1",
    )

    logical_zero_minus, logical_one_minus = electron_logical_states(
        hamiltonian,
        excited_state="-1",
    )

    assert abs(logical_zero_plus.overlap(zero)) == pytest.approx(1.0)
    assert abs(logical_one_plus.overlap(plus_one)) == pytest.approx(1.0)

    assert abs(logical_zero_minus.overlap(zero)) == pytest.approx(1.0)
    assert abs(logical_one_minus.overlap(minus_one)) == pytest.approx(1.0)


def test_electron_logical_states_invalid_excited_branch():
    """The electron logical states raise an error for an invalid excited branch."""
    system = SpinSystem([1])

    hamiltonian = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2.87,
        omega_e=[0.0, 0.0, 0.10],
    )

    with pytest.raises(ValueError):
        electron_logical_states(
            hamiltonian,
            excited_state="banana",
        )


def test_joas_nv1_physical_resonant_pi_pulse():
    """A physical resonant pi pulse on NV1 rotates the logical states."""
    system = SpinSystem([1])

    omega = 2 * np.pi * 295.18
    theta_nv1 = np.deg2rad(74.08)

    omega_nv1 = omega * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv1,
    )

    logical_zero, logical_one = electron_logical_states(
        h_nv1,
        excited_state="+1",
    )

    # Physical spin-1 transverse microwave operator.
    sx, _, _ = spin_operators(1)

    # Resonance frequency of the addressed dressed transition.
    e0 = logical_zero.dag() * h_nv1 * logical_zero
    e1 = logical_one.dag() * h_nv1 * logical_one
    drive_frequency = float(np.real(e1 - e0))

    # Matrix element of the physical Sx operator on the addressed transition.
    matrix_element = abs(
        logical_one.dag() * sx * logical_zero
    )

    pulse_duration = 0.1

    # For H_drive = amplitude*cos(omega*t)*Sx,
    # RWA gives an effective coupling
    # amplitude*matrix_element/2.
    # A pi pulse therefore requires
    # amplitude*matrix_element*pulse_duration = pi.
    amplitude = np.pi / (
        matrix_element * pulse_duration
    )

    def hamiltonian(time):
        return (
            h_nv1
            + amplitude
            * np.cos(drive_frequency * time)
            * sx
        )

    propagator = time_dependent_propagator(
        hamiltonian=hamiltonian,
        t0=0.0,
        t1=pulse_duration,
        steps=5000,
    )

    final_state = propagator * logical_zero

    population_one = abs(
        logical_one.overlap(final_state)
    ) ** 2

    assert population_one > 0.99


def test_joas_nv2_physical_resonant_pi_pulse():
    """A physical resonant pi pulse on NV2 rotates the logical states."""
    system = SpinSystem([1])

    omega = 2 * np.pi * 295.18
    theta_nv2 = np.deg2rad(3.58)

    omega_nv2 = omega * np.array([
        np.sin(theta_nv2),
        0.0,
        np.cos(theta_nv2),
    ])

    h_nv2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv2,
    )

    logical_zero, logical_one = electron_logical_states(
        h_nv2,
        excited_state="-1",
    )

    # Physical spin-1 transverse microwave operator.
    sx, _, _ = spin_operators(1)

    # Resonance frequency of the addressed dressed transition.
    e0 = logical_zero.dag() * h_nv2 * logical_zero
    e1 = logical_one.dag() * h_nv2 * logical_one
    drive_frequency = float(np.real(e1 - e0))

    # Physical transition matrix element.
    matrix_element = abs(
        logical_one.dag() * sx * logical_zero
    )

    pulse_duration = 0.1

    # Calibrate a pi pulse in the rotating-wave approximation.
    amplitude = np.pi / (
        matrix_element * pulse_duration
    )

    def hamiltonian(time):
        return (
            h_nv2
            + amplitude
            * np.cos(drive_frequency * time)
            * sx
        )

    propagator = time_dependent_propagator(
        hamiltonian=hamiltonian,
        t0=0.0,
        t1=pulse_duration,
        steps=5000,
    )

    final_state = propagator * logical_zero

    population_one = abs(
        logical_one.overlap(final_state)
    ) ** 2

    assert population_one > 0.99


def test_joas_two_electron_nv1_physical_pi_pulse():
    """The physical NV1 pulse survives embedding into the interacting \(9D\) system."""
    system = SpinSystem([1])

    omega = 2 * np.pi * 295.18

    # --- Single-NV Hamiltonians ---

    theta_nv1 = np.deg2rad(74.08)
    omega_nv1 = omega * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv1,
    )

    theta_nv2 = np.deg2rad(3.58)
    omega_nv2 = omega * np.array([
        np.sin(theta_nv2),
        0.0,
        np.cos(theta_nv2),
    ])

    h_nv2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv2,
    )

    # --- Dressed operators ---

    basis_nv1 = dressed_spin_one_basis(h_nv1)
    basis_nv2 = dressed_spin_one_basis(h_nv2)

    _, _, sz_nv1 = dressed_spin_operators(
        basis_states=basis_nv1,
        spin=1,
    )

    _, _, sz_nv2 = dressed_spin_operators(
        basis_states=basis_nv2,
        spin=1,
    )

    coupling = 2 * np.pi * 0.11261

    interaction = coupling * tensor(
        sz_nv1,
        sz_nv2,
    )

    h_two = two_electron_hamiltonian(
        h_nv1=h_nv1,
        h_nv2=h_nv2,
        interaction=interaction,
    )

    # --- Logical states ---

    zero_nv1, one_nv1 = electron_logical_states(
        h_nv1,
        excited_state="+1",
    )

    zero_nv2, _ = electron_logical_states(
        h_nv2,
        excited_state="-1",
    )

    initial_state = tensor(
        zero_nv1,
        zero_nv2,
    )

    target_state = tensor(
        one_nv1,
        zero_nv2,
    )

    # --- Physical NV1 microwave drive ---

    sx, _, _ = spin_operators(1)

    sx_nv1 = tensor(
        sx,
        qeye(3),
    )

    e0_nv1 = zero_nv1.dag() * h_nv1 * zero_nv1
    e1_nv1 = one_nv1.dag() * h_nv1 * one_nv1

    drive_frequency = float(
        np.real(e1_nv1 - e0_nv1)
    )

    matrix_element = abs(
        one_nv1.dag() * sx * zero_nv1
    )

    pulse_duration = 0.1

    amplitude = np.pi / (
        matrix_element * pulse_duration
    )

    def hamiltonian(time):
        return (
            h_two
            + amplitude
            * np.cos(drive_frequency * time)
            * sx_nv1
        )

    propagator = time_dependent_propagator(
        hamiltonian=hamiltonian,
        t0=0.0,
        t1=pulse_duration,
        steps=5000,
    )

    final_state = propagator * initial_state

    target_population = abs(
        target_state.overlap(final_state)
    ) ** 2

    assert target_population > 0.99


def test_joas_nv1_conditional_transition_shift():
    """The NV1 transition frequency shifts in the presence of NV2 in the logical |1> state."""
    system = SpinSystem([1])

    omega = 2 * np.pi * 295.18

    # --- Single-NV Hamiltonians ---

    theta_nv1 = np.deg2rad(74.08)
    omega_nv1 = omega * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv1,
    )

    theta_nv2 = np.deg2rad(3.58)
    omega_nv2 = omega * np.array([
        np.sin(theta_nv2),
        0.0,
        np.cos(theta_nv2),
    ])

    h_nv2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv2,
    )

    # --- Effective dressed interaction ---

    basis_nv1 = dressed_spin_one_basis(h_nv1)
    basis_nv2 = dressed_spin_one_basis(h_nv2)

    _, _, sz_nv1 = dressed_spin_operators(
        basis_states=basis_nv1,
        spin=1,
    )

    _, _, sz_nv2 = dressed_spin_operators(
        basis_states=basis_nv2,
        spin=1,
    )

    coupling = 2 * np.pi * 0.11261

    interaction = coupling * tensor(
        sz_nv1,
        sz_nv2,
    )

    h_two = two_electron_hamiltonian(
        h_nv1=h_nv1,
        h_nv2=h_nv2,
        interaction=interaction,
    )

    # --- Actual Joas logical states ---

    zero_nv1, one_nv1 = electron_logical_states(
        h_nv1,
        excited_state="+1",
    )

    _, one_nv2 = electron_logical_states(
        h_nv2,
        excited_state="-1",
    )

    state_01 = tensor(
        zero_nv1,
        one_nv2,
    )

    state_11 = tensor(
        one_nv1,
        one_nv2,
    )

    # Transition frequency with NV2 in |1>.
    energy_01 = state_01.dag() * h_two * state_01
    energy_11 = state_11.dag() * h_two * state_11

    conditional_frequency = float(
        np.real(energy_11 - energy_01)
    )

    # Bare NV1 logical transition.
    energy_0_nv1 = zero_nv1.dag() * h_nv1 * zero_nv1
    energy_1_nv1 = one_nv1.dag() * h_nv1 * one_nv1

    bare_frequency = float(
        np.real(energy_1_nv1 - energy_0_nv1)
    )

    assert conditional_frequency == pytest.approx(
        bare_frequency - coupling,
        abs=1e-10,
    )


def test_joas_two_electron_nv2_physical_pi_pulse():
    """The physical NV2 pulse survives embedding into the interacting \(9D\) system."""
    system = SpinSystem([1])

    omega = 2 * np.pi * 295.18

    # --- Single-NV Hamiltonians ---

    theta_nv1 = np.deg2rad(74.08)
    omega_nv1 = omega * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv1,
    )

    theta_nv2 = np.deg2rad(3.58)
    omega_nv2 = omega * np.array([
        np.sin(theta_nv2),
        0.0,
        np.cos(theta_nv2),
    ])

    h_nv2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv2,
    )

    # --- Dressed interaction ---

    basis_nv1 = dressed_spin_one_basis(h_nv1)
    basis_nv2 = dressed_spin_one_basis(h_nv2)

    _, _, sz_nv1 = dressed_spin_operators(
        basis_states=basis_nv1,
        spin=1,
    )

    _, _, sz_nv2 = dressed_spin_operators(
        basis_states=basis_nv2,
        spin=1,
    )

    coupling = 2 * np.pi * 0.11261

    interaction = coupling * tensor(
        sz_nv1,
        sz_nv2,
    )

    h_two = two_electron_hamiltonian(
        h_nv1=h_nv1,
        h_nv2=h_nv2,
        interaction=interaction,
    )

    # --- Logical states ---

    zero_nv1, _ = electron_logical_states(
        h_nv1,
        excited_state="+1",
    )

    zero_nv2, one_nv2 = electron_logical_states(
        h_nv2,
        excited_state="-1",
    )

    initial_state = tensor(
        zero_nv1,
        zero_nv2,
    )

    target_state = tensor(
        zero_nv1,
        one_nv2,
    )

    # --- Physical NV2 microwave drive ---

    sx, _, _ = spin_operators(1)

    sx_nv2 = tensor(
        qeye(3),
        sx,
    )

    e0_nv2 = zero_nv2.dag() * h_nv2 * zero_nv2
    e1_nv2 = one_nv2.dag() * h_nv2 * one_nv2

    drive_frequency = float(
        np.real(e1_nv2 - e0_nv2)
    )

    matrix_element = abs(
        one_nv2.dag() * sx * zero_nv2
    )

    pulse_duration = 0.1

    amplitude = np.pi / (
        matrix_element * pulse_duration
    )

    def hamiltonian(time):
        return (
            h_two
            + amplitude
            * np.cos(drive_frequency * time)
            * sx_nv2
        )

    propagator = time_dependent_propagator(
        hamiltonian=hamiltonian,
        t0=0.0,
        t1=pulse_duration,
        steps=5000,
    )

    final_state = propagator * initial_state

    target_population = abs(
        target_state.overlap(final_state)
    ) ** 2

    assert target_population > 0.99


def test_joas_selective_dressed_pi_rotations():
    """The selective dressed pi rotations swap the addressed logical states."""
    system = SpinSystem([1])

    omega = 2 * np.pi * 295.18

    # NV1 = target, +1-like addressed branch
    theta_nv1 = np.deg2rad(74.08)
    omega_nv1 = omega * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv1,
    )

    # NV2 = control, -1-like addressed branch
    theta_nv2 = np.deg2rad(3.58)
    omega_nv2 = omega * np.array([
        np.sin(theta_nv2),
        0.0,
        np.cos(theta_nv2),
    ])

    h_nv2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv2,
    )

    zero_nv1, one_nv1 = electron_logical_states(
        h_nv1,
        excited_state="+1",
    )

    zero_nv2, one_nv2 = electron_logical_states(
        h_nv2,
        excited_state="-1",
    )

    plus_nv1, _, minus_nv1 = dressed_spin_one_basis(h_nv1)
    plus_nv2, _, minus_nv2 = dressed_spin_one_basis(h_nv2)

    # Unused dressed level for each addressed transition.
    spectator_nv1 = minus_nv1
    spectator_nv2 = plus_nv2

    x_nv1 = selective_rotation(
        state_0=zero_nv1,
        state_1=one_nv1,
        angle=np.pi,
        axis="x",
    )

    x_nv2 = selective_rotation(
        state_0=zero_nv2,
        state_1=one_nv2,
        angle=np.pi,
        axis="x",
    )

    # Addressed states are swapped.
    assert abs(
        one_nv1.overlap(x_nv1 * zero_nv1)
    ) == pytest.approx(1.0)

    assert abs(
        one_nv2.overlap(x_nv2 * zero_nv2)
    ) == pytest.approx(1.0)

    # Unused spin-1 levels are spectators.
    assert abs(
        spectator_nv1.overlap(x_nv1 * spectator_nv1)
    ) == pytest.approx(1.0)

    assert abs(
        spectator_nv2.overlap(x_nv2 * spectator_nv2)
    ) == pytest.approx(1.0)


def test_joas_two_electron_ideal_xy8_gate():
    """The physical 9D XY8 sequence implements sqrt(ZZ) or its inverse."""

    system = SpinSystem([1])

    omega = 2 * np.pi * 295.18

    # --- Physical single-NV Hamiltonians ---

    theta_nv1 = np.deg2rad(74.08)
    omega_nv1 = omega * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv1,
    )

    theta_nv2 = np.deg2rad(3.58)
    omega_nv2 = omega * np.array([
        np.sin(theta_nv2),
        0.0,
        np.cos(theta_nv2),
    ])

    h_nv2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv2,
    )

    # --- Dressed bases and effective interaction ---

    basis_nv1 = dressed_spin_one_basis(h_nv1)
    basis_nv2 = dressed_spin_one_basis(h_nv2)

    _, _, sz_nv1 = dressed_spin_operators(
        basis_states=basis_nv1,
        spin=1,
    )

    _, _, sz_nv2 = dressed_spin_operators(
        basis_states=basis_nv2,
        spin=1,
    )

    nu_dip = 0.11261
    coupling = 2 * np.pi * nu_dip

    interaction = coupling * tensor(
        sz_nv1,
        sz_nv2,
    )

    h_two = two_electron_hamiltonian(
        h_nv1=h_nv1,
        h_nv2=h_nv2,
        interaction=interaction,
    )

    # --- Actual logical states used in Joas setting 2 ---

    zero_nv1, one_nv1 = electron_logical_states(
        h_nv1,
        excited_state="+1",
    )

    zero_nv2, one_nv2 = electron_logical_states(
        h_nv2,
        excited_state="-1",
    )

    logical_basis = two_electron_logical_basis(
        nv1_states=(zero_nv1, one_nv1),
        nv2_states=(zero_nv2, one_nv2),
    )

    # --- Test both signs of the staggered pulse offset ---

    n_pi = 8

    for tau_2_sign, target in [
        (-1, sqrt_zz_gate()),
        (+1, sqrt_zz_gate().dag()),
    ]:
        tau_2 = tau_2_sign / (
            4 * n_pi * nu_dip
        )

        tau_1 = 2.5 * abs(tau_2)

        schedule = xy8_gate_schedule(
            tau_1=tau_1,
            tau_2=tau_2,
        )

        durations = xy8_gate_intervals(
            tau_1=tau_1,
            tau_2=tau_2,
        )

        # --- Physical selective spin-1 pulses ---

        identity = qeye(3)
        pulses = []

        for _, nv, phase in schedule:
            if nv == 1:
                local_pulse = selective_rotation(
                    state_0=zero_nv1,
                    state_1=one_nv1,
                    angle=np.pi,
                    axis=phase,
                )

                pulse = tensor(
                    local_pulse,
                    identity,
                )

            else:
                local_pulse = selective_rotation(
                    state_0=zero_nv2,
                    state_1=one_nv2,
                    angle=np.pi,
                    axis=phase,
                )

                pulse = tensor(
                    identity,
                    local_pulse,
                )

            pulses.append(pulse)

        # --- Ideal physical 9D XY8 evolution ---

        propagator = finite_pulse_sequence_propagator(
            free_hamiltonian=h_two,
            durations=durations,
            pulse_propagators=pulses,
        )

        # --- Projection onto the logical two-qubit subspace ---

        logical_propagator = project_operator(
            operator=propagator,
            basis_states=logical_basis,
        )

        # Remove the physically irrelevant global phase.
        phase = np.angle(
            logical_propagator.full()[0, 0]
        )

        logical_propagator = (
            np.exp(-1j * phase)
            * logical_propagator
        )

        assert np.allclose(
            logical_propagator.full(),
            target.full(),
            atol=1e-8,
        )


@pytest.mark.parametrize(
    "axis, phase",
    [
        ("x", 0.0),
        ("y", np.pi / 2),
    ],
)
def test_joas_nv1_finite_microwave_pi_pulse_phase(axis, phase):
    """A resonant finite MW pulse implements the expected dressed X/Y pi rotation."""

    system = SpinSystem([1])

    omega_e = 2 * np.pi * 295.18

    theta_nv1 = np.deg2rad(74.08)
    omega_nv1 = omega_e * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv1,
    )

    # --- Addressed dressed transition: |0> <-> |+1> ---

    zero_nv1, one_nv1 = electron_logical_states(
        h_nv1,
        excited_state="+1",
    )

    energy_zero = np.real(
        zero_nv1.dag() * h_nv1 * zero_nv1
    )

    energy_one = np.real(
        one_nv1.dag() * h_nv1 * one_nv1
    )

    transition_frequency = energy_one - energy_zero

    # --- Embed NV1 into the physical two-electron 9D space ---

    identity = qeye(3)

    h_two = tensor(h_nv1, identity)

    sx, _, _ = spin_operators(1)
    sx_nv1 = tensor(sx, identity)

    initial = tensor(
        zero_nv1,
        zero_nv1,
    )

    target = tensor(
        one_nv1,
        zero_nv1,
    )

    # --- Finite sine-shaped pi pulse ---

    pulse_duration = 0.1

    matrix_element = abs(
        one_nv1.dag() * sx * zero_nv1
    )

    # microwave_hamiltonian contributes a sqrt(2) factor.
    # For a sine envelope,
    #
    # integral_0^T sin(pi t / T) dt = 2T/pi.
    #
    # Matching the pi-pulse area of the validated resonant
    # rectangular drive gives:
    #
    # peak_amplitude = pi^2 / (2 sqrt(2) |<1|Sx|0>| T)
    peak_amplitude = (
        np.pi**2
        / (
            2
            * np.sqrt(2.0)
            * matrix_element
            * pulse_duration
        )
    )

    def hamiltonian(time):
        envelope = sine_envelope(
            time=time,
            duration=pulse_duration,
            peak_amplitude=peak_amplitude,
        )

        microwave = microwave_hamiltonian(
            time=time,
            control=sx_nv1,
            drives=[
                {
                    "omega": transition_frequency,
                    "phase": phase,
                    "omega_x": envelope,
                    "omega_y": 0.0,
                }
            ],
        )

        return h_two + microwave

    propagator = time_dependent_propagator(
        hamiltonian=hamiltonian,
        t0=0.0,
        t1=pulse_duration,
        steps=5000,
    )

    final = propagator * initial

    population = abs(
        target.overlap(final)
    ) ** 2

    assert population > 0.99


def test_joas_nv1_finite_microwave_carrier_phase():
    """A pi/2 carrier-phase shift changes the dressed rotation axis by pi/2."""

    system = SpinSystem([1])

    omega_e = 2 * np.pi * 295.18
    theta_nv1 = np.deg2rad(74.08)

    omega_nv1 = omega_e * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv1,
    )

    zero_nv1, one_nv1 = electron_logical_states(
        h_nv1,
        excited_state="+1",
    )

    sx, _, _ = spin_operators(1)

    energy_zero = np.real(
        zero_nv1.dag() * h_nv1 * zero_nv1
    )
    energy_one = np.real(
        one_nv1.dag() * h_nv1 * one_nv1
    )

    transition_frequency = energy_one - energy_zero

    matrix_element = abs(
        one_nv1.dag() * sx * zero_nv1
    )

    pulse_duration = 0.1

    peak_amplitude = (
        np.pi**2
        / (
            2
            * np.sqrt(2.0)
            * matrix_element
            * pulse_duration
        )
    )

    def pulse_propagator(phase):
        def hamiltonian(time):
            envelope = sine_envelope(
                time=time,
                duration=pulse_duration,
                peak_amplitude=peak_amplitude,
            )

            microwave = microwave_hamiltonian(
                time=time,
                control=sx,
                drives=[
                    {
                        "omega": transition_frequency,
                        "phase": phase,
                        "omega_x": envelope,
                        "omega_y": 0.0,
                    }
                ],
            )

            return h_nv1 + microwave

        return time_dependent_propagator(
            hamiltonian=hamiltonian,
            t0=0.0,
            t1=pulse_duration,
            steps=5000,
        )

    x_logical = project_operator(
        operator=pulse_propagator(0.0),
        basis_states=[zero_nv1, one_nv1],
    )

    y_logical = project_operator(
        operator=pulse_propagator(np.pi / 2),
        basis_states=[zero_nv1, one_nv1],
    )

    x_ratio = (
        x_logical.full()[0, 1]
        / x_logical.full()[1, 0]
    )

    y_ratio = (
        y_logical.full()[0, 1]
        / y_logical.full()[1, 0]
    )

    assert y_ratio / x_ratio == pytest.approx(
        -1.0,
        abs=1e-2,
    )


def test_joas_nv1_finite_sine_pi_pulse_in_interacting_pair():
    """A finite sine MW pi pulse drives NV1 correctly in the physical 9D pair."""

    system = SpinSystem([1])

    omega = 2 * np.pi * 295.18

    # --- Physical single-NV Hamiltonians ---

    theta_nv1 = np.deg2rad(74.08)
    omega_nv1 = omega * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv1,
    )

    theta_nv2 = np.deg2rad(3.58)
    omega_nv2 = omega * np.array([
        np.sin(theta_nv2),
        0.0,
        np.cos(theta_nv2),
    ])

    h_nv2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv2,
    )

    # --- Dressed effective interaction ---

    basis_nv1 = dressed_spin_one_basis(h_nv1)
    basis_nv2 = dressed_spin_one_basis(h_nv2)

    _, _, sz_nv1 = dressed_spin_operators(
        basis_states=basis_nv1,
        spin=1,
    )

    _, _, sz_nv2 = dressed_spin_operators(
        basis_states=basis_nv2,
        spin=1,
    )

    nu_dip = 0.11261
    coupling = 2 * np.pi * nu_dip

    interaction = coupling * tensor(
        sz_nv1,
        sz_nv2,
    )

    h_two = two_electron_hamiltonian(
        h_nv1=h_nv1,
        h_nv2=h_nv2,
        interaction=interaction,
    )

    # --- Joas logical branches ---

    zero_nv1, one_nv1 = electron_logical_states(
        h_nv1,
        excited_state="+1",
    )

    zero_nv2, _ = electron_logical_states(
        h_nv2,
        excited_state="-1",
    )

    initial = tensor(
        zero_nv1,
        zero_nv2,
    )

    target = tensor(
        one_nv1,
        zero_nv2,
    )

    # --- Physical NV1 microwave operator ---

    sx, _, _ = spin_operators(1)
    identity = qeye(3)

    sx_nv1 = tensor(
        sx,
        identity,
    )

    # NV1 resonance frequency on the NV2 = |0> branch.
    energy_zero = np.real(
        zero_nv1.dag() * h_nv1 * zero_nv1
    )

    energy_one = np.real(
        one_nv1.dag() * h_nv1 * one_nv1
    )

    transition_frequency = (
        energy_one - energy_zero
    )

    matrix_element = abs(
        one_nv1.dag()
        * sx
        * zero_nv1
    )

    # --- Finite sine-shaped pi pulse ---

    pulse_duration = 0.1

    peak_amplitude = (
        np.pi**2
        / (
            2
            * np.sqrt(2.0)
            * matrix_element
            * pulse_duration
        )
    )

    def hamiltonian(time):
        envelope = sine_envelope(
            time=time,
            duration=pulse_duration,
            peak_amplitude=peak_amplitude,
        )

        microwave = microwave_hamiltonian(
            time=time,
            control=sx_nv1,
            drives=[
                {
                    "omega": transition_frequency,
                    "phase": 0.0,
                    "omega_x": envelope,
                    "omega_y": 0.0,
                }
            ],
        )

        return h_two + microwave

    propagator = time_dependent_propagator(
        hamiltonian=hamiltonian,
        t0=0.0,
        t1=pulse_duration,
        steps=5000,
    )

    final = propagator * initial

    population = abs(
        target.overlap(final)
    ) ** 2

    assert population > 0.99


def test_joas_nv1_finite_sine_pi_pulse_with_conditional_shift():
    """A finite NV1 pi pulse follows the dipolar conditional frequency shift."""

    system = SpinSystem([1])

    omega = 2 * np.pi * 295.18

    theta_nv1 = np.deg2rad(74.08)
    omega_nv1 = omega * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv1,
    )

    theta_nv2 = np.deg2rad(3.58)
    omega_nv2 = omega * np.array([
        np.sin(theta_nv2),
        0.0,
        np.cos(theta_nv2),
    ])

    h_nv2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv2,
    )

    basis_nv1 = dressed_spin_one_basis(h_nv1)
    basis_nv2 = dressed_spin_one_basis(h_nv2)

    _, _, sz_nv1 = dressed_spin_operators(
        basis_states=basis_nv1,
        spin=1,
    )

    _, _, sz_nv2 = dressed_spin_operators(
        basis_states=basis_nv2,
        spin=1,
    )

    nu_dip = 0.11261
    coupling = 2 * np.pi * nu_dip

    interaction = coupling * tensor(
        sz_nv1,
        sz_nv2,
    )

    h_two = two_electron_hamiltonian(
        h_nv1=h_nv1,
        h_nv2=h_nv2,
        interaction=interaction,
    )

    zero_nv1, one_nv1 = electron_logical_states(
        h_nv1,
        excited_state="+1",
    )

    _, one_nv2 = electron_logical_states(
        h_nv2,
        excited_state="-1",
    )

    initial = tensor(
        zero_nv1,
        one_nv2,
    )

    target = tensor(
        one_nv1,
        one_nv2,
    )

    sx, _, _ = spin_operators(1)
    identity = qeye(3)

    sx_nv1 = tensor(
        sx,
        identity,
    )

    energy_zero = np.real(
        initial.dag() * h_two * initial
    )

    energy_one = np.real(
        target.dag() * h_two * target
    )

    transition_frequency = (
        energy_one - energy_zero
    )

    matrix_element = abs(
        one_nv1.dag()
        * sx
        * zero_nv1
    )

    pulse_duration = 0.1

    peak_amplitude = (
        np.pi**2
        / (
            2
            * np.sqrt(2.0)
            * matrix_element
            * pulse_duration
        )
    )

    def hamiltonian(time):
        envelope = sine_envelope(
            time=time,
            duration=pulse_duration,
            peak_amplitude=peak_amplitude,
        )

        microwave = microwave_hamiltonian(
            time=time,
            control=sx_nv1,
            drives=[
                {
                    "omega": transition_frequency,
                    "phase": 0.0,
                    "omega_x": envelope,
                    "omega_y": 0.0,
                }
            ],
        )

        return h_two + microwave

    propagator = time_dependent_propagator(
        hamiltonian=hamiltonian,
        t0=0.0,
        t1=pulse_duration,
        steps=5000,
    )

    final = propagator * initial

    population = abs(
        target.overlap(final)
    ) ** 2

    assert population > 0.99


def test_joas_two_electron_finite_xy8_gate():
    """Finite physical MW pulses reproduce the Joas sqrt(ZZ) sequence."""

    system = SpinSystem([1])

    omega = 2 * np.pi * 295.18

    # --- Single-NV Hamiltonians ---

    theta_nv1 = np.deg2rad(74.08)
    omega_nv1 = omega * np.array([
        np.sin(theta_nv1), 0.0, np.cos(theta_nv1)
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv1,
    )

    theta_nv2 = np.deg2rad(3.58)
    omega_nv2 = omega * np.array([
        np.sin(theta_nv2), 0.0, np.cos(theta_nv2)
    ])

    h_nv2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv2,
    )

    # --- Interaction ---

    basis_nv1 = dressed_spin_one_basis(h_nv1)
    basis_nv2 = dressed_spin_one_basis(h_nv2)

    _, _, sz_nv1 = dressed_spin_operators(
        basis_states=basis_nv1,
        spin=1,
    )

    _, _, sz_nv2 = dressed_spin_operators(
        basis_states=basis_nv2,
        spin=1,
    )

    nu_dip = 0.11261
    coupling = 2 * np.pi * nu_dip

    interaction = coupling * tensor(
        sz_nv1,
        sz_nv2,
    )

    h_two = two_electron_hamiltonian(
        h_nv1=h_nv1,
        h_nv2=h_nv2,
        interaction=interaction,
    )

    # --- Logical states ---

    zero_nv1, one_nv1 = electron_logical_states(
        h_nv1,
        excited_state="+1",
    )

    zero_nv2, one_nv2 = electron_logical_states(
        h_nv2,
        excited_state="-1",
    )

    logical_basis = two_electron_logical_basis(
        nv1_states=(zero_nv1, one_nv1),
        nv2_states=(zero_nv2, one_nv2),
    )

    # --- XY8 timing ---

    n_pi = 8

    tau_2 = 1 / (4 * n_pi * nu_dip)

    tau_1 = 2.5 * abs(tau_2)

    schedule = xy8_gate_schedule(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    gate_duration = 8 * tau_1

    # --- Physical microwave operators ---

    sx, _, _ = spin_operators(1)
    identity = qeye(3)

    sx_nv1 = tensor(sx, identity)
    sx_nv2 = tensor(identity, sx)

    frequency_nv1 = np.real(
        one_nv1.dag() * h_nv1 * one_nv1
        - zero_nv1.dag() * h_nv1 * zero_nv1
    )

    frequency_nv2 = np.real(
        one_nv2.dag() * h_nv2 * one_nv2
        - zero_nv2.dag() * h_nv2 * zero_nv2
    )

    matrix_element_nv1 = abs(
        one_nv1.dag() * sx * zero_nv1
    )

    matrix_element_nv2 = abs(
        one_nv2.dag() * sx * zero_nv2
    )

    # 100 ns finite pi pulses.
    pulse_duration = 0.05

    gate_start = 0.0

    gate_end = 8 * tau_1

    peak_nv1 = (
        np.pi**2
        / (
            2
            * np.sqrt(2)
            * matrix_element_nv1
            * pulse_duration
        )
    )

    peak_nv2 = (
        np.pi**2
        / (
            2
            * np.sqrt(2)
            * matrix_element_nv2
            * pulse_duration
        )
    )

    # --- Full time-dependent Hamiltonian ---

    def hamiltonian(time):
        microwave = 0 * h_two

        for center, nv, axis in schedule:
            if nv == 1:
                control = sx_nv1
                frequency = frequency_nv1
                peak = peak_nv1
            else:
                control = sx_nv2
                frequency = frequency_nv2
                peak = peak_nv2

            envelope = centered_sine_envelope(
                time=time,
                center=center,
                duration=pulse_duration,
                peak_amplitude=peak,
            )

            if envelope == 0.0:
                continue

            axis_phase = (
                0.0
                if axis == "x"
                else np.pi / 2
            )

            phase = axis_phase

            microwave += microwave_hamiltonian(
                time=time,
                control=control,
                drives=[
                    {
                        "omega": frequency,
                        "phase": phase,
                        "omega_x": envelope,
                        "omega_y": 0.0,
                    }
                ],
            )

        return h_two + microwave

    # --- Piecewise finite-pulse propagation ---

    pulse_propagators = []
    pulse_starts = []
    pulse_ends = []

    steps_per_pulse = 5000

    for center, nv, axis in schedule:
        start = center - pulse_duration / 2
        end = center + pulse_duration / 2

        if nv == 1:
            control = sx_nv1
            frequency = frequency_nv1
            peak = peak_nv1
        else:
            control = sx_nv2
            frequency = frequency_nv2
            peak = peak_nv2

        axis_phase = (
            0.0
            if axis == "x"
            else np.pi / 2
        )

        def envelope(
            time,
            center=center,
            peak=peak,
            frequency=frequency,
            axis_phase=axis_phase,
        ):
            amplitude = centered_sine_envelope(
                time=time,
                center=center,
                duration=pulse_duration,
                peak_amplitude=peak,
            )

            return np.sqrt(2.0) * amplitude * np.cos(
                frequency * time + axis_phase
            )

        pulse = finite_pulse_propagator(
            free_hamiltonian=h_two,
            control_operator=control,
            duration=pulse_duration,
            envelope=envelope,
            steps=steps_per_pulse,
            start_time=start,
        )

        pulse_propagators.append(pulse)
        pulse_starts.append(start)
        pulse_ends.append(end)

    durations = [pulse_starts[0] - gate_start]

    for previous_end, next_start in zip(
        pulse_ends[:-1],
        pulse_starts[1:],
    ):
        duration = next_start - previous_end

        if duration < 0:
            raise ValueError(
                "Finite microwave pulses overlap."
            )

        durations.append(duration)

    durations.append(
        gate_end - pulse_ends[-1]
    )

    propagator = finite_pulse_sequence_propagator(
        free_hamiltonian=h_two,
        durations=durations,
        pulse_propagators=pulse_propagators,
    )

    logical_propagator = project_operator(
        operator=propagator,
        basis_states=logical_basis,
    )

    logical_propagator.dims = [[2, 2], [2, 2]]

    # First diagnostic: logical-subspace survival.
    survival = (
        np.linalg.norm(
            logical_propagator.full(),
            "fro",
        ) ** 2
        / 4
    )

    assert survival > 0.95

    target = sqrt_zz_gate().dag()

    corrected_propagator = remove_local_z_phases(
            logical_propagator
        )
    
    corrected_target = remove_local_z_phases(
        target
    )

    corrected_gate_error = np.linalg.norm(
        corrected_propagator.full()
        - corrected_target.full()
    )

    print(
        "corrected gate error =",
        corrected_gate_error,
    )

    logical_basis_states = [
        tensor(basis(2, 0), basis(2, 0)),
        tensor(basis(2, 0), basis(2, 1)),
        tensor(basis(2, 1), basis(2, 0)),
        tensor(basis(2, 1), basis(2, 1)),
    ]

    def dynamical_map(density_matrix):
        return unitary_dynamical_map(
            density_matrix=density_matrix,
            propagator=corrected_propagator,
        )

    fidelity = average_gate_fidelity(
        dynamical_map=dynamical_map,
        target=corrected_target,
        basis_states=logical_basis_states,
    )

    print("average gate fidelity =", fidelity)

    # Remove global phase using the Hilbert-Schmidt overlap.
    overlap = np.trace(
        target.full().conj().T
        @ logical_propagator.full()
    )

    phase = np.angle(overlap)

    corrected = (
        np.exp(-1j * phase)
        * logical_propagator
    )

    gate_error = np.linalg.norm(
        corrected.full() - target.full(),
        "fro",
    )

    print("survival =", survival)
    print("gate_error =", gate_error)
    print("logical propagator =")
    print(corrected.full())
    print(logical_propagator.full()) # to remove later
    print("gate_duration =", gate_duration)
    print("last pulse =", schedule[-1])

    phase = conditional_phase(logical_propagator)
    target_phase = conditional_phase(target)

    phase_error = np.angle(
        np.exp(1j * (phase - target_phase))
    )

    print("conditional phase =", phase)
    print("target conditional phase =", target_phase)
    print("conditional phase error =", phase_error)

    assert survival > 0.999
    assert abs(phase_error) < 0.01
    assert fidelity > 0.999


def test_joas_finite_pi_pulse_at_nonzero_time():
    """A delayed resonant pulse must still implement the intended pi rotation."""

    system = SpinSystem([1])

    omega = 2 * np.pi * 295.18

    theta = np.deg2rad(74.08)
    omega_nv = omega * np.array([
        np.sin(theta),
        0.0,
        np.cos(theta),
    ])

    h_nv = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv,
    )

    zero, one = electron_logical_states(
        h_nv,
        excited_state="+1",
    )

    sx, _, _ = spin_operators(1)

    e0 = np.real(zero.dag() * h_nv * zero)
    e1 = np.real(one.dag() * h_nv * one)

    frequency = e1 - e0

    matrix_element = abs(
        one.dag() * sx * zero
    )

    pulse_duration = 0.1
    center = 0.5

    peak = (
        np.pi**2
        / (
            2
            * np.sqrt(2)
            * matrix_element
            * pulse_duration
        )
    )

    def hamiltonian(time):
        envelope = centered_sine_envelope(
            time=time,
            center=center,
            duration=pulse_duration,
            peak_amplitude=peak,
        )

        microwave = microwave_hamiltonian(
            time=time,
            control=sx,
            drives=[
                {
                    "omega": frequency,
                    "phase": 0.0,
                    "omega_x": envelope,
                    "omega_y": 0.0,
                }
            ],
        )

        return h_nv + microwave

    initial_time = center - pulse_duration / 2
    final_time = center + pulse_duration / 2

    propagator = time_dependent_propagator(
        hamiltonian=hamiltonian,
        t0=initial_time,
        t1=final_time,
        steps=5000,
    )

    final_state = propagator * zero

    population_one = abs(
        one.dag() * final_state
    ) ** 2

    print("population_one =", population_one)

    assert population_one > 0.99


def test_joas_delayed_x_pulse_phase():
    """A delayed physical pulse must preserve the intended rotation axis."""

    system = SpinSystem([1])

    omega = 2 * np.pi * 295.18
    theta = np.deg2rad(74.08)

    omega_nv = omega * np.array([
        np.sin(theta),
        0.0,
        np.cos(theta),
    ])

    h_nv = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv,
    )

    zero, one = electron_logical_states(
        h_nv,
        excited_state="+1",
    )

    sx, _, _ = spin_operators(1)

    e0 = np.real(zero.dag() * h_nv * zero)
    e1 = np.real(one.dag() * h_nv * one)

    frequency = e1 - e0

    matrix_element = abs(
        one.dag() * sx * zero
    )

    pulse_duration = 0.1
    peak = (
        np.pi**2
        / (
            2
            * np.sqrt(2)
            * matrix_element
            * pulse_duration
        )
    )

    def pulse_ratio(center):
        def hamiltonian(time):
            envelope = centered_sine_envelope(
                time=time,
                center=center,
                duration=pulse_duration,
                peak_amplitude=peak,
            )

            return h_nv + microwave_hamiltonian(
                time=time,
                control=sx,
                drives=[{
                    "omega": frequency,
                    "phase": 0.0,
                    "omega_x": envelope,
                    "omega_y": 0.0,
                }],
            )

        t0 = center - pulse_duration / 2
        t1 = center + pulse_duration / 2

        propagator = time_dependent_propagator(
            hamiltonian=hamiltonian,
            t0=t0,
            t1=t1,
            steps=5000,
        )

        free_final = (1j * h_nv * t1).expm()
        free_initial = (-1j * h_nv * t0).expm()

        propagator = (
            free_final
            * propagator
            * free_initial
        )

        u01 = zero.dag() * propagator * one
        u10 = one.dag() * propagator * zero

        return u01 / u10

    ratio_0 = pulse_ratio(0.1)
    ratio_delayed = pulse_ratio(0.5)

    print("ratio_0 =", ratio_0)
    print("ratio_delayed =", ratio_delayed)
    print("relative =", ratio_delayed / ratio_0)


def test_joas_setting_2_full_nv_hamiltonian():
    """Build and evolve the full Joas Setting 2 model with N14 nuclei."""

    # ------------------------------------------------------------------
    # 1. Define the full two-NV system and Joas Setting 2 parameters
    # ------------------------------------------------------------------

    system = SpinSystem([1, 1, 1, 1])

    D_1 = 2 * np.pi * 2865.42
    D_2 = 2 * np.pi * 2867.27

    Q = 2 * np.pi * (-4.945)

    A = 2 * np.pi * np.diag([
        -2.62,
        -2.62,
        -2.162,
    ])

    omega_e = 2 * np.pi * 295.18
    omega_n = 2 * np.pi * 0.03241

    theta_1 = np.deg2rad(74.08)
    theta_2 = np.deg2rad(3.58)

    omega_e_1 = omega_e * np.array([
        np.sin(theta_1),
        0.0,
        np.cos(theta_1),
    ])

    omega_e_2 = omega_e * np.array([
        np.sin(theta_2),
        0.0,
        np.cos(theta_2),
    ])

    omega_n_1 = omega_n * np.array([
        np.sin(theta_1),
        0.0,
        np.cos(theta_1),
    ])

    omega_n_2 = omega_n * np.array([
        np.sin(theta_2),
        0.0,
        np.cos(theta_2),
    ])

    # ------------------------------------------------------------------
    # 2. Build the non-interacting 81D two-NV Hamiltonian
    # ------------------------------------------------------------------

    hamiltonian = nv_register_hamiltonian(
        system=system,
        nv_parameters=[
            {
                "D": D_1,
                "omega_e": omega_e_1,
                "Q": Q,
                "omega_n": omega_n_1,
                "A": A,
            },
            {
                "D": D_2,
                "omega_e": omega_e_2,
                "Q": Q,
                "omega_n": omega_n_2,
                "A": A,
            },
        ],
    )

    assert hamiltonian.shape == (81, 81)
    assert hamiltonian.isherm

    # ------------------------------------------------------------------
    # 3. Build the electronic-only Hamiltonians
    # ------------------------------------------------------------------

    electron_system = SpinSystem([1])

    h_e1 = electronic_nv_hamiltonian(
        system=electron_system,
        electron_site=0,
        D=D_1,
        omega_e=omega_e_1,
    )

    h_e2 = electronic_nv_hamiltonian(
        system=electron_system,
        electron_site=0,
        D=D_2,
        omega_e=omega_e_2,
    )

    # ------------------------------------------------------------------
    # 4. Construct the dressed electronic spin operators
    # ------------------------------------------------------------------

    basis_e1 = dressed_spin_one_basis(h_e1)
    basis_e2 = dressed_spin_one_basis(h_e2)

    _, _, sz_e1 = dressed_spin_operators(
        basis_states=basis_e1,
        spin=1,
    )

    _, _, sz_e2 = dressed_spin_operators(
        basis_states=basis_e2,
        spin=1,
    )

    # ------------------------------------------------------------------
    # 5. Embed the dressed electronic operators in the 81D Hilbert space
    # ------------------------------------------------------------------

    sz_e1_full = embed_operator(
        operator=sz_e1,
        site=0,
        system=system,
    )

    sz_e2_full = embed_operator(
        operator=sz_e2,
        site=2,
        system=system,
    )

    # ------------------------------------------------------------------
    # 6. Build the effective dipolar interaction and free Hamiltonian
    # ------------------------------------------------------------------

    g = 2 * np.pi * 0.11261

    interaction = g * sz_e1_full * sz_e2_full

    h_free = hamiltonian + interaction

    assert interaction.shape == (81, 81)
    assert interaction.isherm

    assert h_free.shape == (81, 81)
    assert h_free.isherm

    # ------------------------------------------------------------------
    # 7. Verify that the embedded interaction retains the Joas coupling g
    # ------------------------------------------------------------------

    zz_full = sz_e1_full * sz_e2_full

    numerator = (
        zz_full.dag()
        * interaction
    ).tr()

    denominator = (
        zz_full.dag()
        * zz_full
    ).tr()

    extracted_g = numerator / denominator

    assert np.isclose(
        np.real(extracted_g),
        g,
        rtol=1e-12,
    )

    assert np.isclose(
        np.imag(extracted_g),
        0.0,
        atol=1e-12,
    )

    # ------------------------------------------------------------------
    # 8. Prepare the electronic and nuclear initial states
    # ------------------------------------------------------------------

    zero_e1 = basis_e1[1]
    zero_e2 = basis_e2[1]

    electron_state = tensor(
        zero_e1,
        zero_e2,
    )

    rho_e = electron_state.proj()

    nuclear_states = [
        qeye(3) / 3,
        qeye(3) / 3,
    ]

    # ------------------------------------------------------------------
    # 9. Propagate the complete electron-nuclear system
    # ------------------------------------------------------------------

    evolution_time = 0.1  # microseconds

    propagator = static_propagator(
        hamiltonian=h_free,
        time=evolution_time,
    )

    rho_evolved = electron_nuclear_dynamical_map(
        density_matrix=rho_e,
        propagator=propagator,
        nuclear_states=nuclear_states,
    )

    # ------------------------------------------------------------------
    # 10. Validate the reduced electronic density matrix
    # ------------------------------------------------------------------

    assert rho_evolved.dims == [[3, 3], [3, 3]]
    assert rho_evolved.isherm

    assert np.isclose(
        rho_evolved.tr(),
        1.0,
    )

    assert (
        np.min(rho_evolved.eigenenergies())
        > -1e-12
    )

    # compute the electronic purity to check for decoherence.
    purity = (rho_evolved * rho_evolved).tr().real

    assert 0.0 < purity < 1.0