import numpy as np
import pytest

from qutip import Qobj, qeye, basis, tensor, sigmax, sigmay

from echo_spin.core.system import SpinSystem
from echo_spin.core.basis import project_operator
from echo_spin.control.rotations import two_qubit_rotation
from echo_spin.nv.hamiltonians import electronic_nv_hamiltonian
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
from echo_spin.gates.gates import sqrt_zz_gate
from echo_spin.dynamics.propagators import time_dependent_propagator
from echo_spin.control.pulses import sine_envelope, sine_pi_pulse_amplitude

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