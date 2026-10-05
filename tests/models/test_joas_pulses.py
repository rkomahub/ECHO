import numpy as np
import pytest

from qutip import basis, qeye, sigmax, sigmay, tensor

from echo_spin.control.microwave import microwave_hamiltonian
from echo_spin.control.pulses import (
    centered_sine_envelope,
    sine_envelope,
    sine_pi_pulse_amplitude,
)
from echo_spin.core.basis import (
    dressed_spin_one_basis,
    dressed_spin_operators,
    project_operator,
)
from echo_spin.core.operators import spin_operators
from echo_spin.core.system import SpinSystem
from echo_spin.dynamics.propagators import (
    time_dependent_propagator,
    unitary_dynamical_map,
)
from echo_spin.dynamics.sequences import (
    finite_pulse_propagator,
    finite_pulse_sequence_propagator,
)
from echo_spin.gates.gates import (
    average_gate_fidelity,
    conditional_phase,
    remove_local_z_phases,
    sqrt_zz_gate,
)
from echo_spin.models.joas import (
    electron_logical_states,
    reduced_free_hamiltonian,
    two_electron_hamiltonian,
    two_electron_logical_basis,
    two_pulse_gate_durations,
    xy8_gate_intervals,
    xy8_gate_schedule,
)
from echo_spin.nv.hamiltonians import electronic_nv_hamiltonian
from echo_spin.control.rotations import two_qubit_rotation


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

# -----------------------------------------------------------------------

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
        D=2 * np.pi * 2867.27, # NV1 = misaligned target, +1-like transition ~2990.8 MHz
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
        D=2 * np.pi * 2865.42, # NV2 = nearly aligned control, -1-like transition ~2571.0 MHz
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
        D=2 * np.pi * 2867.27, # NV1 = misaligned target, +1-like transition ~2990.8 MHz
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
        D=2 * np.pi * 2865.42, # NV2 = nearly aligned control, -1-like transition ~2571.0 MHz
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
        D=2 * np.pi * 2867.27, # NV1 = misaligned target, +1-like transition ~2990.8 MHz
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
        D=2 * np.pi * 2865.42, # NV2 = nearly aligned control, -1-like transition ~2571.0 MHz
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
        D=2 * np.pi * 2867.27, # NV1 = misaligned target, +1-like transition ~2990.8 MHz
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
        D=2 * np.pi * 2865.42, # NV2 = nearly aligned control, -1-like transition ~2571.0 MHz
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

# -----------------------------------------------------------

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
        D=2 * np.pi * 2867.27, # NV1 = misaligned target, +1-like transition ~2990.8 MHz
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
        D=2 * np.pi * 2867.27, # NV1 = misaligned target, +1-like transition ~2990.8 MHz
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
        D = 2 * np.pi * 2867.27, # NV1 = misaligned target, +1-like transition ~2990.8 MHz
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
        D = 2 * np.pi * 2865.42, # NV2 = nearly aligned control, -1-like transition ~2571.0 MHz
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
        D = 2 * np.pi * 2867.27, # NV1 = misaligned target, +1-like transition ~2990.8 MHz
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
        D = 2 * np.pi * 2865.42, # NV2 = nearly aligned control, -1-like transition ~2571.0 MHz
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
        D = 2 * np.pi * 2867.27, # NV1 = misaligned target, +1-like transition ~2990.8 MHz
        omega_e=omega_nv1,
    )

    theta_nv2 = np.deg2rad(3.58)
    omega_nv2 = omega * np.array([
        np.sin(theta_nv2), 0.0, np.cos(theta_nv2)
    ])

    h_nv2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D = 2 * np.pi * 2865.42, # NV2 = nearly aligned control, -1-like transition ~2571.0 MHz
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

# -----------------------------------------------------------

def test_joas_finite_pi_pulse_at_nonzero_time():
    """A delayed resonant pulse must still implement the intended pi rotation."""

    system = SpinSystem([1])

    omega = 2 * np.pi * 295.18

    theta = np.deg2rad(74.08) # NV1 = misaligned target, +1-like transition ~2990.8 MHz
    omega_nv = omega * np.array([
        np.sin(theta),
        0.0,
        np.cos(theta),
    ])

    h_nv = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27, # NV1 = misaligned target, +1-like transition ~2990.8 MHz
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
    theta = np.deg2rad(74.08) # NV1 = misaligned target, +1-like transition ~2990.8 MHz

    omega_nv = omega * np.array([
        np.sin(theta),
        0.0,
        np.cos(theta),
    ])

    h_nv = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27, # NV1 = misaligned target, +1-like transition ~2990.8 MHz
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