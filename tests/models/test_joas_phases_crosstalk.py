import numpy as np
import pytest
from qutip import qeye, sigmax, sigmay, tensor

from echo_spin.control.microwave import microwave_hamiltonian
from echo_spin.control.pulses import (
    centered_sine_envelope,
    sine_envelope,
)
from echo_spin.core.basis import project_operator
from echo_spin.core.operators import spin_operators
from echo_spin.core.system import SpinSystem
from echo_spin.dynamics.propagators import time_dependent_propagator
from echo_spin.dynamics.sequences import logical_sine_pulse_propagator
from echo_spin.models.joas import electron_logical_states
from echo_spin.nv.hamiltonians import electronic_nv_hamiltonian


@pytest.mark.parametrize(
    "splitting_mhz, angle_degrees, excited_state",
    [
        pytest.param(2867.27, 74.08, "+1", id="NV1"),
        pytest.param(2865.42, 3.58, "-1", id="NV2"),
    ],
)
@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("start_time", [0.0, 0.123])
def test_physical_rb_half_pi_primitive(
    splitting_mhz,
    angle_degrees,
    excited_state,
    axis,
    start_time,
):
    """Verify positive physical RB half-pi rotations at zero and delayed start times."""
    angle = np.deg2rad(angle_degrees)
    field = 2 * np.pi * 295.18 * np.array(
        [np.sin(angle), 0.0, np.cos(angle)]
    )

    drift = electronic_nv_hamiltonian(
        system=SpinSystem([1]),
        electron_site=0,
        D=2 * np.pi * splitting_mhz,
        omega_e=field,
    )

    zero, one = electron_logical_states(
        drift,
        excited_state=excited_state,
    )
    sx, _, _ = spin_operators(1)

    carrier = float(np.real(
        one.dag() * drift * one
        - zero.dag() * drift * zero
    ))
    element = complex(one.dag() * sx * zero)
    duration = 0.1

    logical_axis = (
        sigmax().full() if axis == "x"
        else sigmay().full()
    )
    target = (
        np.eye(2) - 1j * logical_axis
    ) / np.sqrt(2)

    fidelities = []

    for steps in [5000, 10000]:
        lab_propagator = logical_sine_pulse_propagator(
            free_hamiltonian=drift,
            control_operator=sx,
            carrier=carrier,
            transition_element=element,
            angle=np.pi / 2,
            axis=axis,
            duration=duration,
            steps=steps,
            start_time=start_time,
        )

        # Transform both endpoints into the same drift interaction picture.
        interaction_propagator = (
            (1j * drift * (start_time + duration)).expm()
            * lab_propagator
            * (-1j * drift * start_time).expm()
        )

        projected = project_operator(
            interaction_propagator,
            [zero, one],
        ).full()

        retained_weight = float(
            np.linalg.norm(projected, ord="fro") ** 2
        )
        overlap = np.trace(target.conj().T @ projected)

        fidelities.append(
            float((retained_weight + abs(overlap) ** 2) / 6)
        )

    assert abs(fidelities[1] - fidelities[0]) < 1e-3
    assert fidelities[1] > 0.99


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


@pytest.mark.parametrize(
    "addressed_index",
    [
        pytest.param(0, id="drive_NV1"),
        pytest.param(1, id="drive_NV2"),
    ],
)
def test_joas_shared_microwave_spin_one_crosstalk(addressed_index):
    """Measure direct spectator excitation under a shared physical microwave field."""
    system = SpinSystem([1])
    sx, _, _ = spin_operators(1)

    hamiltonians = []
    logical_states = []

    for splitting_mhz, angle_degrees, excited_state in [
        (2867.27, 74.08, "+1"),
        (2865.42, 3.58, "-1"),
    ]:
        angle = np.deg2rad(angle_degrees)
        field = 2 * np.pi * 295.18 * np.array(
            [np.sin(angle), 0.0, np.cos(angle)]
        )

        hamiltonian = electronic_nv_hamiltonian(
            system=system,
            electron_site=0,
            D=2 * np.pi * splitting_mhz,
            omega_e=field,
        )

        hamiltonians.append(hamiltonian)
        logical_states.append(
            electron_logical_states(
                hamiltonian,
                excited_state=excited_state,
            )
        )

    addressed_zero, addressed_one = logical_states[addressed_index]
    addressed_hamiltonian = hamiltonians[addressed_index]

    carrier = float(np.real(
        addressed_one.dag()
        * addressed_hamiltonian
        * addressed_one
        - addressed_zero.dag()
        * addressed_hamiltonian
        * addressed_zero
    ))

    matrix_element = abs(
        addressed_one.dag() * sx * addressed_zero
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

    def microwave(time):
        """Return the common carrier drive acting on either local spin."""
        envelope = sine_envelope(
            time=time,
            duration=pulse_duration,
            peak_amplitude=peak_amplitude,
        )
        return microwave_hamiltonian(
            time=time,
            control=sx,
            drives=[{
                "omega": carrier,
                "phase": 0.0,
                "omega_x": envelope,
                "omega_y": 0.0,
            }],
        )

    def evolve_local(drift, initial_state, steps):
        """Evolve one NV under the shared field using midpoint sampling."""
        dt = pulse_duration / steps

        def midpoint_hamiltonian(endpoint_time):
            """Return the local Hamiltonian at the interval midpoint."""
            return drift + microwave(endpoint_time - dt / 2)

        propagator = time_dependent_propagator(
            hamiltonian=midpoint_hamiltonian,
            t0=0.0,
            t1=pulse_duration,
            steps=steps,
        )
        return propagator * initial_state

    spectator_index = 1 - addressed_index
    spectator_zero, spectator_one = logical_states[spectator_index]

    spectator_logical_projector = (
        spectator_zero.proj() + spectator_one.proj()
    )
    spectator_leakage_projector = (
        qeye(3) - spectator_logical_projector
    )

    results = []

    for steps in [5000, 10000]:
        addressed_final = evolve_local(
            addressed_hamiltonian,
            addressed_zero,
            steps,
        )
        spectator_final = evolve_local(
            hamiltonians[spectator_index],
            spectator_zero,
            steps,
        )

        transfer = abs(
            addressed_one.overlap(addressed_final)
        ) ** 2
        spectator_excitation = abs(
            spectator_one.overlap(spectator_final)
        ) ** 2
        spectator_leakage = (
            spectator_leakage_projector * spectator_final
        ).norm() ** 2
        spectator_remaining = abs(
            spectator_zero.overlap(spectator_final)
        ) ** 2

        assert (
            spectator_remaining
            + spectator_excitation
            + spectator_leakage
        ) == pytest.approx(1.0, abs=1e-10)

        results.append(
            [transfer, spectator_excitation, spectator_leakage]
        )

    np.testing.assert_allclose(
        results[0],
        results[1],
        atol=1e-4,
        rtol=0.0,
    )

    transfer, spectator_excitation, spectator_leakage = results[1]

    assert transfer > 0.99

    print(
        f"Drive NV{addressed_index + 1}: "
        f"target transfer={transfer:.8f}, "
        f"spectator logical excitation={spectator_excitation:.8f}, "
        f"spectator leakage={spectator_leakage:.8f}"
    )

    spectator_plus = (
        spectator_zero + spectator_one
    ).unit()

    spectator_drift = hamiltonians[spectator_index]
    undo_free_evolution = (
        1j * spectator_drift * pulse_duration
    ).expm()

    phase_results = []

    for steps in [10000, 20000]:
        spectator_final = evolve_local(
            spectator_drift,
            spectator_plus,
            steps,
        )

        # Remove known drift; retain microwave-induced phase and leakage.
        interaction_state = (
            undo_free_evolution * spectator_final
        )

        amplitude_zero = spectator_zero.overlap(
            interaction_state
        )
        amplitude_one = spectator_one.overlap(
            interaction_state
        )

        coherence = (
            amplitude_zero * np.conjugate(amplitude_one)
        )
        retained_population = (
            abs(amplitude_zero) ** 2
            + abs(amplitude_one) ** 2
        )
        state_fidelity = abs(
            spectator_plus.overlap(interaction_state)
        ) ** 2

        # This identity includes logical population loss without renormalizing.
        assert state_fidelity == pytest.approx(
            retained_population / 2 + np.real(coherence),
            abs=1e-12,
        )
        assert state_fidelity <= retained_population + 1e-10

        phase_results.append(
            [coherence, retained_population, state_fidelity]
        )

    np.testing.assert_allclose(
        phase_results[0],
        phase_results[1],
        atol=1e-4,
        rtol=0.0,
    )

    coherence, retained_population, state_fidelity = phase_results[1]

    relative_phase = float(np.angle(coherence))
    coherence_magnitude = float(2 * abs(coherence))
    superposition_leakage = float(
        1 - retained_population
    )

    print(
        f"Spectator NV{spectator_index + 1}: "
        f"relative phase={relative_phase:.8f} rad, "
        f"coherence magnitude={coherence_magnitude:.8f}, "
        f"superposition leakage={superposition_leakage:.8f}, "
        f"state fidelity={float(state_fidelity):.8f}"
    )
