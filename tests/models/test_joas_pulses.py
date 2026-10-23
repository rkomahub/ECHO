import numpy as np
import pytest
from qutip import Qobj, qeye, tensor

from echo_spin.control.microwave import microwave_hamiltonian
from echo_spin.control.pulses import (
    sine_envelope,
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
)
from echo_spin.models.joas import (
    electron_logical_states,
    two_electron_hamiltonian,
)
from echo_spin.nv.hamiltonians import electronic_nv_hamiltonian


def _assert_logical_subspace_survival(
    propagator,
    logical_states,
    maximum_leakage=0.01,
):
    """Check leakage from each logical input and average projected survival."""
    logical_projector = logical_states[0].proj()

    for state in logical_states[1:]:
        logical_projector += state.proj()

    leakage_projector = (
        qeye(propagator.dims[0]) - logical_projector
    )

    leakages = []

    for initial_state in logical_states:
        final_state = propagator * initial_state

        logical_population = sum(
            abs(state.overlap(final_state)) ** 2
            for state in logical_states
        )
        leakage = float(
            (leakage_projector * final_state).norm() ** 2
        )
        leakages.append(leakage)

        assert logical_population + leakage == pytest.approx(
            1.0, abs=1e-10
        )
        assert leakage < maximum_leakage, (
            f"Logical-input leakage {leakage:.6g} exceeds "
            f"{maximum_leakage:.6g}"
        )

    projected_propagator = project_operator(
        propagator,
        logical_states,
    )

    average_survival = float(
        np.linalg.norm(
            projected_propagator.full(),
            ord="fro",
        ) ** 2 / len(logical_states)
    )

    assert average_survival == pytest.approx(
        1.0 - np.mean(leakages),
        abs=1e-10,
    )
    assert average_survival > 1.0 - maximum_leakage


def _assert_physical_pair_pi_gate(
    propagator,
    drift,
    duration,
    logical_states,
    transition_element,
    driven_site,
):
    """Check pair pi-gate fidelity after removing known drift evolution."""
    interaction_propagator = (
        (1j * drift * duration).expm()
        * propagator
    )

    projected = project_operator(
        operator=interaction_propagator,
        basis_states=logical_states,
    ).full()

    transition_phase = (
        transition_element / abs(transition_element)
    )

    local_pi = -1j * np.array(
        [
            [0.0, np.conjugate(transition_phase)],
            [transition_phase, 0.0],
        ],
        dtype=complex,
    )

    identity = np.eye(2, dtype=complex)

    if driven_site == 0:
        ideal_gate = np.kron(local_pi, identity)
    elif driven_site == 1:
        ideal_gate = np.kron(identity, local_pi)
    else:
        raise ValueError("driven_site must be 0 or 1.")

    dimension = len(logical_states)

    retained_weight = float(
        np.trace(projected.conj().T @ projected).real
    )
    overlap = np.trace(
        ideal_gate.conj().T @ projected
    )

    average_fidelity = float(
        (retained_weight + abs(overlap) ** 2)
        / (dimension * (dimension + 1))
    )
    average_survival = retained_weight / dimension

    assert average_fidelity <= average_survival + 1e-10
    assert average_fidelity > 0.99, (
        f"Pair pi-gate fidelity = {average_fidelity:.8f}, "
        f"survival = {average_survival:.8f}"
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

    _assert_logical_subspace_survival(
    propagator,
    [logical_zero, logical_one],
    )


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

    _assert_logical_subspace_survival(
        propagator,
        [logical_zero, logical_one],
    )


def test_joas_two_electron_nv1_physical_pi_pulse():
    r"""The physical NV1 pulse survives embedding into the interacting \(9D\) system."""
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

    zero_nv2, one_nv2 = electron_logical_states(
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
        """Return the interacting pair Hamiltonian during the physical pulse."""
        return (
            h_two
            + amplitude
            * np.cos(drive_frequency * time)
            * sx_nv1
        )

    steps = 10000
    dt = pulse_duration / steps

    def midpoint_hamiltonian(endpoint_time):
        """Sample the interacting pulse Hamiltonian at the interval midpoint."""
        return hamiltonian(endpoint_time - dt / 2)

    propagator = time_dependent_propagator(
        hamiltonian=midpoint_hamiltonian,
        t0=0.0,
        t1=pulse_duration,
        steps=steps,
    )

    final_state = propagator * initial_state

    target_population = abs(
        target_state.overlap(final_state)
    ) ** 2

    assert target_population > 0.99

    logical_states = [
        tensor(zero_nv1, zero_nv2),
        tensor(zero_nv1, one_nv2),
        tensor(one_nv1, zero_nv2),
        tensor(one_nv1, one_nv2),
    ]

    _assert_logical_subspace_survival(
        propagator,
        logical_states,
    )

    _assert_physical_pair_pi_gate(
        propagator=propagator,
        drift=h_two,
        duration=pulse_duration,
        logical_states=logical_states,
        transition_element=complex(
            one_nv1.dag() * sx * zero_nv1
        ),
        driven_site=0,
    )


def test_joas_two_electron_nv2_physical_pi_pulse():
    r"""The physical NV2 pulse survives embedding into the interacting \(9D\) system."""
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

    zero_nv1, one_nv1 = electron_logical_states(
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
        """Return the interacting pair Hamiltonian during the physical pulse."""
        return (
            h_two
            + amplitude
            * np.cos(drive_frequency * time)
            * sx_nv2
        )

    steps = 10000
    dt = pulse_duration / steps

    def midpoint_hamiltonian(endpoint_time):
        """Sample the interacting pulse Hamiltonian at the interval midpoint."""
        return hamiltonian(endpoint_time - dt / 2)

    propagator = time_dependent_propagator(
        hamiltonian=midpoint_hamiltonian,
        t0=0.0,
        t1=pulse_duration,
        steps=steps,
    )

    final_state = propagator * initial_state

    target_population = abs(
        target_state.overlap(final_state)
    ) ** 2

    assert target_population > 0.99

    logical_states = [
        tensor(zero_nv1, zero_nv2),
        tensor(zero_nv1, one_nv2),
        tensor(one_nv1, zero_nv2),
        tensor(one_nv1, one_nv2),
    ]

    _assert_logical_subspace_survival(
        propagator,
        logical_states,
    )

    _assert_physical_pair_pi_gate(
        propagator=propagator,
        drift=h_two,
        duration=pulse_duration,
        logical_states=logical_states,
        transition_element=complex(
            one_nv2.dag() * sx * zero_nv2
        ),
        driven_site=1,
    )


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


@pytest.mark.parametrize(
    "detuning_ratio",
    [
        pytest.param(0.0, id="resonant"),
        pytest.param(-0.1, id="detuned_minus"),
        pytest.param(0.1, id="detuned_plus"),
    ],
)
@pytest.mark.parametrize(
    "fractional_error",
    [
        pytest.param(-0.05, id="underdrive"),
        pytest.param(0.0, id="calibrated"),
        pytest.param(0.05, id="overdrive"),
    ],
)
@pytest.mark.parametrize(
    "pulse_angle",
    [
        pytest.param(np.pi, id="pi"),
        pytest.param(np.pi / 2, id="half_pi"),
    ],
)
@pytest.mark.parametrize(
    "angle_degrees, splitting_mhz, excited_state",
    [
        pytest.param(
            74.08, 2867.27, "+1", id="NV1"
        ),
        pytest.param(
            3.58, 2865.42, "-1", id="NV2"
        ),
    ],
)
def test_joas_physical_microwave_carrier_phase(
    angle_degrees,
    splitting_mhz,
    excited_state,
    pulse_angle,
    fractional_error,
    detuning_ratio,
):
    """Check physical NV rotations, amplitude sensitivity, leakage and convergence."""

    system = SpinSystem([1])

    omega_e = 2 * np.pi * 295.18
    theta_nv1 = np.deg2rad(angle_degrees)

    omega_nv1 = omega_e * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * splitting_mhz,
        omega_e=omega_nv1,
    )

    zero_nv1, one_nv1 = electron_logical_states(
        h_nv1,
        excited_state=excited_state,
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

    # Normalize carrier error to the nominal average Rabi frequency.
    detuning = detuning_ratio * pulse_angle / pulse_duration
    carrier_frequency = transition_frequency + detuning

    peak_amplitude = (
        pulse_angle * np.pi
        / (
            2
            * np.sqrt(2.0)
            * matrix_element
            * pulse_duration
        )
    )

    peak_amplitude *= 1 + fractional_error

    def pulse_propagator(phase, steps):
        """Return midpoint-sampled pulse evolution in the drift interaction picture."""
        def hamiltonian(time):
            """Return the full spin-1 Hamiltonian during the microwave pulse."""
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
                        "omega": carrier_frequency,
                        "phase": phase,
                        "omega_x": envelope,
                        "omega_y": 0.0,
                    }
                ],
            )

            return h_nv1 + microwave

        dt = pulse_duration / steps

        def midpoint_hamiltonian(endpoint_time):
            """Evaluate the physical Hamiltonian at the interval midpoint."""
            return hamiltonian(endpoint_time - dt / 2)

        lab_propagator = time_dependent_propagator(
            hamiltonian=midpoint_hamiltonian,
            t0=0.0,
            t1=pulse_duration,
            steps=steps,
        )

        return (
            (1j * h_nv1 * pulse_duration).expm()
            * lab_propagator
        )

    logical_states = [zero_nv1, one_nv1]

    transition_element = complex(
        one_nv1.dag() * sx * zero_nv1
    )
    transition_phase = (
        transition_element / abs(transition_element)
    )

    for phase in [0.0, np.pi / 2]:
        coarse_propagator = pulse_propagator(
            phase,
            steps=5000,
        )
        propagator = pulse_propagator(
            phase,
            steps=10000,
        )

        coarse_projected = project_operator(
            operator=coarse_propagator,
            basis_states=logical_states,
        ).full()

        _assert_logical_subspace_survival(
            propagator,
            logical_states,
        )

        projected = project_operator(
            operator=propagator,
            basis_states=logical_states,
        ).full()

        # Preserve the phase convention of the dressed eigenstates.
        logical_axis = np.array(
            [
                [
                    0.0,
                    np.conjugate(transition_phase)
                    * np.exp(1j * phase),
                ],
                [
                    transition_phase * np.exp(-1j * phase),
                    0.0,
                ],
            ],
            dtype=complex,
        )

        ideal_rotation = (
            np.cos(pulse_angle / 2)
            * np.eye(2, dtype=complex)
            - 1j
            * np.sin(pulse_angle / 2)
            * logical_axis
        )

        survival = float(
            np.trace(
                projected.conj().T @ projected
            ).real / 2
        )

        overlap = np.trace(
            ideal_rotation.conj().T @ projected
        )

        # Average target-state fidelity includes loss from the logical subspace.
        average_fidelity = float(
            (
                2 * survival
                + abs(overlap) ** 2
            ) / 6
        )

        coarse_survival = float(
            np.trace(
                coarse_projected.conj().T @ coarse_projected
            ).real / 2
        )
        coarse_overlap = np.trace(
            ideal_rotation.conj().T @ coarse_projected
        )
        coarse_fidelity = float(
            (
                2 * coarse_survival
                + abs(coarse_overlap) ** 2
            ) / 6
        )

        assert abs(
            average_fidelity - coarse_fidelity
        ) < 1e-3, (
            "Gate fidelity has not converged: "
            f"5000 steps = {coarse_fidelity:.8f}, "
            f"10000 steps = {average_fidelity:.8f}"
        )

        assert average_fidelity <= survival + 1e-10

        if detuning_ratio == 0.0:
            angle_error = pulse_angle * fractional_error
            expected_qubit_fidelity = (
                1 + 2 * np.cos(angle_error / 2) ** 2
            ) / 3
        else:
            def reference_propagator(steps):
                """Propagate the shaped two-level RWA reference in the drift interaction picture."""
                dt_reference = pulse_duration / steps

                def reference_hamiltonian(endpoint_time):
                    """Return the slowly varying resonant-transition coupling at the midpoint."""
                    time = endpoint_time - dt_reference / 2

                    envelope = sine_envelope(
                        time=time,
                        duration=pulse_duration,
                        peak_amplitude=peak_amplitude,
                    )

                    upper_coupling = (
                        envelope
                        * np.conjugate(transition_element)
                        / np.sqrt(2.0)
                        * np.exp(1j * (detuning * time + phase))
                    )

                    return Qobj(
                        [
                            [0.0, upper_coupling],
                            [np.conjugate(upper_coupling), 0.0],
                        ],
                        dims=[[2], [2]],
                    )

                return time_dependent_propagator(
                    hamiltonian=reference_hamiltonian,
                    t0=0.0,
                    t1=pulse_duration,
                    steps=steps,
                ).full()

            reference_fidelities = []

            for reference_steps in [1000, 2000]:
                reference = reference_propagator(reference_steps)
                reference_overlap = np.trace(
                    ideal_rotation.conj().T @ reference
                )
                reference_fidelities.append(
                    float((2 + abs(reference_overlap) ** 2) / 6)
                )

            assert abs(
                reference_fidelities[1] - reference_fidelities[0]
            ) < 1e-5

            expected_qubit_fidelity = reference_fidelities[1]

        assert abs(
            average_fidelity - expected_qubit_fidelity
        ) < 2e-3, (
            f"Physical fidelity = {average_fidelity:.8f}, "
            f"qubit reference = {expected_qubit_fidelity:.8f}"
        )

        if fractional_error == 0.0 and detuning_ratio == 0.0:
            assert average_fidelity > 0.99


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
