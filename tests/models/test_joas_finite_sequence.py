import numpy as np
import pytest
from qutip import basis, qeye, sigmax, sigmay, tensor

from echo_spin.control.microwave import microwave_hamiltonian
from echo_spin.control.pulses import (
    centered_sine_envelope,
    sine_envelope,
    sine_pi_pulse_amplitude,
)
from echo_spin.control.rotations import two_qubit_rotation
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
    finite_pulse_intervals,
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
    xy8_gate_schedule,
)
from echo_spin.nv.hamiltonians import electronic_nv_hamiltonian


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
        qubit=0,  # NV1 is the left tensor factor.
        angle=np.pi,
        axis="x",
    )

    x1_operator = tensor(sigmax(), qeye(2))

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
        qubit=1,  # NV2 is the right tensor factor.
        angle=np.pi,
        axis="x",
    )

    x2_operator = tensor(qeye(2), sigmax())

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

    ideal_intervals = two_pulse_gate_durations(
        tau_1=tau_1,
        tau_2=tau_2,
    )
    pulse_centers = np.cumsum(ideal_intervals[:-1]).tolist()
    total_duration = sum(ideal_intervals)

    durations = finite_pulse_intervals(
        pulse_centers=pulse_centers,
        pulse_duration=pulse_duration,
        total_duration=total_duration,
    )

    assert sum(durations) + 4 * pulse_duration == pytest.approx(
        total_duration,
        abs=1e-12,
    )

    # NV1 is the left tensor factor; NV2 is the right tensor factor.
    x1_operator = tensor(sigmax(), qeye(2))
    x2_operator = tensor(qeye(2), sigmax())

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

    schedule = xy8_gate_schedule(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    total_duration = n_pi * tau_1
    intervals = finite_pulse_intervals(
        pulse_centers=[time for time, _, _ in schedule],
        pulse_duration=pulse_duration,
        total_duration=total_duration,
    )

    assert (
        sum(intervals) + len(schedule) * pulse_duration
    ) == pytest.approx(total_duration, abs=1e-12)

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
        qubit = nv - 1
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
