import numpy as np
from qutip import basis, qeye, tensor

from echo_spin.control.microwave import (
    control_operator,
    microwave_hamiltonian,
)
from echo_spin.control.pulses import centered_sine_envelope, sine_envelope
from echo_spin.core.basis import (
    basis_unitary,
    dressed_spin_one_basis,
    dressed_spin_operators,
)
from echo_spin.core.operators import (
    embed_operator,
    spin_operators,
)
from echo_spin.core.system import SpinSystem
from echo_spin.dynamics.propagators import (
    electron_nuclear_dynamical_map,
    logical_electron_nuclear_dynamical_map,
    rotating_frame_hamiltonian,
    rotating_frame_propagator,
    static_propagator,
    time_dependent_propagator,
)
from echo_spin.dynamics.sequences import (
    finite_pulse_intervals,
    finite_pulse_sequence_propagator,
)
from echo_spin.gates.gates import (
    average_gate_fidelity,
    sqrt_zz_gate,
)
from echo_spin.models.joas import xy8_gate_schedule
from echo_spin.nv.hamiltonians import electronic_nv_hamiltonian
from echo_spin.nv.registers import nv_register_hamiltonian


def test_joas_setting_2_full_nv_hamiltonian():
    """Build and evolve the full Joas Setting 2 model with N14 nuclei."""

    # ------------------------------------------------------------------
    # 1. Define the full two-NV system and Joas Setting 2 parameters
    # ------------------------------------------------------------------

    system = SpinSystem([1, 1, 1, 1])

    D_1 = 2 * np.pi * 2867.27  # NV1: misaligned, 2990.8 MHz
    D_2 = 2 * np.pi * 2865.42  # NV2: aligned, 2571.0 MHz

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


def test_joas_setting_2_full_control_operator():
    """Build the full Joas microwave control operator with N14 nuclei."""

    # ------------------------------------------------------------------
    # 1. Define the full two-NV system
    # ------------------------------------------------------------------

    system = SpinSystem([1, 1, 1, 1])
    electron_system = SpinSystem([1])

    D_1 = 2 * np.pi * 2867.27  # NV1: misaligned, 2990.8 MHz
    D_2 = 2 * np.pi * 2865.42  # NV2: aligned, 2571.0 MHz

    omega_e = 2 * np.pi * 295.18

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

    # ------------------------------------------------------------------
    # 2. Build the electronic dressed bases
    # ------------------------------------------------------------------

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

    basis_e1 = dressed_spin_one_basis(h_e1)
    basis_e2 = dressed_spin_one_basis(h_e2)

    # ------------------------------------------------------------------
    # 3. Build the dressed electronic Sx operators
    # ------------------------------------------------------------------

    sx_e1, _, _ = dressed_spin_operators(
        basis_states=basis_e1,
        spin=1,
    )

    sx_e2, _, _ = dressed_spin_operators(
        basis_states=basis_e2,
        spin=1,
    )

    sx_e1_full = embed_operator(
        operator=sx_e1,
        site=0,
        system=system,
    )

    sx_e2_full = embed_operator(
        operator=sx_e2,
        site=2,
        system=system,
    )

    # ------------------------------------------------------------------
    # 4. Build the nuclear Ix operators
    # ------------------------------------------------------------------

    ix, _, _ = spin_operators(1)

    ix_n1_full = embed_operator(
        operator=ix,
        site=1,
        system=system,
    )

    ix_n2_full = embed_operator(
        operator=ix,
        site=3,
        system=system,
    )

    # ------------------------------------------------------------------
    # 5. Construct the complete Joas control operator
    # ------------------------------------------------------------------

    gamma_ratio = 3.076272e-3 / (-28.02495)

    control = control_operator(
        electronic_x_operators=[
            sx_e1_full,
            sx_e2_full,
        ],
        nuclear_x_operators=[
            ix_n1_full,
            ix_n2_full,
        ],
        gamma_ratio=gamma_ratio,
    )

    # ------------------------------------------------------------------
    # 6. Validate the full operator
    # ------------------------------------------------------------------

    expected = (
        sx_e1_full
        + sx_e2_full
        + gamma_ratio * (
            ix_n1_full
            + ix_n2_full
        )
    )

    assert control.shape == (81, 81)
    assert control.isherm
    assert (control - expected).norm() < 1e-12


def test_joas_full_nv1_finite_microwave_pi_pulse():
    """A finite microwave pulse drives NV1 in the full electron-nuclear model."""

    # ------------------------------------------------------------------
    # 1. Define the full two-NV system
    # ------------------------------------------------------------------

    system = SpinSystem([1, 1, 1, 1])
    electron_system = SpinSystem([1])

    D_1 = 2 * np.pi * 2867.27  # NV1: misaligned, 2990.8 MHz
    D_2 = 2 * np.pi * 2865.42  # NV2: aligned, 2571.0 MHz

    Q = 2 * np.pi * (-4.945)

    A = 2 * np.pi * np.diag([
        -2.62,
        -2.62,
        -2.162,
    ])
    # --- Temporary diagnostic: disable hyperfine coupling ---
    # A = np.zeros((3, 3))

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
    # 2. Build the complete static Hamiltonian
    # ------------------------------------------------------------------

    h_nv = nv_register_hamiltonian(
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

    # ------------------------------------------------------------------
    # 3. Build the electronic dressed interaction
    # ------------------------------------------------------------------

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

    basis_e1 = dressed_spin_one_basis(h_e1)
    basis_e2 = dressed_spin_one_basis(h_e2)

    t_e1 = basis_unitary(basis_e1)
    t_e2 = basis_unitary(basis_e2)

    transformation = tensor(
        t_e1,
        qeye(3),
        t_e2,
        qeye(3),
    )

    _, _, sz = spin_operators(1)

    sz_e1_full = embed_operator(
        sz,
        site=0,
        system=system,
    )

    sz_e2_full = embed_operator(
        sz,
        site=2,
        system=system,
    )

    coupling = 2 * np.pi * 0.11261

    h_nv_transformed = (
        transformation.dag()
        * h_nv
        * transformation
    )

    h_free = (
        h_nv_transformed
        + coupling * sz_e1_full * sz_e2_full
    )

    # ------------------------------------------------------------------
    # 4. Construct the full microwave control operator
    # ------------------------------------------------------------------

    sx, _, _ = spin_operators(1)

    sx_e1_full = embed_operator(
        sx,
        site=0,
        system=system,
    )

    sx_e2_full = embed_operator(
        sx,
        site=2,
        system=system,
    )

    ix, _, _ = spin_operators(1)

    ix_n1_full = embed_operator(
        ix,
        site=1,
        system=system,
    )

    ix_n2_full = embed_operator(
        ix,
        site=3,
        system=system,
    )

    gamma_ratio = 3.076272e-3 / (-28.02495)

    control = control_operator(
        electronic_x_operators=[
            sx_e1_full,
            sx_e2_full,
        ],
        nuclear_x_operators=[
            ix_n1_full,
            ix_n2_full,
        ],
        gamma_ratio=gamma_ratio,
    )

    # ------------------------------------------------------------------
    # 5. Prepare the initial and target states
    # ------------------------------------------------------------------

    zero_e1 = basis(3, 1)
    one_e1 = basis(3, 0)

    zero_e2 = basis(3, 1)

    nuclear_zero = basis(3, 1)

    initial_state = tensor(
        zero_e1,
        nuclear_zero,
        zero_e2,
        nuclear_zero,
    )

    target_state = tensor(
        one_e1,
        nuclear_zero,
        zero_e2,
        nuclear_zero,
    )

    # --- Temporary diagnostic: hyperfine-shifted transition ---

    initial_energy = np.real(
        initial_state.dag()
        * h_free
        * initial_state
    )

    target_energy = np.real(
        target_state.dag()
        * h_free
        * target_state
    )

    hyperfine_frequency = target_energy - initial_energy

    print(
        "hyperfine-shifted NV1 frequency =",
        hyperfine_frequency / (2 * np.pi),
        "MHz",
    )

    print(
        "detuning from Joas carrier =",
        (
            2 * np.pi * 2990.8
            - hyperfine_frequency
        ) / (2 * np.pi),
        "MHz",
    )
    # ------------------------------------------------------------------
    # 6. Define the Joas carrier frequencies
    # ------------------------------------------------------------------

    omega_1 = 2 * np.pi * 2990.8
    omega_2 = 2 * np.pi * 2571.0

    # ------------------------------------------------------------------
    # 7. Construct the rotating-frame generator
    # ------------------------------------------------------------------

    h_trans = (
        omega_1 * sz_e1_full**2
        + omega_2 * sz_e2_full**2
    )

    # ------------------------------------------------------------------
    # 8. Define the finite sine pulse
    # ------------------------------------------------------------------

    pulse_duration = 0.1

    peak_amplitude = (
        np.pi**2
        / (2 * pulse_duration)
    )

    # ------------------------------------------------------------------
    # 9. Define the driven Hamiltonian
    # ------------------------------------------------------------------

    def hamiltonian(time):
        envelope = sine_envelope(
            time=time,
            duration=pulse_duration,
            peak_amplitude=peak_amplitude,
        )

        microwave = microwave_hamiltonian(
            time=time,
            control=control,
            drives=[
                {
                    "omega": omega_1,
                    "phase": 0.0,
                    "omega_x": envelope,
                    "omega_y": 0.0,
                }
            ],
        )

        return h_free + microwave

    # ------------------------------------------------------------------
    # 10. Transform to the Joas rotating frame
    # ------------------------------------------------------------------

    def hamiltonian_rotating(time):
        return rotating_frame_hamiltonian(
            time=time,
            driven_hamiltonian=hamiltonian,
            generator=h_trans,
        )

    # ------------------------------------------------------------------
    # 11. Propagate the complete 81D system
    # ------------------------------------------------------------------

    propagator = time_dependent_propagator(
        hamiltonian=hamiltonian_rotating,
        t0=0.0,
        t1=pulse_duration,
        steps=5000,
    )

    assert propagator.shape == (81, 81)
    assert propagator.isunitary

    # ------------------------------------------------------------------
    # 12. Measure the population transferred by the finite pulse
    # ------------------------------------------------------------------

    final_state = propagator * initial_state

    electron_target_population = 0.0

    for nuclear_state in range(3):
        state = tensor(
            one_e1,
            basis(3, nuclear_state),
            zero_e2,
            nuclear_zero,
        )

        population = abs(
            state.overlap(final_state)
        ) ** 2

        electron_target_population += population

        print(
            f"NV1 target electron, nuclear state {nuclear_state}:",
            population,
        )

    print(
        "NV1 electron target population =",
        electron_target_population,
    )

    target_population = abs(
        target_state.overlap(final_state)
    ) ** 2

    print(
        "rotating-frame NV1 target population =",
        target_population,
    )

    assert propagator.isunitary
    assert target_population > 0.95
    assert electron_target_population > 0.98

    # ------------------------------------------------------------------
    # 13. Repeat with maximally mixed nuclear initialization
    # ------------------------------------------------------------------

    electron_initial_state = tensor(
        zero_e1,
        zero_e2,
    )

    electron_initial_density = (
        electron_initial_state
        * electron_initial_state.dag()
    )

    nuclear_mixed_state = qeye(3) / 3

    final_electron_density = electron_nuclear_dynamical_map(
        density_matrix=electron_initial_density,
        propagator=propagator,
        nuclear_states=[
            nuclear_mixed_state,
            nuclear_mixed_state,
        ],
    )

    electron_target_state = tensor(
        one_e1,
        zero_e2,
    )

    mixed_nuclear_target_population = np.real(
        electron_target_state.dag()
        * final_electron_density
        * electron_target_state
    )

    print(
        "NV1 target population with maximally mixed nuclei =",
        mixed_nuclear_target_population,
    )

    print(
        "reduced electronic purity =",
        np.real(
            (final_electron_density * final_electron_density).tr()
        ),
    )

    assert final_electron_density.shape == (9, 9)
    assert np.isclose(final_electron_density.tr(), 1.0)
    assert mixed_nuclear_target_population > 0.90


def test_joas_setting_2_rotating_frame_generator():
    """Construct the rotating-frame generator used for Joas setting 2."""

    system = SpinSystem([1, 1, 1, 1])
    electron_system = SpinSystem([1])

    D_1 = 2 * np.pi * 2867.27  # NV1: misaligned, 2990.8 MHz
    D_2 = 2 * np.pi * 2865.42  # NV2: aligned, 2571.0 MHz

    omega_e = 2 * np.pi * 295.18

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

    # Electronic Hamiltonians
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

    # Electronic dressed bases
    basis_e1 = dressed_spin_one_basis(h_e1)
    basis_e2 = dressed_spin_one_basis(h_e2)

    _, _, sz_e1 = dressed_spin_operators(
        basis_e1,
        spin=1,
    )

    _, _, sz_e2 = dressed_spin_operators(
        basis_e2,
        spin=1,
    )

    # Embed electronic operators into
    # (e1, n1, e2, n2)
    sz_e1_full = embed_operator(
        sz_e1,
        site=0,
        system=system,
    )

    sz_e2_full = embed_operator(
        sz_e2,
        site=2,
        system=system,
    )

    # Joas setting-2 addressed transitions
    omega_1 = 2 * np.pi * 2990.8
    omega_2 = 2 * np.pi * 2571.0

    h_trans = (
        omega_1 * sz_e1_full**2
        + omega_2 * sz_e2_full**2
    )

    expected = (
        omega_1 * (sz_e1_full * sz_e1_full)
        + omega_2 * (sz_e2_full * sz_e2_full)
    )

    assert h_trans.shape == (81, 81)
    assert h_trans.isherm
    assert (h_trans - expected).norm() < 1e-12


def test_joas_full_electronic_basis_transformation():
    """The electronic basis transformation diagonalizes both NV electron Hamiltonians."""

    system = SpinSystem([1, 1, 1, 1])
    electron_system = SpinSystem([1])

    D_1 = 2 * np.pi * 2867.27  # NV1: misaligned, 2990.8 MHz
    D_2 = 2 * np.pi * 2865.42  # NV2: aligned, 2571.0 MHz

    omega_e = 2 * np.pi * 295.18

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

    basis_e1 = dressed_spin_one_basis(h_e1)
    basis_e2 = dressed_spin_one_basis(h_e2)

    t_e1 = basis_unitary(basis_e1)
    t_e2 = basis_unitary(basis_e2)

    identity_n = qeye(3)

    transformation = tensor(
        t_e1,
        identity_n,
        t_e2,
        identity_n,
    )

    # Electronic Hamiltonian embedded in the complete Hilbert space
    h_e1_full = embed_operator(
        h_e1,
        site=0,
        system=system,
    )

    h_e2_full = embed_operator(
        h_e2,
        site=2,
        system=system,
    )

    h_e = h_e1_full + h_e2_full

    h_e_transformed = (
        transformation.dag()
        * h_e
        * transformation
    )

    matrix = h_e_transformed.full()

    off_diagonal = (
        matrix
        - np.diag(np.diag(matrix))
    )

    assert transformation.shape == (81, 81)
    assert transformation.isunitary
    assert np.max(np.abs(off_diagonal)) < 1e-10


def test_joas_nv1_isolated_finite_microwave_pi_pulse():
    """A Joas sine pulse drives the isolated NV1 electronic qubit."""

    # --------------------------------------------------------------
    # 1. Electronic NV1
    # --------------------------------------------------------------

    system = SpinSystem([1])

    D_1 = 2 * np.pi * 2867.27
    omega_e = 2 * np.pi * 295.18
    theta_1 = np.deg2rad(74.08)

    omega_e_1 = omega_e * np.array([
        np.sin(theta_1),
        0.0,
        np.cos(theta_1),
    ])

    h_e1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=D_1,
        omega_e=omega_e_1,
    )

    # --------------------------------------------------------------
    # 2. Transform to electronic eigenbasis
    # --------------------------------------------------------------

    dressed_basis = dressed_spin_one_basis(h_e1)
    transformation = basis_unitary(dressed_basis)

    h_free = (
        transformation.dag()
        * h_e1
        * transformation
    )

    # In this basis:
    # |0> = ground / m_s=0-like state
    # |1> = addressed +1-like state
    zero = basis(3, 1)
    one = basis(3, 0)

    # --- Temporary diagnostic ---

    omega_1 = 2 * np.pi * 2990.8

    energy_zero = np.real(
        zero.dag() * h_free * zero
    )

    energy_one = np.real(
        one.dag() * h_free * one
    )

    model_frequency = energy_one - energy_zero

    # omega_1 = model_frequency

    print(
        "isolated NV1 model frequency =",
        model_frequency / (2 * np.pi),
        "MHz",
    )

    print(
        "carrier detuning =",
        (omega_1 - model_frequency) / (2 * np.pi),
        "MHz",
    )

    # --------------------------------------------------------------
    # 3. Joas control and carrier
    # --------------------------------------------------------------

    sx, _, sz = spin_operators(1)

    h_trans = omega_1 * sz**2

    pulse_duration = 0.1

    peak_amplitude = (
        np.pi**2
        / (2 * pulse_duration)
    )

    # --------------------------------------------------------------
    # 4. Driven Hamiltonian
    # --------------------------------------------------------------

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
                    "omega": omega_1,
                    "phase": 0.0,
                    "omega_x": envelope,
                    "omega_y": 0.0,
                }
            ],
        )

        return h_free + microwave

    # --------------------------------------------------------------
    # 5. Joas rotating frame
    # --------------------------------------------------------------

    def hamiltonian_rotating(time):
        return rotating_frame_hamiltonian(
            time=time,
            driven_hamiltonian=hamiltonian,
            generator=h_trans,
        )

    # --------------------------------------------------------------
    # 6. Propagation
    # --------------------------------------------------------------

    propagator = time_dependent_propagator(
        hamiltonian=hamiltonian_rotating,
        t0=0.0,
        t1=pulse_duration,
        steps=5000,
    )

    final_state = propagator * zero

    target_population = abs(
        one.overlap(final_state)
    ) ** 2

    print(
        "isolated NV1 target population =",
        target_population,
    )

    assert propagator.isunitary


def test_joas_nv2_isolated_finite_microwave_pi_pulse():
    """A Joas sine pulse drives the isolated NV2 electronic qubit."""

    system = SpinSystem([1])

    D_2 = 2 * np.pi * 2865.42
    omega_e = 2 * np.pi * 295.18
    theta_2 = np.deg2rad(3.58)

    omega_e_2 = omega_e * np.array([
        np.sin(theta_2),
        0.0,
        np.cos(theta_2),
    ])

    h_e2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=D_2,
        omega_e=omega_e_2,
    )

    dressed_basis = dressed_spin_one_basis(h_e2)
    transformation = basis_unitary(dressed_basis)

    h_free = transformation.dag() * h_e2 * transformation

    zero = basis(3, 1)
    one = basis(3, 2)  # -1-like branch used for NV2

    # --- Temporary diagnostic ---

    omega_2 = 2 * np.pi * 2571.0

    energy_zero = np.real(
        zero.dag() * h_free * zero
    )

    energy_one = np.real(
        one.dag() * h_free * one
    )

    model_frequency = energy_one - energy_zero

    print(
        "isolated NV2 model frequency =",
        model_frequency / (2 * np.pi),
        "MHz",
    )

    print(
        "carrier detuning =",
        (omega_2 - model_frequency) / (2 * np.pi),
        "MHz",
    )

    sx, _, sz = spin_operators(1)

    h_trans = omega_2 * sz**2

    pulse_duration = 0.1

    peak_amplitude = (
        np.pi**2
        / (2 * pulse_duration)
    )

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
                    "omega": omega_2,
                    "phase": 0.0,
                    "omega_x": envelope,
                    "omega_y": 0.0,
                }
            ],
        )

        return h_free + microwave

    def hamiltonian_rotating(time):
        return rotating_frame_hamiltonian(
            time=time,
            driven_hamiltonian=hamiltonian,
            generator=h_trans,
        )

    propagator = time_dependent_propagator(
        hamiltonian=hamiltonian_rotating,
        t0=0.0,
        t1=pulse_duration,
        steps=5000,
    )

    final_state = propagator * zero

    target_population = abs(
        one.overlap(final_state)
    ) ** 2

    print(
        "isolated NV2 target population =",
        target_population,
    )

    assert propagator.isunitary

# -----------------------------------------------------------------

def test_joas_full_finite_xy8_propagator():
    """Validate the full finite-pulse Joas XY8 entangling gate."""

    # ------------------------------------------------------------------
    # 1. Full two-NV system
    # ------------------------------------------------------------------

    system = SpinSystem([1, 1, 1, 1])
    electron_system = SpinSystem([1])

    D_1 = 2 * np.pi * 2867.27
    D_2 = 2 * np.pi * 2865.42

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
    # 2. Full static Hamiltonian
    # ------------------------------------------------------------------

    h_nv = nv_register_hamiltonian(
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

    # ------------------------------------------------------------------
    # 3. Transform to the electronic eigenbasis
    # ------------------------------------------------------------------

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

    basis_e1 = dressed_spin_one_basis(h_e1)
    basis_e2 = dressed_spin_one_basis(h_e2)

    t_e1 = basis_unitary(basis_e1)
    t_e2 = basis_unitary(basis_e2)

    transformation = tensor(
        t_e1,
        qeye(3),
        t_e2,
        qeye(3),
    )

    h_nv_transformed = (
        transformation.dag()
        * h_nv
        * transformation
    )

    # ------------------------------------------------------------------
    # 4. Effective NV-NV interaction
    # ------------------------------------------------------------------

    sx, _, sz = spin_operators(1)

    sz_e1_full = embed_operator(
        sz,
        site=0,
        system=system,
    )

    sz_e2_full = embed_operator(
        sz,
        site=2,
        system=system,
    )

    nu_dip = 0.11261
    coupling = 2 * np.pi * nu_dip

    h_free = (
        h_nv_transformed
        + coupling * sz_e1_full * sz_e2_full
    )

    # ------------------------------------------------------------------
    # 5. Full microwave control operator
    # ------------------------------------------------------------------

    sx_e1_full = embed_operator(
        sx,
        site=0,
        system=system,
    )

    sx_e2_full = embed_operator(
        sx,
        site=2,
        system=system,
    )

    ix, _, _ = spin_operators(1)

    ix_n1_full = embed_operator(
        ix,
        site=1,
        system=system,
    )

    ix_n2_full = embed_operator(
        ix,
        site=3,
        system=system,
    )

    gamma_ratio = 3.076272e-3 / (-28.02495)

    control = control_operator(
        electronic_x_operators=[
            sx_e1_full,
            sx_e2_full,
        ],
        nuclear_x_operators=[
            ix_n1_full,
            ix_n2_full,
        ],
        gamma_ratio=gamma_ratio,
    )

    # ------------------------------------------------------------------
    # 6. Joas rotating frame
    # ------------------------------------------------------------------

    omega_1 = 2 * np.pi * 2990.8
    omega_2 = 2 * np.pi * 2571.0

    h_trans = (
        omega_1 * sz_e1_full**2
        + omega_2 * sz_e2_full**2
    )

    # ------------------------------------------------------------------
    # 7. XY8 timing and microwave pulse
    # ------------------------------------------------------------------

    n_pi = 8

    # Optimized Joas pulse spacing.
    tau_1 = 0.8  # microseconds

    # Ideal interaction time for the measured dipolar coupling.
    tau_2 = 1 / (
        4 * n_pi * nu_dip
    )

    schedule = xy8_gate_schedule(
        tau_1=tau_1,
        tau_2=tau_2,
    )

    gate_duration = n_pi * tau_1

    # Optimized Joas microwave Rabi frequency.
    rabi_frequency = 23.7  # MHz
    pulse_duration = 1 / (2 * rabi_frequency)

    peak_amplitude = (
        np.pi**2
        / (2 * pulse_duration)
    )

    # ------------------------------------------------------------------
    # 8. Finite pulse propagators
    # ------------------------------------------------------------------

    pulse_propagators = []
    pulse_centers = []

    # 5000 steps over ~21 ns gives a finer resolution than
    # the 100 steps/ns used in the Joas numerical simulation.
    pulse_steps = 5000

    for center, nv, axis in schedule:
        frequency = (
            omega_1
            if nv == 1
            else omega_2
        )

        phase = (
            0.0
            if axis == "x"
            else np.pi / 2
        )

        start_time = center - pulse_duration / 2

        def pulse_hamiltonian(time):
            envelope = centered_sine_envelope(
                time=time,
                center=center,
                duration=pulse_duration,
                peak_amplitude=peak_amplitude,
            )

            drives = [{
                "omega": frequency,
                "phase": phase,
                "omega_x": envelope,
                "omega_y": 0.0,
            }]

            microwave = microwave_hamiltonian(
                time=time,
                control=control,
                drives=drives,
            )

            return h_free + microwave

        pulse_propagator = time_dependent_propagator(
            hamiltonian=pulse_hamiltonian,
            t0=start_time,
            t1=start_time + pulse_duration,
            steps=pulse_steps,
        )

        pulse_propagators.append(
            pulse_propagator
        )
        pulse_centers.append(center)

    # ------------------------------------------------------------------
    # 9. Exact free-evolution intervals
    # ------------------------------------------------------------------

    free_intervals = finite_pulse_intervals(
        pulse_centers=pulse_centers,
        pulse_duration=pulse_duration,
        total_duration=gate_duration,
    )

    # ------------------------------------------------------------------
    # 10. Complete laboratory-frame sequence
    # ------------------------------------------------------------------

    propagator_lab = finite_pulse_sequence_propagator(
        free_hamiltonian=h_free,
        durations=free_intervals,
        pulse_propagators=pulse_propagators,
    )

    propagator = rotating_frame_propagator(
        propagator=propagator_lab,
        generator=h_trans,
        time=gate_duration,
    )

    assert propagator.shape == (81, 81)

    # ------------------------------------------------------------------
    # 11. Logical basis and logical-state survival
    # ------------------------------------------------------------------

    logical_electron_basis = [
        tensor(basis(3, 1), basis(3, 1)),  # |00>
        tensor(basis(3, 1), basis(3, 2)),  # |01>
        tensor(basis(3, 0), basis(3, 1)),  # |10>
        tensor(basis(3, 0), basis(3, 2)),  # |11>
    ]

    logical_states = [
        tensor(basis(2, 0), basis(2, 0)),
        tensor(basis(2, 0), basis(2, 1)),
        tensor(basis(2, 1), basis(2, 0)),
        tensor(basis(2, 1), basis(2, 1)),
    ]

    nuclear_state = qeye(3) / 3

    survivals = []

    for state in logical_states:
        result = logical_electron_nuclear_dynamical_map(
            density_matrix=state.proj(),
            propagator=propagator,
            logical_basis=logical_electron_basis,
            nuclear_states=[
                nuclear_state,
                nuclear_state,
            ],
        )

        survival = float(np.real(result.tr()))
        survivals.append(survival)

    print("logical survivals =", survivals)

    assert all(
        0.0 <= survival <= 1.0 + 1e-10
        for survival in survivals
    )

    # ------------------------------------------------------------------
    # 12. Average logical gate fidelity
    # ------------------------------------------------------------------

    def logical_dynamical_map(density_matrix):
        return logical_electron_nuclear_dynamical_map(
            density_matrix=density_matrix,
            propagator=propagator,
            logical_basis=logical_electron_basis,
            nuclear_states=[
                nuclear_state,
                nuclear_state,
            ],
        )

    target = sqrt_zz_gate().dag()

    fidelity = average_gate_fidelity(
        dynamical_map=logical_dynamical_map,
        target=target,
        basis_states=logical_states,
    )

    print("full Joas average gate fidelity =", fidelity)

    assert 0.0 <= fidelity <= 1.0

    # ------------------------------------------------------------------
    # 13. Nuclear-state-resolved fidelity diagnostic
    # ------------------------------------------------------------------

    nuclear_states_to_test = {
        "mI=+1": basis(3, 0).proj(),
        "mI=0": basis(3, 1).proj(),
        "mI=-1": basis(3, 2).proj(),
        "mixed": qeye(3) / 3,
    }

    for label, nuclear_test_state in nuclear_states_to_test.items():

        def nuclear_dynamical_map(density_matrix):
            return logical_electron_nuclear_dynamical_map(
                density_matrix=density_matrix,
                propagator=propagator,
                logical_basis=logical_electron_basis,
                nuclear_states=[
                    nuclear_test_state,
                    nuclear_test_state,
                ],
            )

        nuclear_fidelity = average_gate_fidelity(
            dynamical_map=nuclear_dynamical_map,
            target=target,
            basis_states=logical_states,
        )

        print(
            f"{label} nuclear fidelity =",
            nuclear_fidelity,
        )
