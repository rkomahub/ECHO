import numpy as np
import pytest
from qutip import Qobj, basis, qeye, tensor

from echo_spin.control.rotations import selective_rotation
from echo_spin.core.basis import (
    dressed_spin_one_basis,
    dressed_spin_operators,
    operator_in_basis,
    project_operator,
)
from echo_spin.core.operators import spin_operators
from echo_spin.core.system import SpinSystem
from echo_spin.dynamics.sequences import finite_pulse_sequence_propagator
from echo_spin.gates.gates import sqrt_zz_gate
from echo_spin.models.joas import (
    electron_logical_states,
    two_electron_hamiltonian,
    two_electron_logical_basis,
    xy8_gate_intervals,
    xy8_gate_schedule,
)
from echo_spin.nv.frames import rotate_to_local_frame
from echo_spin.nv.hamiltonians import electronic_nv_hamiltonian
from echo_spin.spectroscopy.transitions import allowed_transitions


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

# ----------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------

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

    # NV1 = center B = target, strongly misaligned
    theta_nv1 = np.deg2rad(74.08)
    omega_nv1 = omega * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv1,
    )

    # NV2 = center A = control, approximately aligned
    theta_nv2 = np.deg2rad(3.58)
    omega_nv2 = omega * np.array([
        np.sin(theta_nv2),
        0.0,
        np.cos(theta_nv2),
    ])

    h_nv2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
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
        [2827.27932714, 2990.82058371],
    )

    assert np.allclose(
        nv2_transitions,
        [2571.00190538, 3160.19756161],
    )

    # Validate local Sx control strengths in the noninteracting register.
    sx, _, _ = spin_operators(1)
    local_hamiltonians = [h_nv1, h_nv2]
    register_controls = [
        tensor(sx, qeye(3)),
        tensor(qeye(3), sx),
    ]
    local_lines = []

    for h_local, control in zip(
        local_hamiltonians,
        register_controls,
    ):
        lines = allowed_transitions(h_local, sx)
        local_lines.append(lines)

        register_lines = allowed_transitions(h_two, control)

        # Each single-NV transition repeats for three spectator states.
        expected_lines = sorted(
            (float(frequency), float(strength))
            for frequency, strength, _, _ in lines
            for _ in range(3)
        )
        actual_lines = sorted(
            (float(frequency), float(strength))
            for frequency, strength, _, _ in register_lines
        )

        assert len(actual_lines) == len(expected_lines)
        assert np.allclose(
            np.asarray(actual_lines)[:, 0],
            np.asarray(expected_lines)[:, 0],
            atol=1e-8,
            rtol=0.0,
        )
        assert np.allclose(
            np.asarray(actual_lines)[:, 1],
            np.asarray(expected_lines)[:, 1],
            atol=1e-12,
            rtol=0.0,
        )

        # Independent sum rule for transitions out of the ground state:
        # sum_j |<j|Sx|0>|^2 = <Sx^2> - |<Sx>|^2.
        _, states = h_local.eigenstates()
        ground_state = states[0]
        expected_strength_sum = (
            ground_state.overlap(sx**2 * ground_state).real
            - abs(ground_state.overlap(sx * ground_state))**2
        )
        actual_strength_sum = sum(
            strength
            for _, strength, i, _ in lines
            if i == 0
        )

        assert actual_strength_sum == pytest.approx(
            expected_strength_sum,
            abs=1e-9,
        )

    # Include all Sx-allowed single-NV lines as possible unwanted lines.
    # This is a spectral check, without population weighting.
    addressed_pairs = [(0, 2), (0, 1)]
    expected_separations_mhz = [163.54125657, 256.27742176]

    for nv, (pair, expected_separation) in enumerate(
        zip(addressed_pairs, expected_separations_mhz)
    ):
        addressed = [
            frequency
            for frequency, _, i, j in local_lines[nv]
            if (i, j) == pair
        ]
        assert len(addressed) == 1
        target_frequency = addressed[0]

        unwanted = [
            frequency
            for other_nv, lines in enumerate(local_lines)
            for frequency, _, i, j in lines
            if other_nv != nv or (i, j) != pair
        ]

        separation_mhz = min(
            abs(frequency - target_frequency)
            for frequency in unwanted
        ) / (2 * np.pi)

        assert separation_mhz == pytest.approx(
            expected_separation,
            abs=1e-6,
        )


def test_joas_setting_2_dressed_spin_operators():
    """The dressed spin operators are consistent with the spin-one operators."""
    system = SpinSystem([1])

    omega = 2 * np.pi * 295.18

    # NV1 = center B = target, strongly misaligned
    theta_nv1 = np.deg2rad(74.08)
    omega_nv1 = omega * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv1,
    )

    # NV2 = center A = control, approximately aligned
    theta_nv2 = np.deg2rad(3.58)
    omega_nv2 = omega * np.array([
        np.sin(theta_nv2),
        0.0,
        np.cos(theta_nv2),
    ])

    h_nv2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
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

    # NV1 = center B = target, misaligned
    theta_nv1 = np.deg2rad(74.08)
    omega_nv1 = omega * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv1,
    )

    # NV2 = center A = control, aligned
    theta_nv2 = np.deg2rad(3.58)
    omega_nv2 = omega * np.array([
        np.sin(theta_nv2),
        0.0,
        np.cos(theta_nv2),
    ])

    h_nv2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
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

    # NV1 = center B = target, misaligned
    theta_nv1 = np.deg2rad(74.08)
    omega_nv1 = omega * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    h_nv1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv1,
    )

    # NV2 = center A = control, aligned
    theta_nv2 = np.deg2rad(3.58)
    omega_nv2 = omega * np.array([
        np.sin(theta_nv2),
        0.0,
        np.cos(theta_nv2),
    ])

    h_nv2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2 * np.pi * 2865.42,
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

# ----------------------------------------------------------------------------------

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
        D=2 * np.pi * 2867.27, # NV1 = misaligned target, +1-like transition ~2990.8 MHz
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
        D=2 * np.pi * 2865.42, # NV2 = nearly aligned control, -1-like transition ~2571.0 MHz
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

# ----------------------------------------------------------------------------------

def test_noninteracting_nv_local_control_and_spectral_separation():
    """Local drives retain their own lines across all spectator states."""
    system = SpinSystem([1])

    h1 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2.87,
        omega_e=(0.0, 0.0, 0.3),
    )
    h2 = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=3.11,
        omega_e=(0.0, 0.0, 0.17),
    )

    zero_interaction = 0 * tensor(qeye(3), qeye(3))
    register = two_electron_hamiltonian(h1, h2, zero_interaction)

    sx, _, _ = spin_operators(1)
    control_1 = tensor(sx, qeye(3))
    control_2 = tensor(qeye(3), sx)

    expected_lines = [
        np.array([2.57, 3.17]),
        np.array([2.94, 3.28]),
    ]

    for control, lines in zip(
        [control_1, control_2],
        expected_lines,
    ):
        transitions = allowed_transitions(register, control)

        # Two local lines, each repeated for three spectator states.
        assert len(transitions) == 6
        assert np.allclose(
            sorted(item[0] for item in transitions),
            np.repeat(lines, 3),
            atol=1e-12,
            rtol=0.0,
        )
        assert np.allclose(
            [item[1] for item in transitions],
            np.full(6, 0.5),
            atol=1e-12,
            rtol=0.0,
        )

    global_transitions = allowed_transitions(
        register, control_1 + control_2
    )
    expected_global = np.sort(
        np.concatenate([
            np.repeat(expected_lines[0], 3),
            np.repeat(expected_lines[1], 3),
        ])
    )

    assert len(global_transitions) == 12
    assert np.allclose(
        sorted(item[0] for item in global_transitions),
        expected_global,
        atol=1e-12,
        rtol=0.0,
    )

    cross_nv_separation = np.min(
        np.abs(
            expected_lines[0][:, None]
            - expected_lines[1][None, :]
        )
    )
    assert cross_nv_separation == pytest.approx(0.11)
