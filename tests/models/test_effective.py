import numpy as np
import pytest
from qutip import Qobj, basis, tensor

from echo_spin.models.effective import (
    projected_unitary_kraus,
    two_qubit_hamiltonian_reduction,
)


def logical_basis():
    """Return two embedded qubits in a pair of three-level systems."""
    return [
        tensor(basis(3, first), basis(3, second))
        for first in [0, 1]
        for second in [0, 1]
    ]


def test_two_qubit_reduction_invariant_subspace():
    """Recover Pauli coefficients and exact dynamics in an invariant subspace."""
    states = logical_basis()
    energies = [0.3, -0.7, 1.1, 2.5]
    hamiltonian = sum(
        energy * state.proj()
        for energy, state in zip(energies, states)
    )

    result = two_qubit_hamiltonian_reduction(hamiltonian, states)
    coefficients = result["pauli_coefficients"]

    assert np.isclose(coefficients["II"], 0.8)
    assert np.isclose(coefficients["ZI"], -1.0)
    assert np.isclose(coefficients["IZ"], -0.1)
    assert np.isclose(coefficients["ZZ"], 0.6)
    assert result["discarded_coupling_norm"] < 1e-12

    embedding = np.column_stack([
        state.full().ravel()
        for state in states
    ])
    time = 0.37

    projected_evolution = (
        embedding.conj().T
        @ (-1j * hamiltonian * time).expm().full()
        @ embedding
    )
    reduced_evolution = (
        -1j * result["hamiltonian"] * time
    ).expm().full()

    np.testing.assert_allclose(
        projected_evolution,
        reduced_evolution,
        atol=1e-12,
        rtol=0.0,
    )


def test_two_qubit_reduction_detects_discarded_coupling():
    """Detect a transition invisible to the projected Hamiltonian."""
    states = logical_basis()
    outside = tensor(basis(3, 2), basis(3, 0))
    coupling = 0.2

    hamiltonian = coupling * (
        outside * states[0].dag()
        + states[0] * outside.dag()
    )

    result = two_qubit_hamiltonian_reduction(hamiltonian, states)

    assert result["hamiltonian"].norm() < 1e-12
    assert np.isclose(result["discarded_coupling_norm"], coupling)

    time = 1.0
    evolved = (-1j * hamiltonian * time).expm() * states[0]
    survival = sum(
        abs(state.overlap(evolved)) ** 2
        for state in states
    )

    assert np.isclose(survival, np.cos(coupling * time) ** 2)


def test_two_qubit_reduction_rejects_duplicate_states():
    """Reject a nonorthogonal logical basis."""
    states = logical_basis()
    hamiltonian = states[0].proj()

    with pytest.raises(ValueError, match="orthonormal"):
        two_qubit_hamiltonian_reduction(
            hamiltonian,
            [states[0]] * 4,
        )


def test_joas_electron_reduction_in_rotating_frame():
    """Recover the Joas logical ZZ coupling and exact undriven electron evolution."""
    from echo_spin.core.basis import (
        dressed_spin_one_basis,
        operator_in_basis,
    )
    from echo_spin.core.operators import spin_operators
    from echo_spin.core.system import SpinSystem
    from echo_spin.nv.hamiltonians import electronic_nv_hamiltonian

    system = SpinSystem([1])
    _, _, sz = spin_operators(1)

    local_hamiltonians = []

    for splitting, angle_degrees in [
        (2867.27, 74.08),
        (2865.42, 3.58),
    ]:
        angle = np.deg2rad(angle_degrees)

        physical = electronic_nv_hamiltonian(
            system=system,
            electron_site=0,
            D=2 * np.pi * splitting,
            omega_e=2 * np.pi * 295.18 * np.array([
                np.sin(angle),
                0.0,
                np.cos(angle),
            ]),
        )

        local_hamiltonians.append(
            operator_in_basis(
                operator=physical,
                basis_states=dressed_spin_one_basis(physical),
            )
        )

    # Canonical dressed coordinates are ordered +1-like, 0-like, -1-like.
    identity = Qobj(np.eye(3))
    carrier_1 = 2 * np.pi * 2990.8
    carrier_2 = 2 * np.pi * 2571.0
    coupling = 2 * np.pi * 0.11261

    rotating_hamiltonian = (
        tensor(local_hamiltonians[0] - carrier_1 * sz**2, identity)
        + tensor(identity, local_hamiltonians[1] - carrier_2 * sz**2)
        + coupling * tensor(sz, sz)
    )

    states = [
        tensor(basis(3, first), basis(3, second))
        for first in [1, 0]   # NV1: ground-like, +1-like.
        for second in [1, 2]  # NV2: ground-like, -1-like.
    ]

    result = two_qubit_hamiltonian_reduction(
        rotating_hamiltonian,
        states,
    )
    coefficients = result["pauli_coefficients"]

    delta_1 = float(
        (
            local_hamiltonians[0][0, 0]
            - local_hamiltonians[0][1, 1]
        ).real
    ) - carrier_1

    delta_2 = float(
        (
            local_hamiltonians[1][2, 2]
            - local_hamiltonians[1][1, 1]
        ).real
    ) - carrier_2

    # NV1 Sz -> (I-Z)/2; NV2 Sz -> -(I-Z)/2.
    assert np.isclose(coefficients["ZZ"], -coupling / 4)
    assert np.isclose(coefficients["ZI"], -delta_1 / 2 + coupling / 4)
    assert np.isclose(coefficients["IZ"], -delta_2 / 2 + coupling / 4)

    for label, value in coefficients.items():
        if label not in {"II", "ZI", "IZ", "ZZ"}:
            assert abs(value) < 1e-9

    assert result["discarded_coupling_norm"] < 1e-9

    embedding = np.column_stack([
        state.full().ravel()
        for state in states
    ])

    for time in [0.1, 0.8, 6.4]:
        full_evolution = (
            -1j * rotating_hamiltonian * time
        ).expm().full()

        reduced_evolution = (
            -1j * result["hamiltonian"] * time
        ).expm().full()

        np.testing.assert_allclose(
            embedding.conj().T @ full_evolution @ embedding,
            reduced_evolution,
            atol=1e-9,
            rtol=0.0,
        )

    print(
        "\nLogical ZZ coefficient / (2*pi) =",
        coefficients["ZZ"] / (2 * np.pi),
        "MHz",
    )
    print(
        "Discarded coupling norm =",
        result["discarded_coupling_norm"],
    )

def test_joas_shared_control_reduction_detects_leakage():
    """Recover logical microwave amplitudes while detecting discarded-level coupling."""
    from echo_spin.core.operators import spin_operators

    sx, _, _ = spin_operators(1)
    identity = Qobj(np.eye(3))

    states = [
        tensor(basis(3, first), basis(3, second))
        for first in [1, 0]
        for second in [1, 2]
    ]

    shared_control = (
        tensor(sx, identity)
        + tensor(identity, sx)
    )

    result = two_qubit_hamiltonian_reduction(
        shared_control,
        states,
    )

    coefficients = result["pauli_coefficients"]

    assert np.isclose(coefficients["XI"], 1 / np.sqrt(2))
    assert np.isclose(coefficients["IX"], 1 / np.sqrt(2))

    for label, value in coefficients.items():
        if label not in {"XI", "IX"}:
            assert abs(value) < 1e-12

    assert np.isclose(result["discarded_coupling_norm"], 1.0)


def test_joas_selective_control_reduction_preserves_dynamics():
    """Preserve logical dynamics when controls exclude the discarded transitions."""
    identity = Qobj(np.eye(3))

    nv1_transition = basis(3, 0) * basis(3, 1).dag()
    nv2_transition = basis(3, 2) * basis(3, 1).dag()

    control_1 = (
        nv1_transition + nv1_transition.dag()
    ) / np.sqrt(2)

    control_2 = (
        nv2_transition + nv2_transition.dag()
    ) / np.sqrt(2)

    selective_control = (
        tensor(control_1, identity)
        + tensor(identity, control_2)
    )

    states = [
        tensor(basis(3, first), basis(3, second))
        for first in [1, 0]
        for second in [1, 2]
    ]

    result = two_qubit_hamiltonian_reduction(
        selective_control,
        states,
    )

    assert result["discarded_coupling_norm"] < 1e-12

    embedding = np.column_stack([
        state.full().ravel()
        for state in states
    ])

    # The parameter is integrated control amplitude, not a calibrated pulse duration.
    for pulse_area in [0.2, np.pi / np.sqrt(2)]:
        full_evolution = (
            -1j * selective_control * pulse_area
        ).expm().full()

        reduced_evolution = (
            -1j * result["hamiltonian"] * pulse_area
        ).expm().full()

        np.testing.assert_allclose(
            embedding.conj().T @ full_evolution @ embedding,
            reduced_evolution,
            atol=1e-12,
            rtol=0.0,
        )


def test_projected_unitary_kraus_environment_dephasing():
    """Recover dephasing induced by tracing out an initially mixed environment."""
    from qutip import qeye, sigmax, sigmaz

    angle = 0.3
    propagator = (
        -1j * angle * tensor(sigmaz(), sigmaz())
    ).expm()

    embeddings = [
        np.kron(np.eye(2), basis(2, index).full())
        for index in [0, 1]
    ]

    kraus = projected_unitary_kraus(
        propagator,
        embeddings,
        environment_probabilities=[0.5, 0.5],
        logical_dimensions=(2,),
    )

    plus = (basis(2, 0) + basis(2, 1)).unit()
    density = plus.proj()
    output = sum(
        operator * density * operator.dag()
        for operator in kraus
    )

    completeness = sum(
        operator.dag() * operator
        for operator in kraus
    )

    np.testing.assert_allclose(
        completeness.full(),
        qeye(2).full(),
        atol=1e-12,
        rtol=0.0,
    )
    assert np.isclose(
        (output * sigmax()).tr().real,
        np.cos(2 * angle),
    )


def test_projected_unitary_kraus_preserves_leakage():
    """Keep logical population loss without renormalizing the reduced channel."""
    angle = 0.4
    transition = basis(3, 2) * basis(3, 1).dag()
    propagator = (
        -1j * angle * (transition + transition.dag())
    ).expm()

    embedding = np.column_stack([
        basis(3, 0).full().ravel(),
        basis(3, 1).full().ravel(),
    ])

    kraus = projected_unitary_kraus(
        propagator,
        [embedding],
        environment_probabilities=[1.0],
        logical_dimensions=(2,),
    )

    density = basis(2, 1).proj()
    output = sum(
        operator * density * operator.dag()
        for operator in kraus
    )

    assert np.isclose(output.tr().real, np.cos(angle) ** 2)


def test_reduced_channel_composition_discards_environment_memory():
    """Distinguish repeated joint evolution from a freshly reset environment channel."""
    from qutip import qeye, sigmax, sigmaz

    angle = 0.3
    propagator = (
        -1j * angle * tensor(sigmaz(), sigmaz())
    ).expm()

    embeddings = [
        np.kron(np.eye(2), basis(2, index).full())
        for index in [0, 1]
    ]

    kraus = projected_unitary_kraus(
        propagator,
        embeddings,
        environment_probabilities=[0.5, 0.5],
        logical_dimensions=(2,),
    )

    plus = (basis(2, 0) + basis(2, 1)).unit()
    initial = plus.proj()

    def apply_channel(density):
        """Apply the extracted single-operation logical channel."""
        return sum(
            operator * density * operator.dag()
            for operator in kraus
        )

    reset_output = apply_channel(apply_channel(initial))

    joint_initial = tensor(initial, qeye(2) / 2)
    repeated = propagator * propagator
    joint_output = repeated * joint_initial * repeated.dag()
    retained_output = joint_output.ptrace(0)

    reset_coherence = float((reset_output * sigmax()).tr().real)
    retained_coherence = float((retained_output * sigmax()).tr().real)

    assert np.isclose(reset_coherence, np.cos(2 * angle) ** 2)
    assert np.isclose(retained_coherence, np.cos(4 * angle))
    assert not np.isclose(reset_coherence, retained_coherence)

    print(f"\nReset-environment coherence = {reset_coherence:.8f}")
    print(f"Retained-environment coherence = {retained_coherence:.8f}")
