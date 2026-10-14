import numpy as np
import pytest
from qutip import Qobj, basis, tensor

from echo_spin.models.effective import two_qubit_hamiltonian_reduction


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
