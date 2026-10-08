import numpy as np
import pytest
from qutip import Qobj, basis, qeye, sigmax, sigmaz, tensor

from echo_spin.dynamics.propagators import (
    electron_nuclear_dynamical_map,
    logical_electron_nuclear_dynamical_map,
    reduced_dynamical_map,
    rotating_frame_hamiltonian,
    rotating_frame_propagator,
    static_propagator,
    time_dependent_propagator,
    unitary_dynamical_map,
)


def test_static_propagator_at_zero_time():
    """Check that zero-time evolution gives the identity."""
    hamiltonian = Qobj([[1.0, 0.0], [0.0, -1.0]])

    propagator = static_propagator(
        hamiltonian=hamiltonian,
        time=0.0,
    )

    assert np.allclose(
        propagator.full(),
        qeye(2).full(),
    )


def test_static_propagator_diagonal_hamiltonian():
    """Check static propagation against an analytically solvable Hamiltonian."""
    hamiltonian = Qobj([[1.0, 0.0], [0.0, -1.0]])
    time = 0.4

    propagator = static_propagator(
        hamiltonian=hamiltonian,
        time=time,
    )

    expected = Qobj([
        [np.exp(-1j * time), 0.0],
        [0.0, np.exp(1j * time)],
    ])

    assert np.allclose(
        propagator.full(),
        expected.full(),
    )


def test_static_propagator_is_unitary():
    """Check that Hermitian static evolution produces a unitary propagator."""
    hamiltonian = Qobj([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    propagator = static_propagator(
        hamiltonian=hamiltonian,
        time=0.7,
    )

    identity = propagator.dag() * propagator

    assert np.allclose(
        identity.full(),
        qeye(2).full(),
    )


def test_rotating_frame_removes_matching_evolution():
    """Check that a matching rotating frame removes static evolution."""
    generator = Qobj([[1.0, 0.0], [0.0, -1.0]])
    time = 0.6

    propagator = static_propagator(
        hamiltonian=generator,
        time=time,
    )

    rotating = rotating_frame_propagator(
        propagator=propagator,
        generator=generator,
        time=time,
    )

    assert np.allclose(
        rotating.full(),
        qeye(2).full(),
    )


def test_time_dependent_propagator_zero_hamiltonian():
    """Check that a zero Hamiltonian produces identity evolution."""
    hamiltonian = lambda time: Qobj([[0.0, 0.0], [0.0, 0.0]])

    propagator = time_dependent_propagator(
        hamiltonian=hamiltonian,
        t0=0.0,
        t1=1.0,
        steps=10,
    )

    assert np.allclose(propagator.full(), qeye(2).full())


def test_time_dependent_propagator_constant_hamiltonian():
    """Check numerical propagation against exact static evolution."""
    h = Qobj([[1.0, 0.0], [0.0, -1.0]])

    hamiltonian = lambda time: h

    numerical = time_dependent_propagator(
        hamiltonian=hamiltonian,
        t0=0.0,
        t1=0.7,
        steps=20,
    )

    exact = static_propagator(
        hamiltonian=h,
        time=0.7,
    )

    assert np.allclose(numerical.full(), exact.full())


def test_time_dependent_propagator_rejects_invalid_steps():
    """Check that the propagator requires a positive number of steps."""
    hamiltonian = lambda time: Qobj([[0.0, 0.0], [0.0, 0.0]])

    with pytest.raises(ValueError):
        time_dependent_propagator(
            hamiltonian=hamiltonian,
            t0=0.0,
            t1=1.0,
            steps=0,
        )


def test_rotating_frame_hamiltonian_removes_generator():
    """Check that the rotating frame removes its own generator."""
    generator = Qobj([[1.0, 0.0], [0.0, -1.0]])

    def driven_hamiltonian(time):
        return generator

    rotating = rotating_frame_hamiltonian(
        time=0.4,
        driven_hamiltonian=driven_hamiltonian,
        generator=generator,
    )

    assert np.allclose(rotating.full(), np.zeros((2, 2)))


def test_rotating_frame_hamiltonian_preserves_commuting_term():
    """Check that a commuting term remains unchanged in the rotating frame."""
    generator = Qobj([[1.0, 0.0], [0.0, -1.0]])
    interaction = Qobj([[0.2, 0.0], [0.0, -0.2]])

    def driven_hamiltonian(time):
        return generator + interaction

    rotating = rotating_frame_hamiltonian(
        time=0.7,
        driven_hamiltonian=driven_hamiltonian,
        generator=generator,
    )

    assert np.allclose(rotating.full(), interaction.full())


def test_unitary_dynamical_map():
    """Check that a unitary propagator evolves a density matrix correctly."""
    state = basis(2, 0)
    density_matrix = state * state.dag()

    propagator = sigmax()

    evolved = unitary_dynamical_map(
        density_matrix=density_matrix,
        propagator=propagator,
    )

    expected_state = basis(2, 1)
    expected = expected_state * expected_state.dag()

    assert np.allclose(
        evolved.full(),
        expected.full(),
    )


def test_partial_trace_interleaved_electron_nuclear_system():
    """Check that the partial trace works for an interleaved electron-nuclear system."""
    rho_e1 = basis(2, 0).proj()
    rho_n1 = qeye(3) / 3
    rho_e2 = basis(2, 1).proj()
    rho_n2 = qeye(3) / 3

    rho_full = tensor(
        rho_e1,
        rho_n1,
        rho_e2,
        rho_n2,
    )

    rho_electronic = rho_full.ptrace([0, 2])

    expected = tensor(
        rho_e1,
        rho_e2,
    )

    assert (rho_electronic - expected).norm() < 1e-12


def test_embed_electronic_state_with_interleaved_nuclei():
    """Check that an electronic state can be embedded in a larger system with interleaved nuclei."""
    bell = (
    tensor(basis(2, 0), basis(2, 0))
    + tensor(basis(2, 1), basis(2, 1))
    ).unit()

    rho_e = bell.proj()

    rho_n1 = qeye(3) / 3
    rho_n2 = qeye(3) / 3

    rho_full = tensor(
        rho_e,
        rho_n1,
        rho_n2,
    ).permute([0, 2, 1, 3])

    assert rho_full.dims == [
        [2, 3, 2, 3],
        [2, 3, 2, 3],
    ]

    recovered_electrons = rho_full.ptrace([0, 2])

    assert (recovered_electrons - rho_e).norm() < 1e-12


def test_electron_nuclear_dynamical_map_identity():
    """Check that the electron-nuclear dynamical map reduces to the identity for a trivial propagator."""
    bell = (
        tensor(basis(2, 0), basis(2, 0))
        + tensor(basis(2, 1), basis(2, 1))
    ).unit()

    rho_e = bell.proj()

    nuclear_states = [
        qeye(3) / 3,
        qeye(3) / 3,
    ]

    propagator = qeye([2, 3, 2, 3])

    result = electron_nuclear_dynamical_map(
        density_matrix=rho_e,
        propagator=propagator,
        nuclear_states=nuclear_states,
    )

    assert result.dims == [[2, 2], [2, 2]]
    assert (result - rho_e).norm() < 1e-12


def test_electron_nuclear_dynamical_map_spin_one_identity():
    """Preserve a two-electron spin-1 state under trivial evolution."""
    electron_state = tensor(
        basis(3, 1),
        basis(3, 1),
    )

    rho_e = electron_state.proj()

    nuclear_states = [
        qeye(3) / 3,
        qeye(3) / 3,
    ]

    propagator = qeye([3, 3, 3, 3])

    result = electron_nuclear_dynamical_map(
        density_matrix=rho_e,
        propagator=propagator,
        nuclear_states=nuclear_states,
    )

    assert result.dims == [[3, 3], [3, 3]]
    assert (result - rho_e).norm() < 1e-12


def test_logical_electron_nuclear_dynamical_map_identity():
    """An identity evolution preserves a logical state with no leakage."""
    logical_basis = [
        tensor(basis(3, 0), basis(3, 0)),
        tensor(basis(3, 0), basis(3, 1)),
        tensor(basis(3, 1), basis(3, 0)),
        tensor(basis(3, 1), basis(3, 1)),
    ]

    logical_state = tensor(
        basis(2, 0),
        basis(2, 1),
    )

    density_matrix = logical_state.proj()

    propagator = qeye([3, 3, 3, 3])

    nuclear_state = qeye(3) / 3

    result = logical_electron_nuclear_dynamical_map(
        density_matrix=density_matrix,
        propagator=propagator,
        logical_basis=logical_basis,
        nuclear_states=[
            nuclear_state,
            nuclear_state,
        ],
    )

    assert result.dims == [[2, 2], [2, 2]]

    assert np.allclose(
        result.full(),
        density_matrix.full(),
    )

    assert np.isclose(
        result.tr(),
        1.0,
    )


def test_logical_electron_nuclear_dynamical_map_preserves_leakage():
    """Population outside the logical subspace appears as trace loss."""
    logical_basis = [
        tensor(basis(3, 0), basis(3, 0)),
        tensor(basis(3, 0), basis(3, 1)),
        tensor(basis(3, 1), basis(3, 0)),
        tensor(basis(3, 1), basis(3, 1)),
    ]

    initial_state = tensor(
        basis(2, 0),
        basis(2, 0),
    )

    density_matrix = initial_state.proj()

    # Swap physical electron-1 states |0> and |2>.
    swap = Qobj(
        np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]),
    )

    electron_propagator = tensor(
        swap,
        qeye(3),
    )

    # Full ordering: e1, n1, e2, n2.
    propagator = tensor(
        swap,
        qeye(3),
        qeye(3),
        qeye(3),
    )

    nuclear_state = qeye(3) / 3

    result = logical_electron_nuclear_dynamical_map(
        density_matrix=density_matrix,
        propagator=propagator,
        logical_basis=logical_basis,
        nuclear_states=[
            nuclear_state,
            nuclear_state,
        ],
    )

    assert np.isclose(
        result.tr(),
        0.0,
        atol=1e-12,
    )


def test_time_dependent_propagator_converges_for_linear_drive():
    """Check convergence against the exact integral of H(t) = t sigma_x."""
    t0 = 0.3
    t1 = 1.1

    def hamiltonian(time):
        return time * sigmax()

    integrated_amplitude = (t1**2 - t0**2) / 2
    exact = (-1j * integrated_amplitude * sigmax()).expm()

    errors = []

    for steps in [20, 40]:
        numerical = time_dependent_propagator(
            hamiltonian=hamiltonian,
            t0=t0,
            t1=t1,
            steps=steps,
        )
        errors.append(
            np.linalg.norm(numerical.full() - exact.full())
        )

    # The current right-endpoint scheme has first-order global accuracy.
    assert errors[1] < 0.02
    assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.01)


def test_time_dependent_propagator_orders_noncommuting_intervals():
    """Later evolution multiplies earlier evolution on the left."""
    first = 0.3 * sigmax()
    second = 0.4 * sigmaz()

    def hamiltonian(time):
        return first if time <= 0.5 else second

    numerical = time_dependent_propagator(
        hamiltonian=hamiltonian,
        t0=0.0,
        t1=1.0,
        steps=2,
    )
    expected = (
        (-1j * second * 0.5).expm()
        * (-1j * first * 0.5).expm()
    )

    assert np.allclose(
        numerical.full(),
        expected.full(),
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize("phase", [0.0, 0.3, np.pi / 4, np.pi / 2])
def test_reduced_dynamical_map_matches_exact_entangling_evolution(phase):
    """Check partial tracing against an exact system-environment solution."""
    plus = (basis(2, 0) + basis(2, 1)).unit()
    environment_state = basis(2, 0).proj()

    # phase = g * t, with H = g Z_system X_environment.
    propagator = (
        -1j * phase * tensor(sigmaz(), sigmax())
    ).expm()

    actual = reduced_dynamical_map(
        density_matrix=plus.proj(),
        propagator=propagator,
        environment_state=environment_state,
        keep=[0],
    )

    coherence = np.cos(2 * phase)
    expected = np.array([
        [0.5, 0.5 * coherence],
        [0.5 * coherence, 0.5],
    ])

    assert actual.dims == [[2], [2]]
    assert np.allclose(
        actual.full(),
        expected,
        atol=1e-12,
        rtol=0.0,
    )
    assert actual.tr() == pytest.approx(1.0)
    assert np.min(actual.eigenenergies()) >= -1e-12
