import numpy as np
import pytest
from qutip import basis

from echo_spin.core.operators import embed_operator, spin_operators
from echo_spin.core.system import SpinSystem
from echo_spin.nv.hamiltonians import (
    electronic_nv_hamiltonian,
    hyperfine_hamiltonian,
    nuclear_nv_hamiltonian,
    nv_hamiltonian,
    rotated_electronic_nv_hamiltonian,
)


def test_electronic_nv_hamiltonian_dimension():
    """One electronic spin-1 NV Hamiltonian should be 3x3."""
    system = SpinSystem([1])

    hamiltonian = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=1.0,
        omega_e=(0.0, 0.0, 0.0),
    )

    assert hamiltonian.shape == (3, 3)


def test_zero_field_splitting_spectrum():
    """With no magnetic field, the ms = ±1 states are split from ms = 0 by D."""
    system = SpinSystem([1])

    D = 2.87

    hamiltonian = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=D,
        omega_e=(0.0, 0.0, 0.0),
    )

    eigenvalues = np.sort(hamiltonian.eigenenergies())

    assert np.allclose(
        eigenvalues,
        [0.0, D, D],
    )


def test_longitudinal_zeeman_splitting():
    """A longitudinal field should split the ms = ±1 levels."""
    system = SpinSystem([1])

    D = 2.87
    omega_z = 0.1

    hamiltonian = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=D,
        omega_e=(0.0, 0.0, omega_z),
    )

    eigenvalues = np.sort(hamiltonian.eigenenergies())

    expected = np.sort([
        0.0,
        D - omega_z,
        D + omega_z,
    ])

    assert np.allclose(eigenvalues, expected)


def test_nv_hamiltonian_embeds_in_larger_system():
    """The electronic Hamiltonian should act on the full composite space."""
    system = SpinSystem([1, 1])

    hamiltonian = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=1.0,
        omega_e=(0.0, 0.0, 0.0),
    )

    assert hamiltonian.shape == (9, 9)


def test_nv_electron_must_have_spin_one():
    """The electronic NV subsystem must be spin 1."""
    system = SpinSystem([0.5])

    with pytest.raises(ValueError):
        electronic_nv_hamiltonian(
            system=system,
            electron_site=0,
            D=1.0,
            omega_e=(0.0, 0.0, 0.0),
        )


def test_omega_e_requires_three_components():
    """The electronic Larmor vector must contain x, y and z components."""
    system = SpinSystem([1])

    with pytest.raises(ValueError):
        electronic_nv_hamiltonian(
            system=system,
            electron_site=0,
            D=1.0,
            omega_e=(0.0, 0.0),
        )


def test_nuclear_nv_hamiltonian_dimension():
    system = SpinSystem([1])

    hamiltonian = nuclear_nv_hamiltonian(
        system=system,
        nuclear_site=0,
        Q=1.0,
        omega_n=(0.0, 0.0, 0.0),
    )

    assert hamiltonian.shape == (3, 3)


def test_nuclear_quadrupole_spectrum():
    system = SpinSystem([1])

    Q = -4.95

    hamiltonian = nuclear_nv_hamiltonian(
        system=system,
        nuclear_site=0,
        Q=Q,
        omega_n=(0.0, 0.0, 0.0),
    )

    eigenvalues = np.sort(hamiltonian.eigenenergies())

    expected = np.sort([0.0, Q, Q])

    assert np.allclose(eigenvalues, expected)


def test_nuclear_longitudinal_zeeman_splitting():
    system = SpinSystem([1])

    Q = -4.95
    omega_z = 0.1

    hamiltonian = nuclear_nv_hamiltonian(
        system=system,
        nuclear_site=0,
        Q=Q,
        omega_n=(0.0, 0.0, omega_z),
    )

    eigenvalues = np.sort(hamiltonian.eigenenergies())

    expected = np.sort([
        0.0,
        Q - omega_z,
        Q + omega_z,
    ])

    assert np.allclose(eigenvalues, expected)


def test_nuclear_hamiltonian_embeds_in_larger_system():
    system = SpinSystem([1, 1])

    hamiltonian = nuclear_nv_hamiltonian(
        system=system,
        nuclear_site=1,
        Q=1.0,
        omega_n=(0.0, 0.0, 0.0),
    )

    assert hamiltonian.shape == (9, 9)


def test_nuclear_spin_must_be_one():
    system = SpinSystem([0.5])

    with pytest.raises(ValueError):
        nuclear_nv_hamiltonian(
            system=system,
            nuclear_site=0,
            Q=1.0,
            omega_n=(0.0, 0.0, 0.0),
        )


def test_omega_n_requires_three_components():
    system = SpinSystem([1])

    with pytest.raises(ValueError):
        nuclear_nv_hamiltonian(
            system=system,
            nuclear_site=0,
            Q=1.0,
            omega_n=(0.0, 0.0),
        )


def test_hyperfine_hamiltonian_dimension():
    """Electron-nuclear hyperfine interaction should act on the full space."""
    system = SpinSystem([1, 1])

    A = np.eye(3)

    hamiltonian = hyperfine_hamiltonian(
        system=system,
        electron_site=0,
        nuclear_site=1,
        A=A,
    )

    assert hamiltonian.shape == (9, 9)


def test_zero_hyperfine_tensor_gives_zero_hamiltonian():
    """A zero hyperfine tensor should produce the zero operator."""
    system = SpinSystem([1, 1])

    A = np.zeros((3, 3))

    hamiltonian = hyperfine_hamiltonian(
        system=system,
        electron_site=0,
        nuclear_site=1,
        A=A,
    )

    assert np.allclose(
        hamiltonian.full(),
        np.zeros((9, 9)),
    )


def test_isotropic_hyperfine_interaction():
    """For A = a I, the interaction should be a S dot I."""
    system = SpinSystem([1, 1])

    a = 2.0
    A = a * np.eye(3)

    hamiltonian = hyperfine_hamiltonian(
        system=system,
        electron_site=0,
        nuclear_site=1,
        A=A,
    )

    sx, sy, sz = spin_operators(1)

    sx_e = embed_operator(sx, 0, system)
    sy_e = embed_operator(sy, 0, system)
    sz_e = embed_operator(sz, 0, system)

    sx_n = embed_operator(sx, 1, system)
    sy_n = embed_operator(sy, 1, system)
    sz_n = embed_operator(sz, 1, system)

    expected = a * (
        sx_e * sx_n
        + sy_e * sy_n
        + sz_e * sz_n
    )

    assert np.allclose(
        hamiltonian.full(),
        expected.full(),
    )


def test_invalid_hyperfine_tensor_shape():
    """The hyperfine tensor must be a 3x3 matrix."""
    system = SpinSystem([1, 1])

    with pytest.raises(ValueError):
        hyperfine_hamiltonian(
            system=system,
            electron_site=0,
            nuclear_site=1,
            A=np.eye(2),
        )


def test_same_site_hyperfine_interaction_raises_error():
    """Electron and nucleus must occupy different subsystems."""
    system = SpinSystem([1])

    with pytest.raises(ValueError):
        hyperfine_hamiltonian(
            system=system,
            electron_site=0,
            nuclear_site=0,
            A=np.eye(3),
        )


def test_complete_nv_hamiltonian_dimension():
    """A complete electron+nucleus NV Hamiltonian should act on a 9D space."""
    system = SpinSystem([1, 1])

    hamiltonian = nv_hamiltonian(
        system=system,
        electron_site=0,
        nuclear_site=1,
        D=2.87,
        omega_e=(0.0, 0.0, 0.1),
        Q=-4.95,
        omega_n=(0.0, 0.0, 0.01),
        A=np.eye(3),
    )

    assert hamiltonian.shape == (9, 9)


def test_complete_nv_hamiltonian_equals_sum_of_terms():
    """The complete NV Hamiltonian should equal He + Hn + Hhf."""
    system = SpinSystem([1, 1])

    D = 2.87
    omega_e = (0.0, 0.0, 0.1)

    Q = -4.95
    omega_n = (0.0, 0.0, 0.01)

    A = np.diag([0.2, 0.2, 0.3])

    complete = nv_hamiltonian(
        system=system,
        electron_site=0,
        nuclear_site=1,
        D=D,
        omega_e=omega_e,
        Q=Q,
        omega_n=omega_n,
        A=A,
    )

    expected = (
        electronic_nv_hamiltonian(
            system=system,
            electron_site=0,
            D=D,
            omega_e=omega_e,
        )
        + nuclear_nv_hamiltonian(
            system=system,
            nuclear_site=1,
            Q=Q,
            omega_n=omega_n,
        )
        + hyperfine_hamiltonian(
            system=system,
            electron_site=0,
            nuclear_site=1,
            A=A,
        )
    )

    assert np.allclose(
        complete.full(),
        expected.full(),
    )


def test_zero_rotation_recovers_standard_nv_hamiltonian():
    """Zero misalignment should reproduce the ordinary NV Hamiltonian."""
    system = SpinSystem([1])

    standard = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2.87,
        omega_e=(0.1, 0.2, 0.3),
    )

    rotated = rotated_electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=2.87,
        omega_e_lab=(0.1, 0.2, 0.3),
        axis="y",
        angle=0.0,
    )

    assert np.allclose(
        standard.full(),
        rotated.full(),
    )


def test_misaligned_nv_rotates_magnetic_field():
    """A misaligned NV should use the field expressed in its local frame."""
    system = SpinSystem([1])

    angle = np.pi / 2

    hamiltonian = rotated_electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=0.0,
        omega_e_lab=(0.0, 0.0, 1.0),
        axis="y",
        angle=angle,
    )

    sx, _, _ = spin_operators(1)

    expected = -sx

    assert np.allclose(
        hamiltonian.full(),
        expected.full(),
    )


@pytest.mark.parametrize("phase", [0.0, np.pi / 2, 0.7])
def test_transverse_zeeman_spectrum_and_dark_state(phase):
    """Check exact transverse-field energies and the uncoupled eigenstate."""
    system = SpinSystem([1])
    D = 2.87
    omega_perp = 0.4

    hamiltonian = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=D,
        omega_e=(
            omega_perp * np.cos(phase),
            omega_perp * np.sin(phase),
            0.0,
        ),
    )

    splitting = np.sqrt(D**2 + 4 * omega_perp**2)
    expected_energies = np.sort([
        D,
        (D - splitting) / 2,
        (D + splitting) / 2,
    ])

    assert hamiltonian.isherm
    assert np.allclose(
        hamiltonian.eigenenergies(),
        expected_energies,
        atol=1e-12,
        rtol=0.0,
    )

    dark_state = (
        np.exp(-1j * phase) * basis(3, 0)
        - np.exp(1j * phase) * basis(3, 2)
    ).unit()

    assert (hamiltonian * dark_state - D * dark_state).norm() < 1e-12


@pytest.mark.parametrize("E", [-0.2, 0.0, 0.2])
@pytest.mark.parametrize("omega_z", [0.0, 0.3])
def test_nv_strain_and_axial_field_spectrum(E, omega_z):
    """Check the exact strain-plus-axial-Zeeman spectrum."""
    D = 2.87

    hamiltonian = electronic_nv_hamiltonian(
        system=SpinSystem([1]),
        electron_site=0,
        D=D,
        omega_e=(0.0, 0.0, omega_z),
        E=E,
    )

    splitting = np.sqrt(omega_z**2 + E**2)
    expected = np.sort([0.0, D - splitting, D + splitting])

    assert hamiltonian.isherm
    assert np.allclose(
        hamiltonian.eigenenergies(),
        expected,
        atol=1e-12,
        rtol=0.0,
    )

    # In the |+1>, |0>, |-1> basis, strain couples |+1> to |-1>.
    assert hamiltonian[0, 2] == pytest.approx(E)
    assert hamiltonian[2, 0] == pytest.approx(E)


@pytest.mark.parametrize("E", [np.nan, np.inf])
def test_nv_hamiltonian_rejects_nonfinite_strain(E):
    with pytest.raises(ValueError):
        electronic_nv_hamiltonian(
            system=SpinSystem([1]),
            electron_site=0,
            D=2.87,
            omega_e=(0.0, 0.0, 0.0),
            E=E,
        )
