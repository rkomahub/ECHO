import numpy as np
import pytest

from echo_spin.core.operators import spin_operators
from echo_spin.core.system import SpinSystem
from echo_spin.nv.hamiltonians import electronic_nv_hamiltonian
from echo_spin.spectroscopy.transitions import allowed_transitions, electronic_spectrum


def test_joas_setting_2_electronic_transitions():
    """Check Setting 2 using the gate labels NV1=target, NV2=control."""
    omega = 2 * np.pi * 295.18

    # NV1: physical center B, strongly misaligned target.
    theta_nv1 = np.deg2rad(74.08)
    omega_nv1 = omega * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    nv1 = electronic_spectrum(
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv1,
    ) / (2 * np.pi)

    # NV2: physical center A, approximately aligned control.
    theta_nv2 = np.deg2rad(3.58)
    omega_nv2 = omega * np.array([
        np.sin(theta_nv2),
        0.0,
        np.cos(theta_nv2),
    ])

    nv2 = electronic_spectrum(
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv2,
    ) / (2 * np.pi)

    assert np.allclose(
        nv1,
        [2827.27932714, 2990.82058371],
        atol=1e-6,
        rtol=0.0,
    )
    assert np.allclose(
        nv2,
        [2571.00190538, 3160.19756161],
        atol=1e-6,
        rtol=0.0,
    )

    # Compare addressed electronic transitions with reported rounded values.
    assert nv1[1] == pytest.approx(2990.8, abs=0.05)
    assert nv2[0] == pytest.approx(2571.0, abs=0.05)


def test_allowed_transitions_aligned_nv():
    """Test the allowed transitions of a single aligned NV center."""
    system = SpinSystem([1])

    D = 2 * np.pi * 2870.0
    omega_z = 2 * np.pi * 300.0

    hamiltonian = electronic_nv_hamiltonian(
        system=system,
        electron_site=0,
        D=D,
        omega_e=[0.0, 0.0, omega_z],
    )

    sx, _, _ = spin_operators(1)

    transitions = allowed_transitions(
        hamiltonian=hamiltonian,
        control_operator=sx,
    )

    frequencies = sorted(
        transition[0] / (2 * np.pi)
        for transition in transitions
    )

    assert np.allclose(
        frequencies,
        [2570.0, 3170.0],
    )


@pytest.mark.parametrize("axis_index", [0, 1])
@pytest.mark.parametrize("amplitude", [1.0, 2.0])
def test_aligned_nv_transition_strengths(axis_index, amplitude):
    """Check transverse selection rules and squared-amplitude scaling."""
    D = 2.87
    omega_z = 0.3

    hamiltonian = electronic_nv_hamiltonian(
        system=SpinSystem([1]),
        electron_site=0,
        D=D,
        omega_e=(0.0, 0.0, omega_z),
    )
    control = amplitude * spin_operators(1)[axis_index]

    transitions = allowed_transitions(hamiltonian, control)
    transitions = sorted(transitions, key=lambda item: item[0])

    assert len(transitions) == 2
    assert np.allclose(
        [item[0] for item in transitions],
        [D - omega_z, D + omega_z],
        atol=1e-12,
        rtol=0.0,
    )
    assert np.allclose(
        [item[1] for item in transitions],
        [amplitude**2 / 2, amplitude**2 / 2],
        atol=1e-12,
        rtol=0.0,
    )


def test_aligned_nv_longitudinal_drive_has_no_transitions():
    """Sz is diagonal in the aligned NV energy basis."""
    hamiltonian = electronic_nv_hamiltonian(
        system=SpinSystem([1]),
        electron_site=0,
        D=2.87,
        omega_e=(0.0, 0.0, 0.3),
    )
    _, _, sz = spin_operators(1)

    assert allowed_transitions(hamiltonian, sz) == []


def test_transition_threshold_filters_by_strength():
    """The threshold applies to squared matrix elements."""
    hamiltonian = electronic_nv_hamiltonian(
        system=SpinSystem([1]),
        electron_site=0,
        D=2.87,
        omega_e=(0.0, 0.0, 0.3),
    )
    sx, _, _ = spin_operators(1)

    assert len(
        allowed_transitions(hamiltonian, sx, threshold=0.4)
    ) == 2
    assert allowed_transitions(
        hamiltonian, sx, threshold=0.6
    ) == []
