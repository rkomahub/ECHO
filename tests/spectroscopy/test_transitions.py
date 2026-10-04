import numpy as np

from echo_spin.spectroscopy.transitions import electronic_spectrum


def test_joas_setting_2_ab_assignment():
    """Test the electronic transition frequencies of two NV centers in the JOAS setting 2 (misaligned and aligned)."""
    omega = 2 * np.pi * 295.18

    # Hypothesis:
    # A = NV1 (misaligned)
    theta_nv1 = np.deg2rad(74.08)

    omega_nv1 = omega * np.array([
        np.sin(theta_nv1),
        0.0,
        np.cos(theta_nv1),
    ])

    nv1 = electronic_spectrum(
        D=2 * np.pi * 2865.42,
        omega_e=omega_nv1,
    ) / (2 * np.pi)

    # B = NV2 (aligned)
    theta_nv2 = np.deg2rad(3.58)

    omega_nv2 = omega * np.array([
        np.sin(theta_nv2),
        0.0,
        np.cos(theta_nv2),
    ])

    nv2 = electronic_spectrum(
        D=2 * np.pi * 2867.27,
        omega_e=omega_nv2,
    ) / (2 * np.pi)

    assert np.allclose(
    nv1,
    [2825.45481206, 2988.99810054],
    atol=1e-6,
    )

    assert np.allclose(
        nv2,
        [2572.85177892, 3162.04745120],
        atol=1e-6,
    )
    # The electronic-only model differs slightly from the experimental
    # ODMR frequencies because 14N hyperfine structure is not included yet.