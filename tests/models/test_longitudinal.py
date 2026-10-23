import numpy as np
import pytest

from echo_spin.models.longitudinal import longitudinal_qubit_parameters


def test_longitudinal_encoding_preserves_pair_energies():
    """Recover the projected spin-one longitudinal energies including the constant."""
    tensors = np.zeros((2, 2, 3, 3))
    tensors[0, 1, 2, 2] = tensors[1, 0, 2, 2] = 4.0

    detunings, couplings, constant = longitudinal_qubit_parameters(
        [0.0, 0.0], tensors
    )

    np.testing.assert_allclose(detunings, [-2.0, -2.0])
    assert couplings[0, 1] == pytest.approx(1.0)

    z0 = np.array([1, 1, -1, -1])
    z1 = np.array([1, -1, 1, -1])
    energies = (
        constant
        + detunings[0] * z0 / 2
        + detunings[1] * z1 / 2
        + couplings[0, 1] * z0 * z1
    )

    # Physical m_s products for 00, 0(-1), (-1)0, (-1)(-1).
    np.testing.assert_allclose(energies, [0, 0, 0, 4])


def test_longitudinal_parameters_accumulate_all_neighbours():
    """Include every pair shift while leaving the supplied parameters unchanged."""
    tensors = np.zeros((3, 3, 3, 3))
    longitudinal = np.array([[0, 4, -2], [4, 0, 6], [-2, 6, 0]])
    tensors[:, :, 2, 2] = longitudinal
    tensors[0, 1, 0, 0] = tensors[1, 0, 0, 0] = 100.0
    bare = np.array([0.1, 0.2, 0.3])

    detunings, couplings, constant = longitudinal_qubit_parameters(bare, tensors)

    np.testing.assert_allclose(detunings, [-0.9, -4.8, -1.7])
    np.testing.assert_allclose(couplings, longitudinal / 4)
    assert constant == pytest.approx(2.0)
    np.testing.assert_array_equal(bare, [0.1, 0.2, 0.3])


@pytest.mark.parametrize("invalid_case", ["shape", "asymmetry", "self", "nan"])
def test_longitudinal_parameters_reject_invalid_tensors(invalid_case):
    """Reject malformed, nonreciprocal, self-interacting or nonfinite tensors."""
    tensors = np.zeros((2, 2, 3, 3))
    if invalid_case == "shape":
        tensors = tensors[:1]
    elif invalid_case == "asymmetry":
        tensors[0, 1, 2, 2] = 1.0
    elif invalid_case == "self":
        tensors[0, 0, 2, 2] = 1.0
    else:
        tensors[0, 1, 2, 2] = np.nan

    with pytest.raises(ValueError):
        longitudinal_qubit_parameters([0.0, 0.0], tensors)
