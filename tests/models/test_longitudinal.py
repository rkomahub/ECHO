import numpy as np
import pytest

from echo_spin.models.longitudinal import (
    longitudinal_qubit_parameters,
    longitudinal_register_hamiltonian,
)


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


def test_longitudinal_register_energies_and_tensor_order():
    """Match all three-qubit energies with site zero as the leftmost factor."""
    detunings = [0.2, -0.4, 0.8]
    couplings = np.array([
        [0.0, 0.3, -0.2],
        [0.3, 0.0, 0.1],
        [-0.2, 0.1, 0.0],
    ])
    hamiltonian = longitudinal_register_hamiltonian(
        detunings, couplings, constant=0.7
    )

    expected = []
    for state in range(8):
        z0 = 1 - 2 * ((state >> 2) & 1)
        z1 = 1 - 2 * ((state >> 1) & 1)
        z2 = 1 - 2 * (state & 1)
        expected.append(
            0.7
            + 0.1 * z0 - 0.2 * z1 + 0.4 * z2
            + 0.3 * z0 * z1 - 0.2 * z0 * z2 + 0.1 * z1 * z2
        )

    np.testing.assert_allclose(
        hamiltonian.full(), np.diag(expected), atol=1e-12, rtol=0.0
    )


@pytest.mark.parametrize("number_of_qubits", [1, 3, 8])
def test_longitudinal_register_dimensions(number_of_qubits):
    """Construct correctly dimensioned Hermitian operators including eight qubits."""
    hamiltonian = longitudinal_register_hamiltonian(
        np.zeros(number_of_qubits),
        np.zeros((number_of_qubits, number_of_qubits)),
    )

    assert hamiltonian.shape == (2**number_of_qubits,) * 2
    assert hamiltonian.dims == [[2] * number_of_qubits] * 2
    assert hamiltonian.isherm
    np.testing.assert_array_equal(hamiltonian.diag(), 0.0)


@pytest.mark.parametrize(
    "couplings",
    [
        np.zeros((3, 3)),
        np.array([[0.0, 1.0], [0.0, 0.0]]),
        np.eye(2),
        np.array([[0.0, np.inf], [np.inf, 0.0]]),
    ],
)
def test_longitudinal_register_rejects_invalid_couplings(couplings):
    """Reject wrong-sized, asymmetric, diagonal or nonfinite pair coefficients."""
    with pytest.raises(ValueError):
        longitudinal_register_hamiltonian([0.0, 0.0], couplings)
