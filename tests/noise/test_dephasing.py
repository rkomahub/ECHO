import numpy as np
import pytest
from qutip import basis, tensor

from echo_spin.noise.dephasing import (
    dephasing_channel,
    exponential_coherence,
    two_qubit_dephasing_channel,
)


def test_exponential_coherence():
    """Test that exponential coherence has the correct limiting behaviour."""
    assert np.isclose(
        exponential_coherence(
            time=0.0,
            coherence_time=465.0,
        ),
        1.0,
    )

    assert np.isclose(
        exponential_coherence(
            time=465.0,
            coherence_time=465.0,
        ),
        np.exp(-1),
    )


def test_exponential_coherence_rejects_invalid_parameters():
    """Test that invalid time and coherence-time parameters are rejected."""
    with pytest.raises(ValueError):
        exponential_coherence(
            time=-1.0,
            coherence_time=465.0,
        )

    with pytest.raises(ValueError):
        exponential_coherence(
            time=1.0,
            coherence_time=0.0,
        )


def test_joas_xy8_coherence_limit():
    """Reproduce the Joas XY8 coherence-limited gate error of approximately 1.4%."""
    t2_nv1 = 454.0
    t2_nv2 = 476.0

    average_t2 = (
        t2_nv1 + t2_nv2
    ) / 2

    gate_duration = 6.4

    fidelity = exponential_coherence(
        time=gate_duration,
        coherence_time=average_t2,
    )

    error = 1 - fidelity

    print("Joas XY8 coherence fidelity =", fidelity)
    print("Joas XY8 coherence error =", error)

    assert np.isclose(fidelity, 0.986, atol=1e-3)
    assert np.isclose(error, 0.014, atol=1e-3)


def test_dephasing_channel_preserves_populations():
    """Check that pure dephasing leaves basis-state populations unchanged."""
    zero = basis(2, 0)
    one = basis(2, 1)

    density_matrix = (
        0.3 * zero.proj()
        + 0.7 * one.proj()
    )

    result = dephasing_channel(
        density_matrix=density_matrix,
        time=10.0,
        coherence_time=20.0,
    )

    assert np.allclose(
        np.diag(result.full()),
        [0.3, 0.7],
    )


def test_dephasing_channel_decays_coherence():
    """Check exponential decay of the off-diagonal density-matrix elements."""
    plus = (
        basis(2, 0)
        + basis(2, 1)
    ).unit()

    density_matrix = plus.proj()

    time = 10.0
    coherence_time = 20.0

    result = dephasing_channel(
        density_matrix=density_matrix,
        time=time,
        coherence_time=coherence_time,
    )

    expected_coherence = (
        0.5
        * np.exp(-time / coherence_time)
    )

    assert np.isclose(
        result[0, 1],
        expected_coherence,
    )

    assert np.isclose(
        result[1, 0],
        expected_coherence,
    )


def test_dephasing_channel_preserves_trace():
    """Check that the dephasing channel preserves the density-matrix trace."""
    plus = (
        basis(2, 0)
        + basis(2, 1)
    ).unit()

    result = dephasing_channel(
        density_matrix=plus.proj(),
        time=10.0,
        coherence_time=20.0,
    )

    assert np.isclose(
        result.tr(),
        1.0,
    )


def test_two_qubit_dephasing_channel():
    """Check independent exponential dephasing of two qubits."""
    zero = basis(2, 0)
    one = basis(2, 1)

    zero_zero = tensor(zero, zero)
    one_zero = tensor(one, zero)
    zero_one = tensor(zero, one)

    time = 10.0
    coherence_times = (20.0, 40.0)

    state_1 = (
        zero_zero + one_zero
    ).unit()

    result_1 = two_qubit_dephasing_channel(
        density_matrix=state_1.proj(),
        time=time,
        coherence_times=coherence_times,
    )

    assert np.isclose(
        result_1[0, 2],
        0.5 * np.exp(-time / coherence_times[0]),
    )

    state_2 = (
        zero_zero + zero_one
    ).unit()

    result_2 = two_qubit_dephasing_channel(
        density_matrix=state_2.proj(),
        time=time,
        coherence_times=coherence_times,
    )

    assert np.isclose(
        result_2[0, 1],
        0.5 * np.exp(-time / coherence_times[1]),
    )
