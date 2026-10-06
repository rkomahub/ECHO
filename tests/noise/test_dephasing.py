import numpy as np
import pytest

from echo_spin.noise.dephasing import exponential_coherence


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
