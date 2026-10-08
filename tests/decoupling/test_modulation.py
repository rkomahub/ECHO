import numpy as np
import pytest
from qutip import qeye, sigmaz

from echo_spin.control.rotations import single_qubit_rotation
from echo_spin.core.operators import embed_operator
from echo_spin.core.system import SpinSystem
from echo_spin.decoupling.modulation import (
    effective_zz_coupling,
    modulation_function,
)


def test_no_pulses_gives_constant_modulation():
    actual = modulation_function([0.0, 0.5, 1.0], [])
    assert np.array_equal(actual, [1, 1, 1])


def test_hahn_modulation_and_pulse_boundary():
    actual = modulation_function(
        [0.0, 0.49, 0.5, 0.51, 1.0],
        [0.5],
    )
    assert np.array_equal(actual, [1, 1, -1, -1, -1])


def test_xy8_modulation_matches_operator_toggling():
    """Check scalar signs against explicit cumulative X/Y rotations."""
    pulse_times = (np.arange(8) + 0.5) / 8
    sample_times = np.concatenate(([0.0], pulse_times, [1.0]))
    actual = modulation_function(sample_times, pulse_times)

    cumulative = qeye(2)
    assert actual[0] == 1

    for index, axis in enumerate(
        ["x", "y", "x", "y", "y", "x", "y", "x"],
        start=1,
    ):
        pulse = single_qubit_rotation(np.pi, axis)
        cumulative = pulse * cumulative
        toggled = cumulative.dag() * sigmaz() * cumulative

        assert np.allclose(
            toggled.full(),
            (actual[index] * sigmaz()).full(),
        )

    assert actual[-1] == 1


def test_hahn_modulation_cancels_static_phase():
    """Equal intervals with opposite signs cancel static detuning."""
    boundaries = np.array([0.0, 0.5, 1.0])
    midpoints = (boundaries[:-1] + boundaries[1:]) / 2
    signs = modulation_function(midpoints, [0.5])

    integrated_modulation = signs @ np.diff(boundaries)
    assert integrated_modulation == pytest.approx(0.0)


@pytest.mark.parametrize(
    "times, pulse_times",
    [
        ([-0.1], []),
        ([np.nan], []),
        ([[0.0, 1.0]], []),
        ([0.0], [-0.1]),
        ([0.0], [np.inf]),
        ([0.0], [0.5, 0.25]),
        ([0.0], [0.5, 0.5]),
        ([0.0], [[0.5]]),
        ([0.0], [0.5j]),
    ],
)
def test_modulation_rejects_invalid_times(times, pulse_times):
    with pytest.raises(ValueError):
        modulation_function(times, pulse_times)


@pytest.mark.parametrize(
    "pulses_i, pulses_j, expected_factor",
    [
        ([], [], 1.0),
        ([0.5], [0.5], 1.0),
        ([0.5], [], 0.0),
        ([], [0.5], 0.0),
        ([0.25], [], -0.5),
        ([0.2, 0.6], [0.4, 0.8], 0.2),
    ],
)
def test_effective_zz_coupling(pulses_i, pulses_j, expected_factor):
    """Check retention, cancellation and signed partial recoupling."""
    coupling = -0.3

    actual = effective_zz_coupling(
        coupling=coupling,
        total_duration=1.0,
        pulse_times_i=pulses_i,
        pulse_times_j=pulses_j,
    )

    assert actual == pytest.approx(coupling * expected_factor)


def test_effective_zz_coupling_is_invariant_under_time_rescaling():
    original = effective_zz_coupling(
        0.3, 1.0, [0.2, 0.6], [0.4, 0.8]
    )
    rescaled = effective_zz_coupling(
        0.3, 10.0, [2.0, 6.0], [4.0, 8.0]
    )

    assert rescaled == pytest.approx(original)


@pytest.mark.parametrize(
    "coupling, duration, pulses_i, pulses_j",
    [
        (np.nan, 1.0, [], []),
        (0.3, 0.0, [], []),
        (0.3, np.inf, [], []),
        (0.3, 1.0, [1.1], []),
        (0.3, 1.0, [], [1.1]),
        (0.3, 1.0, [0.5, 0.5], []),
        (0.3, 1.0, [], [0.8, 0.2]),
    ],
)
def test_effective_zz_coupling_rejects_invalid_parameters(
    coupling, duration, pulses_i, pulses_j
):
    with pytest.raises(ValueError):
        effective_zz_coupling(
            coupling, duration, pulses_i, pulses_j
        )


def test_three_qubit_recoupling_suppresses_spectator():
    """Retain the target ZZ interaction and cancel both spectator terms."""
    system = SpinSystem([0.5, 0.5, 0.5])
    total_duration = 2.0

    j01 = 0.3
    j02 = -0.2
    j12 = 0.17

    z = [
        embed_operator(sigmaz(), site, system)
        for site in range(3)
    ]

    target_hamiltonian = j01 * z[0] * z[1]
    free_hamiltonian = (
        target_hamiltonian
        + j02 * z[0] * z[2]
        + j12 * z[1] * z[2]
    )

    schedules = [[], [], [total_duration / 2]]

    for i, j, coupling, expected in [
        (0, 1, j01, j01),
        (0, 2, j02, 0.0),
        (1, 2, j12, 0.0),
    ]:
        effective = effective_zz_coupling(
            coupling=coupling,
            total_duration=total_duration,
            pulse_times_i=schedules[i],
            pulse_times_j=schedules[j],
        )
        assert effective == pytest.approx(expected, abs=1e-12)

    spectator_pulse = embed_operator(
        single_qubit_rotation(np.pi, "x"),
        site=2,
        system=system,
    )

    half_evolution = (
        -1j * free_hamiltonian * total_duration / 2
    ).expm()

    lab_evolution = (
        half_evolution
        * spectator_pulse
        * half_evolution
    )

    # Remove the known final spectator rotation.
    echo_evolution = spectator_pulse.dag() * lab_evolution

    expected_evolution = (
        -1j * target_hamiltonian * total_duration
    ).expm()

    assert np.allclose(
        echo_evolution.full(),
        expected_evolution.full(),
        atol=1e-12,
        rtol=0.0,
    )
