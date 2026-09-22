import numpy as np
import pytest

from echo_spin.interactions import (
    dipolar_geometry,
    dipolar_interaction,
)
from echo_spin.system import SpinSystem


def test_dipolar_interaction_dimension():
    """The dipolar interaction should act on the full composite Hilbert space."""
    system = SpinSystem([1, 1])

    interaction = dipolar_interaction(
        system=system,
        site_i=0,
        site_j=1,
        coupling=1.0,
        direction=(0.0, 0.0, 1.0),
    )

    assert interaction.shape == (9, 9)


def test_zero_coupling_gives_zero_operator():
    """A zero dipolar coupling should produce the zero operator."""
    system = SpinSystem([1, 1])

    interaction = dipolar_interaction(
        system=system,
        site_i=0,
        site_j=1,
        coupling=0.0,
        direction=(0.0, 0.0, 1.0),
    )

    assert np.allclose(
        interaction.full(),
        np.zeros((9, 9)),
    )


def test_direction_is_normalized_internally():
    """The dipolar interaction should be independent of the direction's magnitude."""
    system = SpinSystem([1, 1])

    interaction_1 = dipolar_interaction(
        system,
        0,
        1,
        1.0,
        (0.0, 0.0, 1.0),
    )

    interaction_2 = dipolar_interaction(
        system,
        0,
        1,
        1.0,
        (0.0, 0.0, 5.0),
    )

    assert np.allclose(
        interaction_1.full(),
        interaction_2.full(),
    )


def test_same_site_raises_error():
    """A spin cannot interact dipolarly with itself."""
    system = SpinSystem([1])

    with pytest.raises(ValueError):
        dipolar_interaction(
            system,
            0,
            0,
            1.0,
            (0.0, 0.0, 1.0),
        )


def test_zero_direction_raises_error():
    """The direction vector must be non-zero so that a unit vector can be defined."""
    system = SpinSystem([1, 1])

    with pytest.raises(ValueError):
        dipolar_interaction(
            system,
            0,
            1,
            1.0,
            (0.0, 0.0, 0.0),
        )


def test_dipolar_geometry_two_spins():
    """Two positions should generate the expected distance-dependent coupling."""
    positions = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
    ]

    couplings, directions = dipolar_geometry(
        positions=positions,
        coupling_prefactor=8.0,
    )

    assert np.isclose(couplings[0, 1], 1.0)

    assert np.allclose(
        directions[0, 1],
        [0.0, 0.0, 1.0],
    )


def test_dipolar_geometry_is_symmetric():
    """Couplings should be symmetric and pair directions antisymmetric."""
    positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    couplings, directions = dipolar_geometry(
        positions,
        coupling_prefactor=1.0,
    )

    assert np.isclose(
        couplings[0, 1],
        couplings[1, 0],
    )

    assert np.allclose(
        directions[0, 1],
        -directions[1, 0],
    )


def test_dipolar_geometry_inverse_cube_scaling():
    """Doubling the separation should reduce the coupling by a factor of eight."""
    positions_1 = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    positions_2 = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]

    couplings_1, _ = dipolar_geometry(positions_1, 1.0)
    couplings_2, _ = dipolar_geometry(positions_2, 1.0)

    assert np.isclose(
        couplings_1[0, 1],
        8 * couplings_2[0, 1],
    )