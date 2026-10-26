import json
from pathlib import Path

import numpy as np
from scipy.constants import hbar, mu_0

from echo_spin.characterisation import (
    characterise_local_electrons,
    characterise_longitudinal_scenario,
    local_spectral_overlap,
)
from echo_spin.scenario import longitudinal_scenario_from_dict


def test_chain_characterisation_matches_dipolar_law_and_distance_scaling():
    """Recover signed chain couplings, projection shifts and cubic timings."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/scenario.json"
    )
    specification = json.loads(path.read_text(encoding="utf-8"))
    scenario = longitudinal_scenario_from_dict(specification)
    result = characterise_longitudinal_scenario(scenario)

    spacing = 15e-9
    gamma_si = specification["sites"][0]["gamma_e"] * 1e6
    nearest = (
        mu_0 * hbar * gamma_si**2
        / (4 * np.pi * spacing**3)
        * 1e-6 / 4
    )
    expected = nearest * np.array([
        [0.0, 1.0, 1 / 8],
        [1.0, 0.0, 1.0],
        [1 / 8, 1.0, 0.0],
    ])

    np.testing.assert_allclose(
        result["couplings_rad_per_us"],
        expected,
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result["detunings_rad_per_us"],
        -2 * expected.sum(axis=1),
    )
    np.testing.assert_allclose(
        result["constant_rad_per_us"],
        np.triu(expected, k=1).sum(),
    )
    np.testing.assert_allclose(
        result["ideal_gate_times_us"][0, 1],
        np.pi / (4 * nearest),
    )
    assert np.all(np.isnan(np.diag(result["ideal_gate_times_us"])))
    assert result["zero_coupling_pairs"].shape == (0, 2)

    for position in specification["register"]["positions"]:
        position[:] = [2 * coordinate for coordinate in position]

    scaled = characterise_longitudinal_scenario(
        longitudinal_scenario_from_dict(specification)
    )
    np.testing.assert_allclose(
        scaled["couplings_rad_per_us"],
        expected / 8,
    )
    np.testing.assert_allclose(
        scaled["ideal_gate_times_us"][0, 1],
        8 * result["ideal_gate_times_us"][0, 1],
    )


def test_aligned_chain_has_identical_axial_electronic_transitions():
    """Recover axial spin-one transition frequencies and transverse strengths."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/scenario.json"
    )
    specification = json.loads(path.read_text(encoding="utf-8"))
    scenario = longitudinal_scenario_from_dict(specification)

    spectra = characterise_local_electrons(scenario)
    site = specification["sites"][0]
    splitting = abs(site["gamma_e"] * 0.01)
    expected = [site["D"] - splitting, site["D"] + splitting]

    assert len(spectra) == 3
    for spectrum in spectra:
        transitions = spectrum["transitions"]
        assert len(transitions) == 2
        np.testing.assert_allclose(
            [line["frequency_rad_per_us"] for line in transitions],
            expected,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            [line["strength"] for line in transitions],
            [0.5, 0.5],
            atol=1e-12,
        )


def test_spectral_overlap_distinguishes_coincident_and_separated_lines():
    """Identify shared lines without treating empty spectra as separated."""
    spectra = [
        {
            "site": 0,
            "transitions": [{"frequency_rad_per_us": 10.0}],
        },
        {
            "site": 1,
            "transitions": [{"frequency_rad_per_us": 10.0}],
        },
        {
            "site": 2,
            "transitions": [{"frequency_rad_per_us": 13.0}],
        },
        {
            "site": 3,
            "transitions": [],
        },
    ]

    result = local_spectral_overlap(spectra)

    assert result["coincident_site_pairs"] == [[0, 1]]
    assert result["minimum_gaps_rad_per_us"][0][2] == 3.0
    assert result["minimum_gaps_rad_per_us"][2][0] == 3.0
    assert result["minimum_gaps_rad_per_us"][0][0] is None
    assert result["minimum_gaps_rad_per_us"][0][3] is None
