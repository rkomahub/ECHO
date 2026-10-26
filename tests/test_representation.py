import json
from pathlib import Path

import numpy as np
from qutip import basis, qeye, tensor

from echo_spin.characterisation import longitudinal_characterisation_report
from echo_spin.representation import (
    assess_dipolar_projection,
    build_longitudinal_model,
    longitudinal_model_report,
)
from echo_spin.scenario import longitudinal_scenario_from_dict


def test_model_preserves_site_order_and_longitudinal_energy():
    """Recover the energy and local flip of a nonsymmetric logical basis state."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/scenario.json"
    )
    specification = json.loads(path.read_text(encoding="utf-8"))
    report = longitudinal_characterisation_report(
        longitudinal_scenario_from_dict(specification)
    )
    model = build_longitudinal_model(report)

    assert model.system.dimension == 8
    assert model.hamiltonian.dims == [[2, 2, 2], [2, 2, 2]]
    assert model.hamiltonian.isherm

    state = tensor(basis(2, 1), basis(2, 0), basis(2, 0))
    flipped = model.pauli_operators["X"][0] * state
    expected_state = tensor(basis(2, 0), basis(2, 0), basis(2, 0))
    assert (flipped - expected_state).norm() < 1e-12

    signs = np.array([-1.0, 1.0, 1.0])
    detunings = np.asarray(report["detunings_rad_per_us"])
    couplings = np.asarray(report["couplings_rad_per_us"])
    energy = (
        report["constant_rad_per_us"]
        + np.dot(detunings, signs) / 2
        + sum(
            couplings[i, j] * signs[i] * signs[j]
            for i in range(3)
            for j in range(i + 1, 3)
        )
    )
    assert (
        model.hamiltonian * state - energy * state
    ).norm() < 1e-12


def test_bare_projection_exposes_transverse_exchange_in_aligned_chain():
    """Preserve the logical embedding while detecting omitted resonant exchange."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/scenario.json"
    )
    specification = json.loads(path.read_text(encoding="utf-8"))
    scenario = longitudinal_scenario_from_dict(specification)

    result = assess_dipolar_projection(scenario)
    embedding = result["embedding"]

    assert embedding.shape == (27, 8)
    assert (
        embedding.dag() * embedding - qeye([2, 2, 2])
    ).norm() < 1e-12

    report = longitudinal_characterisation_report(scenario)
    nearest = report["couplings_rad_per_us"][0][1]
    projected = result["projected_dipolar_hamiltonian"].full()

    # For a chain along local x, the |100> -> |010> exchange is -2J.
    np.testing.assert_allclose(
        projected[2, 4],
        -2 * nearest,
        rtol=1e-12,
        atol=1e-12,
    )
    assert result["omitted_in_subspace_norm_rad_per_us"] > 0
    assert result["discarded_subspace_coupling_norm_rad_per_us"] > 0


def test_model_report_preserves_diagonal_and_assessment_in_json():
    """Export the selected Hamiltonian without claiming physical validation."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/scenario.json"
    )
    specification = json.loads(path.read_text(encoding="utf-8"))
    source = longitudinal_characterisation_report(
        longitudinal_scenario_from_dict(specification)
    )

    model = build_longitudinal_model(source)
    report = longitudinal_model_report(source, assess_dipolar=True)
    restored = json.loads(json.dumps(report, allow_nan=False))

    np.testing.assert_allclose(
        restored["hamiltonian_diagonal_rad_per_us"],
        model.hamiltonian.diag(),
    )
    assert restored["subsystem_dimensions"] == [2, 2, 2]
    assert restored["assessment"]["physical_approximation_validated"] is False
    assert restored["dipolar_projection"]["status"] == "static_diagnostic_only"

    saved = restored["dipolar_projection"]["matrices_rad_per_us"]
    projected = (
        np.asarray(saved["projected"]["real"])
        + 1j * np.asarray(saved["projected"]["imag"])
    )
    assert projected.shape == (8, 8)
    np.testing.assert_allclose(projected, projected.conj().T, atol=1e-12)

    reconstructed = build_longitudinal_model(restored)

    assert reconstructed.system.dimensions == model.system.dimensions
    np.testing.assert_allclose(
        reconstructed.hamiltonian.full(),
        model.hamiltonian.full(),
        atol=1e-12,
        rtol=0.0,
    )
    assert (
        reconstructed.pauli_operators["X"][0]
        - model.pauli_operators["X"][0]
    ).norm() < 1e-12
