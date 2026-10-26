import json
from pathlib import Path

import numpy as np
import pytest

from echo_spin.scenario import (
    load_longitudinal_scenario,
    longitudinal_scenario_from_dict,
    validate_control_design,
    validate_environment,
    validate_evaluation_parameters,
    validate_logical_preparation,
    validate_noise_parameters,
    validate_numerical_parameters,
    validate_uncertainty_parameters,
)

CONFIGURATION = (
    Path(__file__).resolve().parents[1]
    / "configs/registers/linear/n3/scenario.json"
)


def test_three_nv_scenario_preserves_task_and_source():
    """Resolve the trial while preserving its source and independent copies."""
    source = json.loads(CONFIGURATION.read_text(encoding="utf-8"))
    scenario = longitudinal_scenario_from_dict(source)

    assert scenario.register.number_of_nv == 3
    assert scenario.target_pair == (0, 1)
    assert scenario.specification == source
    np.testing.assert_allclose(
        scenario.register.positions[:, 0],
        [0.0, 15e-9, 30e-9],
        atol=1e-22,
        rtol=0.0,
    )

    source["register"]["positions"][1][0] = 99.0
    assert scenario.specification["register"]["positions"][1][0] == 15e-9
    assert scenario.register.positions[1, 0] == 15e-9

    loaded = load_longitudinal_scenario(CONFIGURATION)
    assert loaded.target_pair == scenario.target_pair


@pytest.mark.parametrize(
    "pair",
    [[0, 0], [0, 3], [True, 1]],
)
def test_scenario_rejects_invalid_target_pair(pair):
    """Reject repeated, out-of-range and noninteger target indices."""
    source = json.loads(CONFIGURATION.read_text(encoding="utf-8"))
    source["task"]["target_pair"] = pair

    with pytest.raises(ValueError, match="target_pair"):
        longitudinal_scenario_from_dict(source)


def test_scenario_rejects_incomplete_physics_and_unsupported_control():
    """Reject missing site parameters and unimplemented pulse assumptions."""
    source = json.loads(CONFIGURATION.read_text(encoding="utf-8"))
    del source["sites"][0]["gamma_e"]

    with pytest.raises(ValueError, match="Each site requires"):
        longitudinal_scenario_from_dict(source)

    source = json.loads(CONFIGURATION.read_text(encoding="utf-8"))
    source["control"]["pulse_model"] = "finite"

    with pytest.raises(ValueError, match="control"):
        longitudinal_scenario_from_dict(source)


def test_scenario_requires_parameter_provenance():
    """Reject a scenario missing the source of a required parameter family."""
    source = json.loads(CONFIGURATION.read_text(encoding="utf-8"))
    del source["provenance"]["gamma_e"]

    with pytest.raises(ValueError, match="provenance"):
        longitudinal_scenario_from_dict(source)


@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), True],
)
def test_scenario_rejects_invalid_physical_coefficients(invalid_value):
    """Reject nonfinite coefficients and booleans before model construction."""
    source = json.loads(CONFIGURATION.read_text(encoding="utf-8"))
    source["sites"][0]["bare_detuning"] = invalid_value

    with pytest.raises(ValueError, match="finite real"):
        longitudinal_scenario_from_dict(source)


def test_scenario_supports_four_sites_without_changing_the_loader():
    """Resolve another register size using only configuration changes."""
    source = json.loads(CONFIGURATION.read_text(encoding="utf-8"))
    source["register"]["positions"].append([45e-9, 0.0, 0.0])
    source["register"]["nv_axes"].append([0.0, 0.0, 1.0])
    source["register"]["transverse_references"].append([1.0, 0.0, 0.0])
    source["sites"].append(dict(source["sites"][0]))
    source["task"]["target_pair"] = [1, 3]

    scenario = longitudinal_scenario_from_dict(source)

    assert scenario.register.number_of_nv == 4
    assert scenario.target_pair == (1, 3)
    assert len(scenario.specification["sites"]) == 4


def test_omitted_environment_is_explicitly_unresolved():
    """Preserve older inputs without inventing physical settings."""
    source = json.loads(CONFIGURATION.read_text(encoding="utf-8"))
    source.pop("environment", None)

    scenario = longitudinal_scenario_from_dict(source)

    assert "environment" not in scenario.specification
    assert all(
        record == {"status": "unresolved", "parameters": {}}
        for record in scenario.environment.values()
    )


def test_environment_rejects_nonfinite_nested_parameters():
    """Reject invalid numerical values inside future descriptive records."""
    source = json.loads(CONFIGURATION.read_text(encoding="utf-8"))
    source["environment"] = {
        "noise": {
            "status": "specified",
            "parameters": {"strength": float("nan")},
        }
    }

    with pytest.raises(ValueError, match="finite JSON"):
        longitudinal_scenario_from_dict(source)


def test_nuclear_records_validate_ordered_sites_and_copy_inputs():
    """Accept local-frame nitrogen tensors without activating nuclear dynamics."""
    site = {
        "isotope": "14N",
        "spin": 1,
        "gamma_n": 1.0,
        "Q": -2.0,
        "A": [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        "frame": "nv_local",
        "source": "Synthetic validation values; not physical calibration.",
    }
    environment = {
        "nuclear_parameters": {
            "status": "specified",
            "parameters": {"sites": [site, site]},
        }
    }

    result = validate_environment(environment, number_of_nv=2)

    site["A"][0][0] = 99.0
    records = result["nuclear_parameters"]["parameters"]["sites"]
    assert records[0]["A"][0][0] == 1.0

    with pytest.raises(ValueError, match="one nuclear record"):
        validate_environment(environment, number_of_nv=3)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("spin", 0.5, "spin"),
        ("gamma_n", True, "gamma_n"),
        ("A", [[1.0]], "3x3"),
        ("frame", "laboratory", "nv_local"),
    ],
)
def test_nuclear_records_reject_inconsistent_parameters(field, value, message):
    """Reject isotope, coefficient, tensor and frame inconsistencies."""
    site = {
        "isotope": "14N",
        "spin": 1,
        "gamma_n": 1.0,
        "Q": 0.0,
        "A": np.eye(3).tolist(),
        "frame": "nv_local",
        "source": "Synthetic test parameters.",
    }
    site[field] = value
    environment = {
        "nuclear_parameters": {
            "status": "specified",
            "parameters": {"sites": [site]},
        }
    }

    with pytest.raises(ValueError, match=message):
        validate_environment(environment, number_of_nv=1)


def test_scenario_checks_static_bounds():
    """Validate separation and field bounds against the resolved register."""
    source = json.loads(CONFIGURATION.read_text(encoding="utf-8"))
    source["environment"]["constraints"] = {
        "status": "specified",
        "parameters": {
            "minimum_separation_m": 5e-9,
            "maximum_field_t": 0.02,
        },
    }

    scenario = longitudinal_scenario_from_dict(source)

    assert scenario.constraint_checks == {
        "minimum_separation": "Passed.",
        "maximum_site_field": "Passed.",
    }


@pytest.mark.parametrize(
    "name,value",
    [
        ("minimum_separation_m", 20e-9),
        ("maximum_field_t", 0.005),
        ("maximum_field_t", -0.01),
    ],
)
def test_scenario_rejects_invalid_or_violated_static_bounds(name, value):
    """Reject invalid bounds and configurations that exceed valid bounds."""
    source = json.loads(CONFIGURATION.read_text(encoding="utf-8"))
    source["environment"]["constraints"] = {
        "status": "specified",
        "parameters": {name: value},
    }

    with pytest.raises(ValueError, match=name):
        longitudinal_scenario_from_dict(source)


def test_logical_preparation_accepts_complex_and_mixed_states():
    """Accept physical local states while enforcing the register site count."""
    parameters = {
        "representation": "logical_product_density",
        "sites": [
            {
                "real": [[0.5, 0.0], [0.0, 0.5]],
                "imag": [[0.0, -0.5], [0.5, 0.0]],
            },
            {
                "real": [[0.5, 0.0], [0.0, 0.5]],
                "imag": [[0.0, 0.0], [0.0, 0.0]],
            },
        ],
    }

    validate_logical_preparation(parameters, number_of_nv=2)

    with pytest.raises(ValueError, match="one preparation record"):
        validate_logical_preparation(parameters, number_of_nv=3)


@pytest.mark.parametrize(
    "real,message",
    [
        ([[0.5, 0.2], [0.0, 0.5]], "Hermitian"),
        ([[1.0, 0.0], [0.0, 1.0]], "unit trace"),
        ([[1.1, 0.0], [0.0, -0.1]], "positive semidefinite"),
    ],
)
def test_logical_preparation_rejects_unphysical_states(real, message):
    """Reject non-Hermitian, unnormalised and negative local states."""
    parameters = {
        "representation": "logical_product_density",
        "sites": [{
            "real": real,
            "imag": [[0.0, 0.0], [0.0, 0.0]],
        }],
    }

    with pytest.raises(ValueError, match=message):
        validate_logical_preparation(parameters, number_of_nv=1)


def test_ou_noise_record_validates_values_and_site_count():
    """Validate OU records while rejecting invalid times and site counts."""
    parameters = {
        "model": "ornstein_uhlenbeck",
        "operator": "Z/2",
        "spatial_correlation": "independent",
        "sites": [{
            "sigma_rad_per_us": 0.1,
            "correlation_time_us": 2.0,
        }],
        "seed": 42,
    }

    validate_noise_parameters(parameters, number_of_nv=1)

    with pytest.raises(ValueError, match="one noise record"):
        validate_noise_parameters(parameters, number_of_nv=2)

    parameters["sites"][0]["correlation_time_us"] = 0.0
    with pytest.raises(ValueError, match="correlation time"):
        validate_noise_parameters(parameters, number_of_nv=1)


def test_noiseless_scenario_rejects_active_noise():
    """Prevent a valid noise record from being silently ignored."""
    source = json.loads(CONFIGURATION.read_text(encoding="utf-8"))
    source["environment"]["noise"] = {
        "status": "specified",
        "parameters": {
            "model": "quasistatic",
            "operator": "Z/2",
            "spatial_correlation": "independent",
            "sites": [
                {"sigma_rad_per_us": 0.1}
                for _ in source["sites"]
            ],
            "seed": 42,
        },
    }

    with pytest.raises(ValueError, match="Active noise is incompatible"):
        longitudinal_scenario_from_dict(source)


def test_uncertainty_validates_distribution_and_parameter_path():
    """Validate scalar input references while rejecting absent parameters."""
    source = json.loads(CONFIGURATION.read_text(encoding="utf-8"))
    parameters = {
        "model": "independent",
        "seed": 42,
        "samples": 10,
        "variables": [{
            "path": ["sites", 0, "bare_detuning"],
            "distribution": "normal",
            "mean": 0.0,
            "std": 0.1,
        }],
    }

    validate_uncertainty_parameters(parameters, specification=source)

    parameters["variables"][0]["path"] = ["sites", 0, "missing"]
    with pytest.raises(ValueError, match="path does not exist"):
        validate_uncertainty_parameters(parameters, specification=source)


def test_uncertainty_rejects_invalid_distribution_width():
    """Reject reversed bounds without drawing random samples."""
    parameters = {
        "model": "independent",
        "seed": 42,
        "samples": 10,
        "variables": [{
            "path": ["sites", 0, "bare_detuning"],
            "distribution": "uniform",
            "low": 0.2,
            "high": -0.2,
        }],
    }

    with pytest.raises(ValueError, match="low < high"):
        validate_uncertainty_parameters(parameters)


def test_control_design_rejects_duplicate_protocols():
    """Reject repeated protocol identifiers in a comparison request."""
    with pytest.raises(ValueError, match="unique"):
        validate_control_design({
            "candidate_protocols": ["free", "free"],
        })


def test_microwave_template_does_not_activate_finite_control():
    """Accept a design template while preserving the active ideal model."""
    source = json.loads(CONFIGURATION.read_text(encoding="utf-8"))
    template = {
        "shape": "sine",
        "duration_us": 0.05,
        "peak_coefficient_rad_per_us": 1.0,
        "carrier_rad_per_us": 18000.0,
        "phase_rad": 0.0,
        "coefficient_convention": "sqrt2_A_cos_Sx",
    }
    design = source["environment"]["control_design"]["parameters"]
    design["microwave_template"] = template

    scenario = longitudinal_scenario_from_dict(source)

    assert scenario.specification["control"]["pulse_model"] == "instantaneous"
    assert scenario.environment["control_design"]["parameters"] == design

    template["duration_us"] = 0.0
    with pytest.raises(ValueError, match="duration_us"):
        longitudinal_scenario_from_dict(source)


def test_numerical_settings_require_valid_time_resolution():
    """Require a positive step count only for discretised propagation."""
    parameters = {
        "method": "static_exponential",
        "comparison_atol": 1e-10,
        "comparison_rtol": 0.0,
    }
    validate_numerical_parameters(parameters)

    parameters["method"] = "piecewise_exponential"
    parameters["steps"] = 0
    with pytest.raises(ValueError, match="positive integer"):
        validate_numerical_parameters(parameters)

    parameters["steps"] = 100
    validate_numerical_parameters(parameters)


def test_evaluation_settings_reject_unknown_metrics():
    """Accept the trial outputs while rejecting unsupported metric requests."""
    source = json.loads(CONFIGURATION.read_text(encoding="utf-8"))
    parameters = source["environment"]["evaluation"]["parameters"]

    validate_evaluation_parameters(parameters)

    parameters["metrics"].append("undefined_metric")
    with pytest.raises(ValueError, match="supported evaluation"):
        validate_evaluation_parameters(parameters)
