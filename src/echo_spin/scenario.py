import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from echo_spin.nv.registers import (
    NVRegisterConfiguration,
    register_configuration_from_dict,
)

ENVIRONMENT_SECTIONS = (
    "nuclear_parameters",
    "preparation",
    "noise",
    "uncertainty",
    "control_design",
    "constraints",
    "numerics",
    "evaluation",
)


def _finite_real(value) -> bool:
    """Recognise finite JSON real numbers while excluding booleans."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and np.isfinite(value)
    )


def validate_nuclear_parameters(parameters, number_of_nv=None):
    """Validate ordered nitrogen records in the NV local-frame convention."""
    if not isinstance(parameters, dict) or set(parameters) != {"sites"}:
        raise ValueError("Nuclear parameters require a sites list.")

    sites = parameters["sites"]
    if not isinstance(sites, list) or not sites:
        raise ValueError("Provide a nonempty nuclear sites list.")

    if number_of_nv is not None and len(sites) != number_of_nv:
        raise ValueError("Provide one nuclear record per NV.")

    required = {
        "isotope",
        "spin",
        "gamma_n",
        "Q",
        "A",
        "frame",
        "source",
    }
    for site in sites:
        if not isinstance(site, dict) or set(site) != required:
            raise ValueError("Provide all supported nuclear record fields.")

        isotope = site["isotope"]
        if isotope not in ("14N", "15N"):
            raise ValueError("Supported nitrogen isotopes are 14N and 15N.")

        expected_spin = 1.0 if isotope == "14N" else 0.5
        if (
            not _finite_real(site["spin"])
            or site["spin"] != expected_spin
        ):
            raise ValueError("Nuclear spin must match the nitrogen isotope.")

        if (
            not _finite_real(site["gamma_n"])
            or site["gamma_n"] == 0
            or not _finite_real(site["Q"])
        ):
            raise ValueError("Require finite Q and finite nonzero gamma_n.")

        if isotope == "15N" and site["Q"] != 0:
            raise ValueError("Use Q=0 for the spin-half 15N record.")

        tensor = site["A"]
        if (
            not isinstance(tensor, list)
            or len(tensor) != 3
            or any(
                not isinstance(row, list)
                or len(row) != 3
                or not all(_finite_real(value) for value in row)
                for row in tensor
            )
        ):
            raise ValueError("A must be a finite real 3x3 tensor.")

        if site["frame"] != "nv_local":
            raise ValueError("Hyperfine components must use frame='nv_local'.")

        if not isinstance(site["source"], str) or not site["source"].strip():
            raise ValueError("Provide a nuclear parameter source.")


def validate_static_constraints(parameters, register=None) -> dict:
    """Validate optional separation and site-field bounds for any register size."""
    supported = {"minimum_separation_m", "maximum_field_t"}

    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("Specify at least one supported static constraint.")

    if set(parameters) - supported:
        raise ValueError("Unsupported static constraint.")

    for name, value in parameters.items():
        if not _finite_real(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")

    checks = {
        "minimum_separation": "Not specified.",
        "maximum_site_field": "Not specified.",
    }
    if register is None:
        return checks

    if "minimum_separation_m" in parameters:
        positions = register.positions
        if len(positions) < 2:
            checks["minimum_separation"] = "Not applicable: fewer than two sites."
        else:
            i, j = np.triu_indices(len(positions), k=1)
            distances = np.linalg.norm(positions[i] - positions[j], axis=1)
            if np.any(distances < parameters["minimum_separation_m"]):
                raise ValueError("Register violates minimum_separation_m.")
            checks["minimum_separation"] = "Passed."

    if "maximum_field_t" in parameters:
        magnitudes = np.linalg.norm(register.magnetic_fields, axis=1)
        if np.any(magnitudes > parameters["maximum_field_t"]):
            raise ValueError("Register violates maximum_field_t.")
        checks["maximum_site_field"] = "Passed."

    return checks


def validate_logical_preparation(parameters, number_of_nv=None):
    """Validate ordered logical product-state density matrices."""
    if (
        not isinstance(parameters, dict)
        or set(parameters) != {"representation", "sites"}
        or parameters["representation"] != "logical_product_density"
    ):
        raise ValueError(
            "Preparation requires representation='logical_product_density' "
            "and sites."
        )

    sites = parameters["sites"]
    if not isinstance(sites, list) or not sites:
        raise ValueError("Provide a nonempty preparation sites list.")

    if number_of_nv is not None and len(sites) != number_of_nv:
        raise ValueError("Provide one preparation record per NV.")

    for site in sites:
        if not isinstance(site, dict) or set(site) != {"real", "imag"}:
            raise ValueError("Each preparation requires real and imag arrays.")

        components = []
        for name in ("real", "imag"):
            values = site[name]
            if (
                not isinstance(values, list)
                or len(values) != 2
                or any(
                    not isinstance(row, list)
                    or len(row) != 2
                    or not all(_finite_real(value) for value in row)
                    for row in values
                )
            ):
                raise ValueError(
                    "Preparation components must be finite real 2x2 arrays."
                )
            components.append(np.asarray(values, dtype=float))

        density = components[0] + 1j * components[1]
        tolerance = 1e-10

        if not np.allclose(
            density, density.conj().T, atol=tolerance, rtol=0.0
        ):
            raise ValueError("Preparation density matrix must be Hermitian.")

        if not np.isclose(
            np.trace(density), 1.0, atol=tolerance, rtol=0.0
        ):
            raise ValueError("Preparation density matrix must have unit trace.")

        hermitian = (density + density.conj().T) / 2
        if np.linalg.eigvalsh(hermitian).min() < -tolerance:
            raise ValueError(
                "Preparation density matrix must be positive semidefinite."
            )


def validate_noise_parameters(parameters, number_of_nv=None):
    """Validate independent local dephasing records without generating noise."""
    if not isinstance(parameters, dict):
        raise ValueError("Noise parameters must be a JSON object.")

    model = parameters.get("model")
    if model == "none":
        if set(parameters) != {"model"}:
            raise ValueError("Noise model 'none' accepts no extra parameters.")
        return

    if model not in ("quasistatic", "ornstein_uhlenbeck"):
        raise ValueError("Unsupported noise model.")

    required = {
        "model", "operator", "spatial_correlation", "sites", "seed"
    }
    if set(parameters) != required:
        raise ValueError("Provide exactly the supported noise fields.")

    if parameters["operator"] != "Z/2":
        raise ValueError("Local dephasing requires operator='Z/2'.")

    if parameters["spatial_correlation"] != "independent":
        raise ValueError("Only independent site noise is supported here.")

    seed = parameters["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("Noise seed must be a nonnegative integer.")

    sites = parameters["sites"]
    if not isinstance(sites, list) or not sites:
        raise ValueError("Provide a nonempty noise sites list.")

    if number_of_nv is not None and len(sites) != number_of_nv:
        raise ValueError("Provide one noise record per NV.")

    fields = {"sigma_rad_per_us"}
    if model == "ornstein_uhlenbeck":
        fields.add("correlation_time_us")

    for site in sites:
        if not isinstance(site, dict) or set(site) != fields:
            raise ValueError("Provide exactly the required site noise fields.")

        sigma = site["sigma_rad_per_us"]
        if not _finite_real(sigma) or sigma < 0:
            raise ValueError("Noise sigma must be finite and nonnegative.")

        if model == "ornstein_uhlenbeck":
            correlation_time = site["correlation_time_us"]
            if not _finite_real(correlation_time) or correlation_time <= 0:
                raise ValueError("Noise correlation time must be positive.")


def validate_uncertainty_parameters(parameters, specification=None):
    """Validate independent parameter distributions and optional input paths."""
    if not isinstance(parameters, dict):
        raise ValueError("Uncertainty parameters must be a JSON object.")

    if parameters.get("model") == "deterministic":
        if set(parameters) != {"model"}:
            raise ValueError("Deterministic uncertainty accepts no extra fields.")
        return

    if parameters.get("model") != "independent":
        raise ValueError("Unsupported uncertainty model.")

    if set(parameters) != {"model", "seed", "samples", "variables"}:
        raise ValueError("Provide exactly the supported uncertainty fields.")

    if type(parameters["seed"]) is not int or parameters["seed"] < 0:
        raise ValueError("Uncertainty seed must be a nonnegative integer.")

    if type(parameters["samples"]) is not int or parameters["samples"] < 1:
        raise ValueError("Uncertainty samples must be a positive integer.")

    variables = parameters["variables"]
    if not isinstance(variables, list) or not variables:
        raise ValueError("Provide a nonempty uncertainty variables list.")

    seen = set()
    for variable in variables:
        if not isinstance(variable, dict):
            raise ValueError("Each uncertainty variable must be an object.")

        distribution = variable.get("distribution")
        if distribution == "normal":
            required = {"path", "distribution", "mean", "std"}
        elif distribution == "uniform":
            required = {"path", "distribution", "low", "high"}
        else:
            raise ValueError("Unsupported uncertainty distribution.")

        if set(variable) != required:
            raise ValueError("Provide exactly the distribution fields.")

        path = variable["path"]
        if (
            not isinstance(path, list)
            or len(path) < 2
            or path[0] not in ("register", "sites")
            or any(
                not isinstance(part, str) and type(part) is not int
                for part in path
            )
            or any(type(part) is int and part < 0 for part in path)
        ):
            raise ValueError("Invalid uncertainty parameter path.")

        key = tuple(path)
        if key in seen:
            raise ValueError("Duplicate uncertainty parameter path.")
        seen.add(key)

        values = [variable[name] for name in required - {"path", "distribution"}]
        if not all(_finite_real(value) for value in values):
            raise ValueError("Distribution parameters must be finite real.")

        if distribution == "normal" and variable["std"] <= 0:
            raise ValueError("Normal uncertainty requires positive std.")

        if distribution == "uniform" and variable["low"] >= variable["high"]:
            raise ValueError("Uniform uncertainty requires low < high.")

        if specification is not None:
            value = specification
            for part in path:
                if isinstance(value, dict) and isinstance(part, str):
                    if part not in value:
                        raise ValueError("Uncertainty path does not exist.")
                    value = value[part]
                elif isinstance(value, list) and type(part) is int:
                    if part >= len(value):
                        raise ValueError("Uncertainty path does not exist.")
                    value = value[part]
                else:
                    raise ValueError("Uncertainty path does not exist.")

            if not _finite_real(value):
                raise ValueError("Uncertainty path must identify a real scalar.")


def validate_environment(environment, number_of_nv=None) -> dict:
    """Validate and copy descriptive records without claiming simulation support."""
    if not isinstance(environment, dict):
        raise ValueError("environment must be a JSON object.")

    unknown = set(environment) - set(ENVIRONMENT_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown environment sections: {sorted(unknown)}.")

    resolved = {}
    for name in ENVIRONMENT_SECTIONS:
        record = environment.get(name, {
            "status": "unresolved",
            "parameters": {},
        })
        if not isinstance(record, dict) or set(record) != {
            "status", "parameters"
        }:
            raise ValueError(
                f"{name} requires status and parameters."
            )

        if record["status"] not in (
            "specified", "unresolved", "not_applicable"
        ):
            raise ValueError(f"Invalid status for {name}.")

        if not isinstance(record["parameters"], dict):
            raise ValueError(f"{name}.parameters must be a JSON object.")

        if record["status"] != "specified" and record["parameters"]:
            raise ValueError(
                f"{name} parameters require status='specified'."
            )

        if record["status"] == "specified" and not record["parameters"]:
            raise ValueError(
                f"Specified section {name} requires parameters."
            )

        try:
            json.dumps(record, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{name} must contain finite JSON-compatible values."
            ) from error

        if name == "nuclear_parameters" and record["status"] == "specified":
            validate_nuclear_parameters(
                record["parameters"],
                number_of_nv=number_of_nv,
            )

        if name == "constraints" and record["status"] == "specified":
            validate_static_constraints(record["parameters"])

        if name == "preparation" and record["status"] == "specified":
            validate_logical_preparation(
                record["parameters"],
                number_of_nv=number_of_nv,
            )

        if name == "noise" and record["status"] == "specified":
            validate_noise_parameters(
                record["parameters"],
                number_of_nv=number_of_nv,
            )

        if name == "uncertainty" and record["status"] == "specified":
            validate_uncertainty_parameters(record["parameters"])

        resolved[name] = deepcopy(record)

    return resolved


@dataclass
class LongitudinalScenario:
    """Collect a validated ideal-qubit task and its original specification."""

    register: NVRegisterConfiguration
    specification: dict

    @property
    def target_pair(self) -> tuple[int, int]:
        """Return the ordered target-site indices."""
        return tuple(self.specification["task"]["target_pair"])

    @property
    def environment(self) -> dict:
        """Return general records with omitted sections explicitly unresolved."""
        return validate_environment(
            self.specification.get("environment", {}),
            number_of_nv=self.register.number_of_nv,
        )

    @property
    def constraint_checks(self) -> dict:
        """Report static-bound checks separately from experimental feasibility."""
        record = self.environment["constraints"]
        if record["status"] != "specified":
            return {
                "minimum_separation": "Not specified.",
                "maximum_site_field": "Not specified.",
            }

        return validate_static_constraints(
            record["parameters"],
            register=self.register,
        )


def longitudinal_scenario_from_dict(specification) -> LongitudinalScenario:
    """Validate the supported ideal longitudinal scenario specification."""
    if not isinstance(specification, dict):
        raise ValueError("The scenario must be a JSON object.")

    data = deepcopy(specification)
    required = {
        "schema_version",
        "register",
        "model",
        "sites",
        "task",
        "control",
        "provenance",
    }
    if (
        not required.issubset(data)
        or set(data) - required - {"environment"}
    ):
        raise ValueError("Provide exactly the supported scenario fields.")

    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported scenario schema_version.")

    if data["model"] != "ideal_longitudinal":
        raise ValueError("Only ideal_longitudinal is supported here.")

    register = register_configuration_from_dict(data["register"])
    number_of_nv = register.number_of_nv

    environment = validate_environment(
        data.get("environment", {}),
        number_of_nv=number_of_nv,
    )

    constraints = environment["constraints"]
    if constraints["status"] == "specified":
        validate_static_constraints(
            constraints["parameters"],
            register=register,
        )

    uncertainty = environment["uncertainty"]
    if uncertainty["status"] == "specified":
        validate_uncertainty_parameters(
            uncertainty["parameters"],
            specification=data,
        )

    sites = data["sites"]
    if not isinstance(sites, list) or len(sites) != number_of_nv:
        raise ValueError("Provide one physical parameter record per NV.")

    for site in sites:
        if not isinstance(site, dict) or set(site) != {
            "D", "E", "gamma_e", "bare_detuning"
        }:
            raise ValueError(
                "Each site requires D, E, gamma_e and bare_detuning."
            )

        for value in site.values():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError("Site parameters must be finite real numbers.")

        if site["D"] <= 0 or site["gamma_e"] == 0:
            raise ValueError("Require positive D and nonzero gamma_e.")

    task = data["task"]
    expected_task = {
        "target_pair",
        "target",
        "spectators",
        "encoding",
        "tensor_order",
        "detuning_convention",
        "local_corrections",
    }
    if not isinstance(task, dict) or set(task) != expected_task:
        raise ValueError("Provide exactly the supported task fields.")

    pair = task["target_pair"]
    if (
        not isinstance(pair, list)
        or len(pair) != 2
        or any(type(index) is not int for index in pair)
        or pair[0] == pair[1]
        or any(index < 0 or index >= number_of_nv for index in pair)
    ):
        raise ValueError("target_pair must contain two distinct valid indices.")

    conventions = {
        "target": "signed_sqrt_zz",
        "spectators": "identity",
        "encoding": "ms0_msminus1",
        "tensor_order": "site_0_left",
        "detuning_convention": "reference_minus_transition",
        "local_corrections": "none",
    }
    for name, expected in conventions.items():
        if task[name] != expected:
            raise ValueError(f"Unsupported task setting: {name}.")

    if data["control"] != {
        "pulse_model": "instantaneous",
        "addressing": "independent_ideal",
        "noise": "none",
    }:
        raise ValueError("Unsupported control assumptions.")

    provenance = data["provenance"]
    required_provenance = {
        "geometry_and_field",
        "D",
        "E",
        "gamma_e",
        "bare_detuning",
        "units",
    }
    if (
        not isinstance(provenance, dict)
        or not required_provenance.issubset(provenance)
    ):
        raise ValueError("Provide provenance for every required parameter family.")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in provenance.values()
    ):
        raise ValueError("Provenance entries must be nonempty strings.")

    noise = environment["noise"]
    if (
        noise["status"] == "specified"
        and noise["parameters"]["model"] != "none"
    ):
        raise ValueError(
            "Active noise is incompatible with the current noise='none' "
            "control setting; noisy scenario execution is not integrated."
        )

    return LongitudinalScenario(
        register=register,
        specification=data,
    )


def load_longitudinal_scenario(path) -> LongitudinalScenario:
    """Load an ideal longitudinal scenario from a versioned JSON file."""
    with Path(path).open(encoding="utf-8") as stream:
        specification = json.load(stream)

    return longitudinal_scenario_from_dict(specification)
