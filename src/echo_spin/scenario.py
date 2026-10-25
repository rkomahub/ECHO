import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from echo_spin.nv.registers import (
    NVRegisterConfiguration,
    register_configuration_from_dict,
)


@dataclass
class LongitudinalScenario:
    """Collect a validated ideal-qubit task and its original specification."""

    register: NVRegisterConfiguration
    specification: dict

    @property
    def target_pair(self) -> tuple[int, int]:
        """Return the ordered target-site indices."""
        return tuple(self.specification["task"]["target_pair"])


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
    if set(data) != required:
        raise ValueError("Provide exactly the supported scenario fields.")

    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported scenario schema_version.")

    if data["model"] != "ideal_longitudinal":
        raise ValueError("Only ideal_longitudinal is supported here.")

    register = register_configuration_from_dict(data["register"])
    number_of_nv = register.number_of_nv

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
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("Provide parameter provenance.")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in provenance.values()
    ):
        raise ValueError("Provenance entries must be nonempty strings.")

    return LongitudinalScenario(
        register=register,
        specification=data,
    )


def load_longitudinal_scenario(path) -> LongitudinalScenario:
    """Load an ideal longitudinal scenario from a versioned JSON file."""
    with Path(path).open(encoding="utf-8") as stream:
        specification = json.load(stream)

    return longitudinal_scenario_from_dict(specification)
