from dataclasses import dataclass

import numpy as np
from qutip import Qobj, qeye, sigmax, sigmay, sigmaz, tensor

from echo_spin.characterisation import characterise_longitudinal_scenario
from echo_spin.core.operators import embed_operator, spin_operators
from echo_spin.core.system import SpinSystem
from echo_spin.models.longitudinal import (
    longitudinal_qubit_parameters,
    longitudinal_register_hamiltonian,
)
from echo_spin.scenario import longitudinal_scenario_from_dict


@dataclass
class LongitudinalModel:
    """Collect an ideal register Hamiltonian and consistently ordered Pauli operators."""

    system: SpinSystem
    hamiltonian: Qobj
    pauli_operators: dict
    assessment: dict


def build_longitudinal_model(report) -> LongitudinalModel:
    """Assemble the ideal qubit representation from a Level 2 report."""
    if not isinstance(report, dict):
        raise ValueError("Characterisation report must be a dictionary.")

    required = {
        "model",
        "specification",
        "detunings_rad_per_us",
        "couplings_rad_per_us",
        "constant_rad_per_us",
    }
    if not required.issubset(report):
        raise ValueError("Characterisation report is missing model inputs.")

    if report["model"] != "ideal_longitudinal":
        raise ValueError("Only the ideal longitudinal representation is supported.")

    scenario = longitudinal_scenario_from_dict(report["specification"])
    number_of_nv = scenario.register.number_of_nv

    detunings = report["detunings_rad_per_us"]
    if np.shape(detunings) != (number_of_nv,):
        raise ValueError("Coefficient count must match the scenario site count.")

    hamiltonian = longitudinal_register_hamiltonian(
        detunings=detunings,
        couplings=report["couplings_rad_per_us"],
        constant=report["constant_rad_per_us"],
    )

    system = SpinSystem([0.5] * number_of_nv)
    operators = {
        name: tuple(
            embed_operator(operator, site, system)
            for site in range(number_of_nv)
        )
        for name, operator in (
            ("X", sigmax()),
            ("Y", sigmay()),
            ("Z", sigmaz()),
        )
    }

    return LongitudinalModel(
        system=system,
        hamiltonian=hamiltonian,
        pauli_operators=operators,
        assessment={
            "status": "ideal_design_baseline",
            "tensor_order": "site_0_left",
            "hamiltonian_units": "rad/us",
            "rotation_convention": "exp(-i*theta*Pauli/2)",
            "physical_approximation_validated": False,
            "omitted": [
                "transverse interactions",
                "electronic leakage",
                "nuclear dynamics",
                "noise",
                "finite-pulse effects",
                "shared-drive crosstalk",
            ],
        },
    )


def bare_electronic_embedding(number_of_nv) -> Qobj:
    """Embed ordered logical qubits into electronic ms=0,-1 states."""
    if type(number_of_nv) is not int or number_of_nv < 1:
        raise ValueError("number_of_nv must be a positive integer.")

    local = Qobj(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dims=[[3], [2]],
    )
    return tensor([local] * number_of_nv)


def assess_dipolar_projection(scenario) -> dict:
    """Assess bare dipolar projection for at most three electronic qutrits."""
    number_of_nv = scenario.register.number_of_nv
    if number_of_nv > 3:
        raise ValueError(
            "This reference diagnostic is limited to three electronic qutrits."
        )

    parameters = characterise_longitudinal_scenario(scenario)
    tensors = parameters["dipolar_tensors_rad_per_us"]
    system = SpinSystem([1] * number_of_nv)
    local_operators = spin_operators(1)
    operators = [
        [
            embed_operator(operator, site, system)
            for operator in local_operators
        ]
        for site in range(number_of_nv)
    ]

    interaction = 0.0 * qeye([3] * number_of_nv)
    for i in range(number_of_nv):
        for j in range(i + 1, number_of_nv):
            for alpha in range(3):
                for beta in range(3):
                    interaction += (
                        tensors[i, j, alpha, beta]
                        * operators[i][alpha]
                        * operators[j][beta]
                    )

    embedding = bare_electronic_embedding(number_of_nv)
    projected = embedding.dag() * interaction * embedding

    # Compare dipolar terms only; prescribed bare detunings are excluded.
    detunings, couplings, constant = longitudinal_qubit_parameters(
        np.zeros(number_of_nv),
        tensors,
    )
    longitudinal = longitudinal_register_hamiltonian(
        detunings, couplings, constant
    )

    omitted = projected - longitudinal
    leakage_coupling = interaction * embedding - embedding * projected

    return {
        "embedding": embedding,
        "projected_dipolar_hamiltonian": projected,
        "longitudinal_dipolar_hamiltonian": longitudinal,
        "omitted_in_subspace_norm_rad_per_us": float(
            np.linalg.norm(omitted.full(), ord=2)
        ),
        "discarded_subspace_coupling_norm_rad_per_us": float(
            np.linalg.norm(leakage_coupling.full(), ord=2)
        ),
        "assessment": "Static dipolar diagnostic; no dynamical validity claim.",
    }


def longitudinal_model_report(report, assess_dipolar=False) -> dict:
    """Describe the selected model with optional small-register dipolar diagnostics."""
    if type(assess_dipolar) is not bool:
        raise ValueError("assess_dipolar must be Boolean.")

    model = build_longitudinal_model(report)
    scenario = longitudinal_scenario_from_dict(report["specification"])

    output = {
        "report_version": 1,
        "model": report["model"],
        "specification": report["specification"],
        "subsystem_dimensions": list(model.system.dimensions),
        "dimension": model.system.dimension,
        "detunings_rad_per_us": report["detunings_rad_per_us"],
        "couplings_rad_per_us": report["couplings_rad_per_us"],
        "constant_rad_per_us": report["constant_rad_per_us"],
        "hamiltonian_diagonal_rad_per_us": (
            np.real(model.hamiltonian.diag()).tolist()
        ),
        "operator_conventions": {
            "tensor_order": "site_0_left",
            "computational_basis": "lexicographic binary order",
            "local_operators": "Embedded Pauli X, Y, Z.",
            "rotation": "exp(-i*theta*Pauli/2)",
            "logical_embedding": "logical 0 -> ms=0; logical 1 -> ms=-1",
        },
        "assessment": dict(model.assessment),
        "dipolar_projection": {
            "status": "not_performed",
        },
    }

    if assess_dipolar:
        # Ensure diagnostics and the selected model describe the same inputs.
        expected = characterise_longitudinal_scenario(scenario)
        for name in (
            "detunings_rad_per_us",
            "couplings_rad_per_us",
            "constant_rad_per_us",
        ):
            if not np.allclose(
                report[name], expected[name], atol=1e-12, rtol=1e-12
            ):
                raise ValueError(
                    "Level 2 coefficients do not match its scenario specification."
                )

        diagnostic = assess_dipolar_projection(scenario)

        output["dipolar_projection"] = {
            "status": "static_diagnostic_only",
            "physical_dimension": 3 ** model.system.number_of_spins,
            "logical_dimension": model.system.dimension,
            "omitted_in_subspace_norm_rad_per_us": diagnostic[
                "omitted_in_subspace_norm_rad_per_us"
            ],
            "discarded_subspace_coupling_norm_rad_per_us": diagnostic[
                "discarded_subspace_coupling_norm_rad_per_us"
            ],
            "interpretation": (
                "Dipolar terms only; no energetic or dynamical "
                "suppression assessment. Norms are not gate errors."
            ),
        }

        output["dipolar_projection"]["matrices_rad_per_us"] = {
            name: {
                "real": diagnostic[key].full().real.tolist(),
                "imag": diagnostic[key].full().imag.tolist(),
            }
            for name, key in (
                ("projected", "projected_dipolar_hamiltonian"),
                ("longitudinal", "longitudinal_dipolar_hamiltonian"),
            )
        }

    return output
