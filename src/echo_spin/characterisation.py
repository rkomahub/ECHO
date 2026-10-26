import numpy as np
from scipy.constants import hbar, mu_0

from echo_spin.core.operators import spin_operators
from echo_spin.core.system import SpinSystem
from echo_spin.models.longitudinal import longitudinal_qubit_parameters
from echo_spin.nv.fields import local_zeeman_vector
from echo_spin.nv.hamiltonians import electronic_nv_hamiltonian
from echo_spin.nv.interactions import register_dipolar_tensors
from echo_spin.spectroscopy.transitions import allowed_transitions


def characterise_longitudinal_scenario(scenario) -> dict:
    """Derive distances, dipolar tensors and ideal longitudinal pair timings."""
    register = scenario.register
    sites = scenario.specification["sites"]
    number_of_nv = register.number_of_nv

    differences = (
        register.positions[:, None, :]
        - register.positions[None, :, :]
    )
    distances = np.linalg.norm(differences, axis=2)

    gamma = np.array([site["gamma_e"] for site in sites])
    bare_detunings = np.array([site["bare_detuning"] for site in sites])

    # gamma is in rad/us/T: converting both factors and the result
    # contributes (1e6)**2 * 1e-6 = 1e6.
    tensors = register_dipolar_tensors(
        positions=register.positions,
        local_frames=register.local_frames,
        coupling_prefactor=mu_0 * hbar * 1e6 / (4 * np.pi),
    )
    tensors *= (gamma[:, None] * gamma[None, :])[:, :, None, None]

    detunings, couplings, constant = longitudinal_qubit_parameters(
        bare_detunings,
        tensors,
    )

    off_diagonal = ~np.eye(number_of_nv, dtype=bool)
    coupled = off_diagonal & (couplings != 0.0)

    gate_times = np.full((number_of_nv, number_of_nv), np.inf)
    gate_times[coupled] = np.pi / (4 * np.abs(couplings[coupled]))
    np.fill_diagonal(gate_times, np.nan)

    zero_pairs = np.argwhere(
        np.triu(off_diagonal & ~coupled, k=1)
    )

    return {
        "distances_m": distances,
        "dipolar_tensors_rad_per_us": tensors,
        "detunings_rad_per_us": detunings,
        "couplings_rad_per_us": couplings,
        "constant_rad_per_us": constant,
        "ideal_gate_times_us": gate_times,
        "zero_coupling_pairs": zero_pairs,
    }


def characterise_local_electrons(scenario, strength_threshold=1e-10) -> list:
    """Return isolated-site electronic energies and local-Sx transition strengths."""
    if (
        isinstance(strength_threshold, bool)
        or not isinstance(strength_threshold, (int, float))
        or not np.isfinite(strength_threshold)
        or strength_threshold < 0
    ):
        raise ValueError("strength_threshold must be finite and nonnegative.")

    register = scenario.register
    system = SpinSystem([1])
    sx, _, _ = spin_operators(1)
    results = []

    for index, parameters in enumerate(scenario.specification["sites"]):
        omega = local_zeeman_vector(
            field_lab=register.magnetic_fields[index],
            local_frame=register.local_frames[index],
            gyromagnetic_ratio=parameters["gamma_e"],
        )
        hamiltonian = electronic_nv_hamiltonian(
            system=system,
            electron_site=0,
            D=parameters["D"],
            E=parameters["E"],
            omega_e=omega,
        )
        transitions = allowed_transitions(
            hamiltonian,
            sx,
            threshold=strength_threshold,
        )

        results.append({
            "site": index,
            "zeeman_vector_rad_per_us": omega,
            "energies_rad_per_us": np.asarray(
                hamiltonian.eigenenergies(), dtype=float
            ),
            "control_operator": "local_Sx",
            "strength_threshold": float(strength_threshold),
            "transitions": [
                {
                    "frequency_rad_per_us": float(frequency),
                    "strength": float(strength),
                    "lower_eigenstate": int(lower),
                    "upper_eigenstate": int(upper),
                }
                for frequency, strength, lower, upper in transitions
            ],
        })

    return results


def longitudinal_characterisation_report(scenario) -> dict:
    """Collect JSON-compatible register interactions and local electronic spectra."""
    interactions = characterise_longitudinal_scenario(scenario)
    spectra = characterise_local_electrons(scenario)
    times = interactions["ideal_gate_times_us"]

    return {
        "report_version": 1,
        "model": scenario.specification["model"],
        "specification": scenario.specification,
        "units": {
            "distances": "m",
            "dipolar_tensors": "rad/us",
            "detunings": "rad/us",
            "couplings": "rad/us",
            "constant": "rad/us",
            "ideal_gate_times": "us",
            "electronic_energies": "rad/us",
            "transition_frequencies": "rad/us",
            "transition_strengths": "dimensionless",
        },
        "distances_m": interactions["distances_m"].tolist(),
        "dipolar_tensors_rad_per_us": (
            interactions["dipolar_tensors_rad_per_us"].tolist()
        ),
        "detunings_rad_per_us": (
            interactions["detunings_rad_per_us"].tolist()
        ),
        "couplings_rad_per_us": (
            interactions["couplings_rad_per_us"].tolist()
        ),
        "constant_rad_per_us": interactions["constant_rad_per_us"],
        "ideal_gate_times_us": [
            [
                float(value) if np.isfinite(value) else None
                for value in row
            ]
            for row in times
        ],
        "zero_coupling_pairs": interactions["zero_coupling_pairs"].tolist(),
        "local_electrons": [
            {
                **spectrum,
                "zeeman_vector_rad_per_us": (
                    spectrum["zeeman_vector_rad_per_us"].tolist()
                ),
                "energies_rad_per_us": (
                    spectrum["energies_rad_per_us"].tolist()
                ),
            }
            for spectrum in spectra
        ],
        "conventions": {
            "gate_times": (
                "pi/(4*abs(J)); null on the diagonal and for zero couplings."
            ),
            "zero_coupling": "Exact numerical zero; no near-zero cutoff.",
            "logical_projection": (
                "Bare ms=0,-1 encoding; only local zz dipolar components retained."
            ),
            "spectroscopy": (
                "Isolated electronic sites; diagnostic local Sx operator."
            ),
        },
        "limits": [
            "Hyperfine and inter-site spectral shifts are excluded.",
            "Transition strengths are not fluorescence contrasts.",
            "No physical shared-drive model or bandwidth comparison is included.",
            "Individual addressability is not established.",
            "The longitudinal approximation is assumed, not validated here.",
        ],
        "spectral_overlap": local_spectral_overlap(spectra),
    }


def local_spectral_overlap(spectra, tolerance_rad_per_us=1e-8) -> dict:
    """Compare retained isolated-site lines without asserting addressability."""
    if (
        isinstance(tolerance_rad_per_us, bool)
        or not isinstance(tolerance_rad_per_us, (int, float))
        or not np.isfinite(tolerance_rad_per_us)
        or tolerance_rad_per_us < 0
    ):
        raise ValueError("Spectral tolerance must be finite and nonnegative.")

    number_of_nv = len(spectra)
    minimum_gaps = [
        [None for _ in range(number_of_nv)]
        for _ in range(number_of_nv)
    ]
    coincident_pairs = []

    for i in range(number_of_nv):
        frequencies_i = [
            line["frequency_rad_per_us"]
            for line in spectra[i]["transitions"]
        ]
        for j in range(i + 1, number_of_nv):
            frequencies_j = [
                line["frequency_rad_per_us"]
                for line in spectra[j]["transitions"]
            ]
            if not frequencies_i or not frequencies_j:
                continue

            gap = float(np.min(np.abs(
                np.asarray(frequencies_i)[:, None]
                - np.asarray(frequencies_j)[None, :]
            )))
            minimum_gaps[i][j] = gap
            minimum_gaps[j][i] = gap

            if gap <= tolerance_rad_per_us:
                coincident_pairs.append([
                    int(spectra[i]["site"]),
                    int(spectra[j]["site"]),
                ])

    return {
        "site_order": [int(spectrum["site"]) for spectrum in spectra],
        "minimum_gaps_rad_per_us": minimum_gaps,
        "coincidence_tolerance_rad_per_us": float(tolerance_rad_per_us),
        "coincident_site_pairs": coincident_pairs,
        "interpretation": (
            "All retained isolated-site lines are compared. "
            "Null gaps mean self-comparison or no retained lines. "
            "Nonzero gaps do not establish physical addressability."
        ),
    }
