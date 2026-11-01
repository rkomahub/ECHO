from itertools import product

import numpy as np
from qutip import Qobj, basis, qeye, tensor

from echo_spin.control.microwave import control_amplitude
from echo_spin.control.pulses import sine_envelope
from echo_spin.control.rotations import single_qubit_rotation
from echo_spin.core.operators import embed_operator, spin_operators
from echo_spin.core.system import SpinSystem
from echo_spin.dynamics.propagators import (
    hermitian_propagator,
    hermitian_time_dependent_propagator,
    static_propagator,
)
from echo_spin.dynamics.sequences import hermitian_pulse_propagator
from echo_spin.implementation import implement_finite_protocols
from echo_spin.models.longitudinal import (
    longitudinal_qubit_parameters,
    longitudinal_register_hamiltonian,
)
from echo_spin.nv.fields import local_zeeman_vector
from echo_spin.representation import build_electronic_reference


def electronic_population_diagnostics(scenario, times_us) -> dict:
    """Separate logical population transfer and leakage under full electronic drift."""
    times = np.asarray(times_us)
    if (
        times.ndim != 1
        or times.size == 0
        or times.dtype.kind not in "iuf"
        or not np.all(np.isfinite(times))
        or np.any(times < 0)
    ):
        raise ValueError("Provide finite nonnegative times in microseconds.")

    reference = build_electronic_reference(scenario)
    local = reference["local_hamiltonian"]
    full = reference["hamiltonian"]
    embedding = reference["embedding"]
    projector = embedding * embedding.dag()

    local_escape = local * embedding - projector * local * embedding
    if local_escape.norm("fro") > 1e-10:
        raise ValueError(
            "These bare-state diagnostics require an invariant local subspace."
        )

    dimension = embedding.shape[1]
    weights = np.array([
        bin(index).count("1") for index in range(dimension)
    ])
    same_weight = weights[:, None] == weights[None, :]
    exchange_mask = same_weight & ~np.eye(dimension, dtype=bool)
    parameters = longitudinal_qubit_parameters(
        np.zeros(reference["system"].number_of_spins),
        reference["dipolar_tensors_rad_per_us"],
    )
    longitudinal = longitudinal_register_hamiltonian(*parameters)
    records = []

    for time in times:
        interaction_propagator = (
            static_propagator(local, -float(time))
            * static_propagator(full, float(time))
        )
        evolved = interaction_propagator * embedding
        logical = embedding.dag() * evolved
        probabilities = np.abs(logical.full())**2
        outside = evolved - embedding * logical
        predicted = static_propagator(longitudinal, float(time))
        overlap = complex((predicted.dag() * logical).tr())
        comparison_fidelity = (
            logical.norm("fro")**2 + abs(overlap)**2
        ) / (dimension * (dimension + 1))

        records.append({
            "time_us": float(time),
            "mean_stay_probability": float(
                np.trace(probabilities) / dimension
            ),
            "mean_exchange_probability": float(
                probabilities[exchange_mask].sum() / dimension
            ),
            "mean_other_logical_transfer_probability": float(
                probabilities[~same_weight].sum() / dimension
            ),
            "mean_leakage_probability": float(
                outside.norm("fro")**2 / dimension
            ),
            "longitudinal_comparison_fidelity": float(comparison_fidelity),
        })

    return {
        "physical_dimension": full.shape[0],
        "logical_dimension": dimension,
        "frame": "Exact interaction picture of the local electronic Hamiltonian.",
        "average": {
            "populations": "Uniform over logical computational-basis inputs.",
            "comparison_fidelity": "Normalized Haar average; no postselection.",
        },
        "records": records,
        "limits": [
            "Bare logical subspace must be invariant under local evolution.",
            "No controls, nuclei or noise.",
            "Population diagnostics do not quantify all phase or gate errors.",
        ],
    }


def electronic_magnetic_drive_operator(scenario, field_profile_tesla):
    """Construct the electronic coupling to a laboratory microwave field profile."""
    number_of_nv = scenario.register.number_of_nv
    if number_of_nv > 3:
        raise ValueError("This electronic reference supports at most three sites.")

    if np.iscomplexobj(field_profile_tesla):
        raise ValueError("field_profile_tesla must be real.")

    fields = np.asarray(field_profile_tesla, dtype=float)
    if (
        fields.shape != (number_of_nv, 3)
        or not np.all(np.isfinite(fields))
    ):
        raise ValueError("Provide a finite laboratory field vector per site.")

    system = SpinSystem([1] * number_of_nv)
    operator = 0.0 * qeye([3] * number_of_nv)

    for site, parameters in enumerate(scenario.specification["sites"]):
        coefficients = local_zeeman_vector(
            field_lab=fields[site],
            local_frame=scenario.register.local_frames[site],
            gyromagnetic_ratio=parameters["gamma_e"],
        )
        for coefficient, spin in zip(coefficients, spin_operators(1)):
            operator += coefficient * embed_operator(spin, site, system)

    return operator


def local_square_pulse_populations(spectra, carrier, duration_us) -> list:
    """Evaluate isolated three-level RWA populations after a shared square pulse."""
    if (
        not np.isfinite(carrier)
        or carrier <= 0
        or not np.isfinite(duration_us)
        or duration_us <= 0
    ):
        raise ValueError("Provide a positive finite carrier and duration.")

    records = []
    for site in spectra:
        if site["control_operator"] != "physical_microwave_profile":
            raise ValueError("Physical microwave matrix elements are required.")

        energies = np.asarray(site["energies_rad_per_us"])
        matrix = np.diag(
            energies - energies[0] - np.array([0, carrier, carrier])
        ).astype(complex)

        for transition in site["transitions"]:
            if transition["lower_eigenstate"] == 0:
                upper = transition["upper_eigenstate"]
                coupling = np.sqrt(transition["strength"]) / 2
                matrix[upper, 0] = coupling
                matrix[0, upper] = coupling

        state = (
            static_propagator(Qobj(matrix), duration_us) * basis(3, 0)
        )
        records.append({
            "site": site["site"],
            "eigenstate_populations": (
                np.abs(state.full().ravel())**2
            ).tolist(),
        })

    return records


def build_shared_drive_rwa(
    scenario, field_profile_tesla, carrier, phase=0.0
) -> dict:
    """Build a common-carrier electronic RWA retaining excitation-conserving exchange."""
    if (
        not np.isfinite(carrier)
        or carrier <= 0
        or not np.isfinite(phase)
    ):
        raise ValueError("Provide a positive finite carrier and finite phase.")

    reference = build_electronic_reference(scenario)
    number_of_nv = reference["system"].number_of_spins
    local = reference["local_hamiltonian"]

    local_bases = []
    for site in range(number_of_nv):
        # Partial tracing adds only an identity shift and a positive scale
        # to the local Hamiltonian, preserving its ordered eigenvectors.
        _, states = local.ptrace(site).eigenstates()
        local_bases.append(Qobj(
            np.column_stack([state.full().ravel() for state in states]),
            dims=[[3], [3]],
        ))
    basis_to_lab = tensor(local_bases)

    local_energy = basis_to_lab.dag() * local * basis_to_lab
    interaction = (
        basis_to_lab.dag()
        * reference["dipolar_hamiltonian"]
        * basis_to_lab
    )
    magnetic = (
        basis_to_lab.dag()
        * electronic_magnetic_drive_operator(
            scenario, field_profile_tesla
        )
        * basis_to_lab
    )

    excitations = np.array([
        sum(level != 0 for level in levels)
        for levels in product(range(3), repeat=number_of_nv)
    ])
    difference = excitations[:, None] - excitations[None, :]
    dimensions = local_energy.dims

    number_operator = Qobj(
        np.diag(excitations), dims=dimensions
    )
    interaction_rwa = Qobj(
        interaction.full() * (difference == 0), dims=dimensions
    )
    raising = Qobj(
        magnetic.full() * (difference == 1), dims=dimensions
    )
    drive = (
        np.exp(-1j * phase) * raising
        + np.exp(1j * phase) * raising.dag()
    ) / 2
    local_rotating = local_energy - carrier * number_operator

    return {
        "drift_hamiltonian": local_rotating + interaction_rwa,
        "local_rotating_hamiltonian": local_rotating,
        "dipolar_hamiltonian": interaction_rwa,
        "drive_operator": drive,
        "excitation_operator": number_operator,
        "basis_to_lab": basis_to_lab,
        "carrier_rad_per_us": float(carrier),
        "phase_rad": float(phase),
        "frame": "Common-carrier rotating frame in local energy eigenbases.",
        "approximation": "Excitation-number RWA; transverse exchange retained.",
        "raising_operator": raising,
        "specification": scenario.specification,
    }


def shared_drive_propagator(model, duration_us, *, envelope=None, steps=128):
    """Propagate the shared-drive RWA with a constant or supplied envelope."""
    if not np.isfinite(duration_us) or duration_us < 0:
        raise ValueError("duration_us must be finite and nonnegative.")

    if envelope is None or duration_us == 0:
        return hermitian_propagator(
            model["drift_hamiltonian"] + model["drive_operator"],
            duration_us,
        )
    return hermitian_pulse_propagator(
        model["drift_hamiltonian"],
        model["drive_operator"],
        duration_us,
        envelope,
        steps=steps,
    )


def shared_pulse_populations(
    scenario,
    field_profile_tesla,
    carrier,
    duration_us,
    *,
    envelope=None,
    steps=128,
) -> dict:
    """Evaluate interacting-register populations under a shared pulse envelope."""
    if not np.isfinite(duration_us) or duration_us < 0:
        raise ValueError("duration_us must be finite and nonnegative.")

    model = build_shared_drive_rwa(
        scenario, field_profile_tesla, carrier
    )

    number_of_nv = scenario.register.number_of_nv

    initial = tensor([basis(3, 0)] * number_of_nv)

    propagator = shared_drive_propagator(
        model, duration_us, envelope=envelope, steps=steps
    )
    state = propagator * initial

    configurations = list(product(range(3), repeat=number_of_nv))
    probabilities = np.abs(state.full().ravel())**2
    outside = np.array([
        any(level == 2 for level in levels)
        for levels in configurations
    ])

    target_site = scenario.target_pair[0]
    target = tensor([
        basis(3, 1 if site == target_site else 0)
        for site in range(number_of_nv)
    ])

    return {
        "duration_us": float(duration_us),
        "target_site": target_site,
        "target_population_probability": float(
            abs(target.overlap(state))**2
        ),
        "leakage_probability": float(probabilities[outside].sum()),
        "total_probability": float(probabilities.sum()),
        "local_populations": [
            {
                "site": site,
                "eigenstate_populations": (
                    state.ptrace(site).diag().real.tolist()
                ),
            }
            for site in range(number_of_nv)
        ],
        "model": model["approximation"],
        "initial_state": "Product of local lowest-energy eigenstates.",
        "envelope": "square" if envelope is None else "supplied_callable",
        "integration_steps": (
            int(steps) if envelope is not None and duration_us > 0 else None
        ),
    }


def shared_square_pulse_populations(
    scenario, field_profile_tesla, carrier, duration_us
) -> dict:
    """Evaluate a square pulse through the general shared-envelope interface."""
    return shared_pulse_populations(
        scenario, field_profile_tesla, carrier, duration_us
    )


def shared_pulse_gate_diagnostics(
    scenario,
    field_profile_tesla,
    carrier,
    duration_us,
    *,
    envelope=None,
    steps=128,
) -> dict:
    """Evaluate a local pi-X target on the whole logical register without postselection."""
    model = build_shared_drive_rwa(
        scenario, field_profile_tesla, carrier
    )
    propagator = shared_drive_propagator(
        model, duration_us, envelope=envelope, steps=steps
    )
    interaction_propagator = (
        hermitian_propagator(
            model["local_rotating_hamiltonian"], -duration_us
        ) * propagator
    )

    number_of_nv = scenario.register.number_of_nv
    local_embedding = Qobj(
        [[1, 0], [0, 1], [0, 0]], dims=[[3], [2]]
    )
    embedding = tensor([local_embedding] * number_of_nv)
    evolved = interaction_propagator * embedding
    logical = embedding.dag() * evolved
    outside = evolved - embedding * logical

    system = SpinSystem([0.5] * number_of_nv)
    target_site = scenario.target_pair[0]
    target = embed_operator(
        single_qubit_rotation(np.pi, "x"), target_site, system
    )
    dimension = system.dimension
    overlap = complex((target.dag() * logical).tr())
    fidelity = (
        logical.norm("fro")**2 + abs(overlap)**2
    ) / (dimension * (dimension + 1))

    return {
        "duration_us": float(duration_us),
        "target_site": target_site,
        "average_gate_fidelity": float(fidelity),
        "average_leakage": float(outside.norm("fro")**2 / dimension),
        "frame": "Interaction picture of the local rotating Hamiltonian.",
        "target": "pi_X on the target site; identity on spectators.",
        "postselection": False,
        "physical_local_corrections": False,
        "model": model["approximation"],
    }


def shared_multitone_hamiltonian(time, model, tones):
    """Evaluate shared microwave tones in the common-carrier electronic RWA."""
    if not np.isfinite(time):
        raise ValueError("time must be finite.")

    hamiltonian = model["drift_hamiltonian"].copy()
    raising = model["raising_operator"]

    for tone in tones:
        carrier = tone["carrier_rad_per_us"]
        phase = tone.get("phase_rad", 0.0)
        amplitude = control_amplitude(tone.get("envelope", 1.0), time)

        if (
            any(np.iscomplexobj(value) for value in (carrier, phase, amplitude))
            or not np.all(np.isfinite([carrier, phase, amplitude]))
            or carrier <= 0
        ):
            raise ValueError("Provide finite real tone parameters and positive carriers.")

        angle = (
            (carrier - model["carrier_rad_per_us"]) * time + phase
        )
        coefficient = amplitude * np.exp(-1j * angle) / 2
        hamiltonian += (
            coefficient * raising + coefficient.conjugate() * raising.dag()
        )

    return hamiltonian


def compile_shared_selective_events(
    protocol_report,
    model,
    pulse_duration_us,
    *,
    phase_cycle="xy8_targets",
    protocol_name="hadamard_selective",
) -> dict:
    """Compile selective pulse events into calibrated shared microwave tones."""
    if (
        protocol_report["source_model"]["specification"]
        != model["specification"]
    ):
        raise ValueError("Protocol and physical model describe different scenarios.")

    compiled = implement_finite_protocols(
        protocol_report,
        pulse_duration_us,
        phase_cycle=phase_cycle,
    )
    if protocol_name not in compiled["protocols"]:
        raise ValueError("Unknown protocol name.")
    control = compiled["protocols"][protocol_name]
    number_of_nv = len(model["drift_hamiltonian"].dims[0])
    diagonal = model["local_rotating_hamiltonian"].diag().real
    raising = model["raising_operator"].full()
    transitions = {}

    for site in range(number_of_nv):
        index = 3**(number_of_nv - site - 1)
        element = raising[index, 0]
        if abs(element) == 0:
            raise ValueError(f"The selected transition at site {site} is dark.")

        transitions[site] = {
            "carrier_rad_per_us": float(
                model["carrier_rad_per_us"]
                + diagonal[index] - diagonal[0]
            ),
            "phase_rad": float(np.angle(element)),
            "envelope_scale": float(
                np.pi / (abs(element) * pulse_duration_us)
            ),
        }

    events = []
    for start, end, sites, axes in zip(
        control["pulse_start_times_us"],
        control["event_times_us"],
        control["addressed_sites"],
        control["pulse_axes"],
    ):
        tones = []
        for site, axis in zip(sites, axes):
            transition = transitions[site]
            tones.append({
                "intended_site": site,
                "logical_axis": axis,
                "carrier_rad_per_us": transition["carrier_rad_per_us"],
                "phase_rad": (
                    transition["phase_rad"]
                    - (np.pi / 2 if axis == "Y" else 0.0)
                ),
                "envelope_scale": transition["envelope_scale"],
            })
        events.append({
            "start_us": float(start),
            "end_us": float(end),
            "tones": tones,
        })

    return {
        "duration_us": compiled["duration_us"],
        "pulse_duration_us": float(pulse_duration_us),
        "phase_cycle": phase_cycle,
        "events": events,
        "envelope_convention": "Unit mean over each pulse window.",
        "protocol_name": protocol_name,
    }


def propagate_shared_events(
    model,
    schedule,
    *,
    shape="sine",
    steps=1024,
    drift_at_time=None,
    free_step_us=0.05,
):
    """Propagate shared tones with nominal or time-dependent drift throughout."""
    if shape not in ("square", "sine"):
        raise ValueError("shape must be 'square' or 'sine'.")
    if not np.isfinite(free_step_us) or free_step_us <= 0:
        raise ValueError("free_step_us must be finite and positive.")
    if drift_at_time is not None and not callable(drift_at_time):
        raise ValueError("drift_at_time must be callable.")

    drift = model["drift_hamiltonian"]
    propagator = qeye(drift.dims[0])
    previous = 0.0

    def free_interval(start, end):
        """Propagate free evolution without interrupting the noise trajectory."""
        if drift_at_time is None:
            return hermitian_propagator(drift, end - start)
        count = max(1, int(np.ceil((end - start) / free_step_us)))
        return hermitian_time_dependent_propagator(
            drift_at_time, start, end, count
        )

    for event in schedule["events"]:
        start, end = event["start_us"], event["end_us"]
        if start < previous or end <= start or end > schedule["duration_us"]:
            raise ValueError("Invalid or overlapping physical pulse windows.")

        if start > previous:
            propagator = free_interval(previous, start) * propagator

        def hamiltonian(time):
            """Evaluate shared tones and the current drift during a pulse."""
            envelope = (
                1.0 if shape == "square"
                else sine_envelope(time - start, end - start, np.pi / 2)
            )
            tones = [
                {
                    **tone,
                    "envelope": tone["envelope_scale"] * envelope,
                }
                for tone in event["tones"]
            ]
            result = shared_multitone_hamiltonian(time, model, tones)
            if drift_at_time is not None:
                result += drift_at_time(time) - drift
            return result

        pulse = hermitian_time_dependent_propagator(
            hamiltonian, start, end, steps
        )
        propagator = pulse * propagator
        previous = end

    if previous < schedule["duration_us"]:
        propagator = (
            free_interval(previous, schedule["duration_us"]) * propagator
        )
    return propagator
