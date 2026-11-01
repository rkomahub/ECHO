from itertools import product

import numpy as np
from qutip import Qobj, basis, qeye, tensor
from scipy.sparse.linalg import expm_multiply

from echo_spin.control.pulses import sine_envelope
from echo_spin.core.operators import embed_operator, spin_operators
from echo_spin.core.system import SpinSystem
from echo_spin.dynamics.propagators import hermitian_propagator
from echo_spin.electronic_reference import (
    electronic_magnetic_drive_operator,
    shared_multitone_hamiltonian,
)
from echo_spin.nv.fields import local_zeeman_vector
from echo_spin.nv.registers import nv_register_hamiltonian
from echo_spin.representation import build_electronic_reference
from echo_spin.scenario import validate_nuclear_parameters
from echo_spin.spectroscopy.transitions import allowed_transitions


def build_nitrogen_reference(scenario, nuclear_parameters) -> dict:
    """Assemble a laboratory-frame electron--14N register in interleaved order."""
    number_of_nv = scenario.register.number_of_nv
    validate_nuclear_parameters(nuclear_parameters, number_of_nv)

    if any(
        record["isotope"] != "14N"
        for record in nuclear_parameters["sites"]
    ):
        raise ValueError("This reference currently supports 14N only.")

    electronic = build_electronic_reference(scenario)
    system = SpinSystem([1, 1] * number_of_nv)
    parameters = []

    for site, (electron, nucleus) in enumerate(zip(
        scenario.specification["sites"],
        nuclear_parameters["sites"],
    )):
        field = scenario.register.magnetic_fields[site]
        frame = scenario.register.local_frames[site]
        parameters.append({
            "D": electron["D"],
            "E": electron["E"],
            "omega_e": local_zeeman_vector(
                field, frame, electron["gamma_e"]
            ),
            "Q": nucleus["Q"],
            "omega_n": local_zeeman_vector(
                field, frame, nucleus["gamma_n"]
            ),
            "A": nucleus["A"],
        })

    local = nv_register_hamiltonian(
        system=system,
        nv_parameters=parameters,
    )

    # Insert nuclear identities, then restore (e0,n0,e1,n1,...) order.
    permutation = [
        index
        for site in range(number_of_nv)
        for index in (site, number_of_nv + site)
    ]
    interaction = tensor(
        electronic["dipolar_hamiltonian"],
        qeye([3] * number_of_nv),
    ).permute(permutation)

    return {
        "system": system,
        "hamiltonian": local + interaction,
        "local_hamiltonian": local,
        "dipolar_hamiltonian": interaction,
        "nuclear_parameters": nuclear_parameters,
        "frame": "laboratory",
        "subsystem_order": [
            label
            for site in range(number_of_nv)
            for label in (f"e{site}", f"n{site}")
        ],
    }


def nitrogen_product_preparation(projections):
    """Prepare spin-1 nuclei in specified mI states or maximally mixed states."""
    projections = list(projections)
    if not 1 <= len(projections) <= 3:
        raise ValueError("Provide one to three nuclear preparation entries.")

    states = []
    for projection in projections:
        if projection is None:
            states.append(qeye(3) / 3)
        elif (
            not isinstance(projection, (bool, np.bool_))
            and isinstance(projection, (int, np.integer))
            and projection in (-1, 0, 1)
        ):
            states.append(basis(3, 1 - int(projection)).proj())
        else:
            raise ValueError("Use mI=-1, 0, +1, or None for an unpolarised nucleus.")

    return tensor(states)


def electron_nitrogen_preparation(electron_density, nuclear_density):
    """Combine electronic and nuclear densities in interleaved subsystem order."""
    number_of_nv = len(electron_density.dims[0])
    if not 1 <= number_of_nv <= 3:
        raise ValueError("This reference supports one to three NV sites.")

    dimensions = [[3] * number_of_nv, [3] * number_of_nv]
    for density in (electron_density, nuclear_density):
        if (
            density.dims != dimensions
            or not density.isherm
            or not np.all(np.isfinite(density.full()))
            or abs(complex(density.tr()) - 1) > 1e-10
            or np.min(density.eigenenergies()) < -1e-10
        ):
            raise ValueError("Provide normalised positive spin-1 density matrices.")

    permutation = [
        index
        for site in range(number_of_nv)
        for index in (site, number_of_nv + site)
    ]
    return tensor(electron_density, nuclear_density).permute(permutation)


def nitrogen_magnetic_drive_operator(
    scenario, nuclear_parameters, field_profile_tesla
):
    """Construct the laboratory magnetic coupling of electrons and intrinsic 14N."""
    number_of_nv = scenario.register.number_of_nv
    validate_nuclear_parameters(nuclear_parameters, number_of_nv)
    if any(
        record["isotope"] != "14N"
        for record in nuclear_parameters["sites"]
    ):
        raise ValueError("This reference currently supports 14N only.")

    electronic = electronic_magnetic_drive_operator(
        scenario, field_profile_tesla
    )
    permutation = [
        index
        for site in range(number_of_nv)
        for index in (site, number_of_nv + site)
    ]
    operator = tensor(
        electronic, qeye([3] * number_of_nv)
    ).permute(permutation)

    system = SpinSystem([1, 1] * number_of_nv)
    fields = np.asarray(field_profile_tesla, dtype=float)
    for site, nucleus in enumerate(nuclear_parameters["sites"]):
        coefficients = local_zeeman_vector(
            fields[site],
            scenario.register.local_frames[site],
            nucleus["gamma_n"],
        )
        for coefficient, spin in zip(coefficients, spin_operators(1)):
            operator += coefficient * embed_operator(
                spin, 2 * site + 1, system
            )

    return operator


def nitrogen_local_spectra(reference, magnetic_operator) -> list:
    """Resolve isolated electron--nitrogen transitions under the physical drive."""
    number_of_nv = reference["system"].number_of_spins // 2
    local_register = reference["local_hamiltonian"]
    if magnetic_operator.dims != local_register.dims:
        raise ValueError("Magnetic operator and reference dimensions must match.")

    normalisation = 9**(number_of_nv - 1)
    results = []

    for site in range(number_of_nv):
        pair = [2 * site, 2 * site + 1]
        local = local_register.ptrace(pair) / normalisation
        drive = magnetic_operator.ptrace(pair) / normalisation

        # Other sites contribute only a common identity shift.
        energies = np.asarray(local.eigenenergies(), dtype=float)
        energies -= energies[0]
        transitions = allowed_transitions(local, drive)

        results.append({
            "site": site,
            "energies_relative_to_ground_rad_per_us": energies.tolist(),
            "transitions": [
                {
                    "frequency_rad_per_us": float(frequency),
                    "strength": float(strength),
                    "lower_eigenstate": int(lower),
                    "upper_eigenstate": int(upper),
                }
                for frequency, strength, lower, upper in transitions
            ],
            "strength_units": "(rad/us)^2",
        })

    return results


def calibrate_local_nitrogen_square_pulse(
    reference, magnetic_operator, site=0
) -> dict:
    """Calibrate a local square pulse on the predominantly mI=+1 logical branch."""
    number_of_nv = reference["system"].number_of_spins // 2
    if (
        isinstance(site, bool)
        or not isinstance(site, (int, np.integer))
        or not 0 <= site < number_of_nv
    ):
        raise ValueError("Provide a valid NV site index.")

    pair = [2 * site, 2 * site + 1]
    normalisation = 9**(number_of_nv - 1)
    local = reference["local_hamiltonian"].ptrace(pair) / normalisation
    magnetic = magnetic_operator.ptrace(pair) / normalisation
    energies, states = local.eigenstates()
    transform = Qobj(
        np.column_stack([state.full().ravel() for state in states]),
        dims=local.dims,
    )
    matrix = (transform.dag() * magnetic * transform).full()

    initial = tensor(basis(3, 1), basis(3, 0))
    target = tensor(basis(3, 2), basis(3, 0))
    initial_weights = np.array([
        abs(state.overlap(initial))**2 for state in states
    ])
    target_weights = np.array([
        abs(state.overlap(target))**2 for state in states
    ])
    lower = int(np.argmax(initial_weights))
    upper = int(np.argmax(target_weights))
    carrier = float(energies[upper] - energies[lower])
    coupling = abs(matrix[upper, lower])
    if carrier <= 0 or coupling == 0:
        raise ValueError("The selected branch must have positive frequency and coupling.")
    duration = float(np.pi / coupling)

    excited_projector = tensor(
        Qobj(np.diag([1, 0, 1])), qeye(3)
    )
    weights = np.array([
        float(np.real(state.dag() * excited_projector * state))
        for state in states
    ])
    if np.any((weights > 0.1) & (weights < 0.9)):
        raise ValueError("Electronic manifolds are too mixed for this local RWA.")
    excitation = (weights > 0.5).astype(int)
    raising = Qobj(
        matrix * ((excitation[:, None] - excitation[None, :]) == 1),
        dims=local.dims,
    )
    rotating = Qobj(
        np.diag(
            np.asarray(energies) - energies[0] - carrier * excitation
        ),
        dims=local.dims,
    )
    rotating += (raising + raising.dag()) / 2
    frame = Qobj(
        np.diag(np.exp(-1j * carrier * duration * excitation)),
        dims=local.dims,
    )
    propagator = (
        transform * frame * hermitian_propagator(rotating, duration)
        * transform.dag()
    )

    return {
        "propagator": propagator,
        "carrier_rad_per_us": carrier,
        "duration_us": duration,
        "initial_state_overlap": float(initial_weights[lower]),
        "target_state_overlap": float(target_weights[upper]),
        "model": "Local dressed electron-manifold microwave RWA.",
        "logical_x_carrier_phase_rad": float(np.angle(
            target.overlap(states[upper])
            * matrix[upper, lower]
            * states[lower].overlap(initial)
        )),
    }


def build_nitrogen_drive_rwa(reference, magnetic_operator, carrier):
    """Build a sparse joint electron--14N microwave RWA without nuclear reduction."""
    if not np.isfinite(carrier) or carrier <= 0:
        raise ValueError("carrier must be finite and positive.")

    local_register = reference["local_hamiltonian"]
    if magnetic_operator.dims != local_register.dims:
        raise ValueError("Magnetic operator and reference dimensions must match.")

    number_of_nv = reference["system"].number_of_spins // 2
    normalisation = 9**(number_of_nv - 1)
    electronic_excited = tensor(
        Qobj(np.diag([1, 0, 1])), qeye(3)
    )
    bases = []
    classifications = []

    for site in range(number_of_nv):
        pair = [2 * site, 2 * site + 1]
        local = local_register.ptrace(pair) / normalisation
        _, states = local.eigenstates()
        weights = np.array([
            float(np.real(state.dag() * electronic_excited * state))
            for state in states
        ])
        if np.any((weights > 0.1) & (weights < 0.9)):
            raise ValueError("Electronic manifolds are too mixed for this RWA.")

        classification = (weights > 0.5).astype(int)
        if classification.sum() != 6:
            raise ValueError("Expected three ground-like and six excited-like states.")

        bases.append(Qobj(
            np.column_stack([state.full().ravel() for state in states]),
            dims=local.dims,
        ))
        classifications.append(classification)

    transform = tensor(bases).to("CSR")
    local_diagonal = (
        transform.dag() * local_register * transform
    ).diag().real
    interaction = (
        transform.dag() * reference["dipolar_hamiltonian"] * transform
    )
    magnetic = transform.dag() * magnetic_operator * transform

    excitation = np.array([
        sum(classifications[site][level] for site, level in enumerate(levels))
        for levels in product(range(9), repeat=number_of_nv)
    ])
    difference = excitation[:, None] - excitation[None, :]
    dimensions = local_register.dims

    interaction_rwa = Qobj(
        interaction.full() * (difference == 0), dims=dimensions
    ).to("CSR")
    raising = Qobj(
        magnetic.full() * (difference == 1), dims=dimensions
    ).to("CSR")
    local_rotating = Qobj(
        np.diag(local_diagonal - carrier * excitation),
        dims=dimensions,
    ).to("CSR")

    return {
        "drift_hamiltonian": local_rotating + interaction_rwa,
        "local_rotating_hamiltonian": local_rotating,
        "dipolar_hamiltonian": interaction_rwa,
        "raising_operator": raising,
        "drive_operator": (raising + raising.dag()) / 2,
        "excitation_operator": Qobj(
            np.diag(excitation), dims=dimensions
        ).to("CSR"),
        "basis_to_lab": transform,
        "carrier_rad_per_us": float(carrier),
        "nuclear_reduction": False,
        "approximation": "Local dressed-manifold microwave RWA.",
    }


def joint_exponential_action(hamiltonian, columns, time):
    """Evolve joint amplitude columns without constructing a dense propagator."""
    columns = np.asarray(columns, dtype=complex)
    if (
        not hamiltonian.isherm
        or not np.isfinite(time)
        or columns.ndim != 2
        or columns.shape[0] != hamiltonian.shape[0]
        or not np.all(np.isfinite(columns))
    ):
        raise ValueError("Provide a Hermitian Hamiltonian and matching finite columns.")

    sparse = hamiltonian.to("CSR").data.as_scipy()
    if not np.all(np.isfinite(sparse.data)):
        raise ValueError("Hamiltonian entries must be finite.")

    if time == 0:
        return columns.copy()

    return expm_multiply(
        (-1j * time) * sparse,
        columns,
        traceA=-1j * time * complex(hamiltonian.tr()),
    )


def nitrogen_logical_input_columns(nuclear_density) -> dict:
    """Embed arbitrary logical electronic inputs with a specified nuclear density."""
    number_of_nv = len(nuclear_density.dims[0])
    dimensions = [[3] * number_of_nv, [3] * number_of_nv]
    if (
        not 1 <= number_of_nv <= 3
        or nuclear_density.dims != dimensions
        or not nuclear_density.isherm
        or not np.all(np.isfinite(nuclear_density.full()))
        or abs(complex(nuclear_density.tr()) - 1) > 1e-10
    ):
        raise ValueError("Provide a normalised spin-1 nuclear density matrix.")

    values, states = nuclear_density.eigenstates()
    values = np.asarray(values, dtype=float)
    if values.min() < -1e-10:
        raise ValueError("Nuclear density must be positive.")
    values = np.maximum(values, 0)
    selected = np.flatnonzero(values > 0)
    probabilities = values[selected] / values.sum()

    logical_states = list(product((0, 1), repeat=number_of_nv))
    nuclear_states = list(product(range(3), repeat=number_of_nv))
    dimension = 2**number_of_nv
    rows = np.zeros((dimension, 3**number_of_nv), dtype=int)

    for logical_index, bits in enumerate(logical_states):
        for nuclear_index, projections in enumerate(nuclear_states):
            row = 0
            for bit, projection in zip(bits, projections):
                row = 9 * row + 3 * (1 + bit) + projection
            rows[logical_index, nuclear_index] = row

    columns = np.zeros(
        (9**number_of_nv, dimension * len(selected)), dtype=complex
    )
    for sector, eigenstate in enumerate(selected):
        vector = states[eigenstate].full().ravel()
        for logical_index in range(dimension):
            columns[
                rows[logical_index], sector * dimension + logical_index
            ] = vector

    return {
        "columns": columns,
        "probabilities": probabilities,
        "logical_rows": rows,
        "number_of_nv": number_of_nv,
    }


def nitrogen_electronic_gate_metrics(evolved_columns, inputs, target) -> dict:
    """Evaluate the final electronic channel without nuclear reset or postselection."""
    evolved = np.asarray(evolved_columns, dtype=complex)
    if (
        evolved.shape != inputs["columns"].shape
        or not np.all(np.isfinite(evolved))
    ):
        raise ValueError("Provide matching finite propagated joint columns.")

    number_of_nv = inputs["number_of_nv"]
    dimension = 2**number_of_nv
    if (
        target.dims != [[2] * number_of_nv, [2] * number_of_nv]
        or not np.all(np.isfinite(target.full()))
        or (
            target.dag() * target - qeye([2] * number_of_nv)
        ).norm("fro") > 1e-10
    ):
        raise ValueError("Provide a matching unitary logical target.")

    probabilities = inputs["probabilities"]
    rank = len(probabilities)
    rows = inputs["logical_rows"]
    blocks = evolved[rows, :].reshape(
        dimension, 3**number_of_nv, rank, dimension
    )
    kraus = blocks.transpose(2, 1, 0, 3)
    kraus = kraus * np.sqrt(probabilities)[:, None, None, None]

    norm_sum = float(np.sum(np.abs(kraus)**2))
    overlaps = np.einsum(
        "ij,abij->ab", target.full().conj(), kraus, optimize=False
    )
    fidelity = (
        norm_sum + np.sum(np.abs(overlaps)**2)
    ) / (dimension * (dimension + 1))

    outside = np.ones(evolved.shape[0], dtype=bool)
    outside[rows.ravel()] = False
    discarded = evolved[outside, :].reshape(-1, rank, dimension)
    leakage = np.sum(
        np.abs(discarded)**2 * probabilities[None, :, None]
    ) / dimension
    total = np.sum(
        np.abs(evolved.reshape(-1, rank, dimension))**2
        * probabilities[None, :, None]
    ) / dimension

    return {
        "average_gate_fidelity": float(fidelity),
        "average_leakage": float(leakage),
        "total_probability": float(total),
        "nuclear_rank": rank,
        "postselection": False,
    }


def propagate_nitrogen_events(
    model,
    schedule,
    columns,
    *,
    shape="sine",
    steps=256,
    free_eigensystem=None,
    drift_at_time=None,
    free_step_us=0.05,
):
    """Propagate joint amplitude columns through shared tones without nuclear resets."""
    if shape not in ("square", "sine"):
        raise ValueError("shape must be 'square' or 'sine'.")
    if (
        isinstance(steps, (bool, np.bool_))
        or not isinstance(steps, (int, np.integer))
        or steps < 1
    ):
        raise ValueError("steps must be a positive integer.")

    drift = model["drift_hamiltonian"]

    if not np.isfinite(free_step_us) or free_step_us <= 0:
        raise ValueError("free_step_us must be finite and positive.")

    def current_drift(time):
        """Return a matching finite Hermitian drift at the requested time."""
        candidate = drift if drift_at_time is None else drift_at_time(time)
        if (
            not isinstance(candidate, Qobj)
            or candidate.dims != drift.dims
            or not candidate.isherm
            or not np.all(np.isfinite(candidate.full()))
        ):
            raise ValueError(
                "Time-dependent drift must be finite, Hermitian, "
                "and match the model dimensions."
            )
        return candidate

    state = np.asarray(columns, dtype=complex).copy()
    if (
        state.ndim != 2
        or state.shape[0] != drift.shape[0]
        or not np.all(np.isfinite(state))
    ):
        raise ValueError("Provide matching finite joint amplitude columns.")

    if drift_at_time is None and free_eigensystem is None:
        energies, eigenstates = drift.eigenstates()
        vectors = np.column_stack([
            eigenstate.full().ravel() for eigenstate in eigenstates
        ])
    elif drift_at_time is None:
        energies, vectors = free_eigensystem

    def free_action(amplitudes, duration, start_time):
        """Apply fixed drift evolution using the cached eigensystem."""
        if duration == 0:
            return amplitudes

        if drift_at_time is not None:
            count = max(1, int(np.ceil(duration / free_step_us)))
            dt = duration / count
            for index in range(count):
                time = start_time + (index + 0.5) * dt
                amplitudes = joint_exponential_action(
                    current_drift(time), amplitudes, dt
                )
            return amplitudes

        coefficients = np.einsum(
            "ki,kj->ij", vectors.conj(), amplitudes, optimize=False
        )
        coefficients *= np.exp(-1j * energies * duration)[:, None]
        return np.einsum(
            "ki,ij->kj", vectors, coefficients, optimize=False
        )

    previous = 0.0
    for event in schedule["events"]:
        start, end = event["start_us"], event["end_us"]
        if start < previous or end <= start or end > schedule["duration_us"]:
            raise ValueError("Invalid or overlapping pulse windows.")

        state = free_action(state, start - previous, previous)
        dt = (end - start) / steps

        for step in range(steps):
            time = start + (step + 0.5) * dt
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
            hamiltonian = shared_multitone_hamiltonian(time, model, tones)
            if drift_at_time is not None:
                hamiltonian += current_drift(time) - drift
            state = joint_exponential_action(hamiltonian, state, dt)

        previous = end

    return free_action(
        state, schedule["duration_us"] - previous, previous
    )


def slow_operator_in_nitrogen_rwa(model, laboratory_operator):
    """Transform a slow laboratory perturbation into the joint microwave RWA."""
    transform = model["basis_to_lab"]
    if (
        not isinstance(laboratory_operator, Qobj)
        or laboratory_operator.dims != transform.dims
        or not laboratory_operator.isherm
        or not np.all(np.isfinite(laboratory_operator.full()))
    ):
        raise ValueError(
            "Provide a finite Hermitian laboratory operator "
            "matching the joint model."
        )

    dressed = transform.dag() * laboratory_operator * transform
    excitation = model["excitation_operator"].diag().real
    retained = np.isclose(
        excitation[:, None],
        excitation[None, :],
        atol=1e-12,
        rtol=0.0,
    )
    return Qobj(
        dressed.full() * retained,
        dims=dressed.dims,
    ).to("CSR")
