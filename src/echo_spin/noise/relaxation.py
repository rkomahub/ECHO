import numpy as np
from qutip import Qobj, basis, sigmax

from echo_spin.core.operators import spin_operators
from echo_spin.noise.dephasing import dephasing_channel


def amplitude_damping_channel(
    density_matrix: Qobj,
    time: float,
    relaxation_time: float,
) -> Qobj:
    """Apply qubit relaxation from logical excited |1> to ground |0>."""
    if density_matrix.dims != [[2], [2]]:
        raise ValueError("Amplitude damping requires a single-qubit operator.")

    if not np.isfinite(time) or time < 0:
        raise ValueError("time must be finite and non-negative.")

    if not np.isfinite(relaxation_time) or relaxation_time <= 0:
        raise ValueError("relaxation_time must be finite and positive.")

    survival = np.exp(-time / relaxation_time)
    decay_probability = -np.expm1(-time / relaxation_time)

    ground = basis(2, 0)
    excited = basis(2, 1)

    k_zero = (
        ground.proj()
        + np.sqrt(survival) * excited.proj()
    )
    k_one = (
        np.sqrt(decay_probability)
        * ground * excited.dag()
    )

    return (
        k_zero * density_matrix * k_zero.dag()
        + k_one * density_matrix * k_one.dag()
    )


def relaxation_dephasing_channel(
    density_matrix: Qobj,
    time: float,
    relaxation_time: float,
    pure_dephasing_time: float,
) -> Qobj:
    """Combine qubit relaxation with an independent pure-dephasing channel."""
    relaxed = amplitude_damping_channel(
        density_matrix=density_matrix,
        time=time,
        relaxation_time=relaxation_time,
    )

    return dephasing_channel(
        density_matrix=relaxed,
        time=time,
        coherence_time=pure_dephasing_time,
    )


def generalized_amplitude_damping_channel(
    density_matrix: Qobj,
    time: float,
    relaxation_time: float,
    equilibrium_excited_population: float,
) -> Qobj:
    """Apply upward and downward relaxation toward a specified qubit population."""
    population = equilibrium_excited_population

    if not np.isfinite(population) or not 0 <= population <= 1:
        raise ValueError(
            "equilibrium_excited_population must be finite and between 0 and 1."
        )

    downward = amplitude_damping_channel(
        density_matrix=density_matrix,
        time=time,
        relaxation_time=relaxation_time,
    )

    flip = sigmax()

    upward = flip * amplitude_damping_channel(
        density_matrix=flip * density_matrix * flip,
        time=time,
        relaxation_time=relaxation_time,
    ) * flip

    return (
        (1 - population) * downward
        + population * upward
    )


def spin_one_relaxation_operators(decay_rates) -> list:
    """Construct independent spin-1 jumps from mS=+1 and -1 into mS=0."""
    if np.iscomplexobj(decay_rates):
        raise ValueError("decay_rates must be real.")

    decay_rates = np.asarray(decay_rates, dtype=float)

    if (
        decay_rates.shape != (2,)
        or not np.all(np.isfinite(decay_rates))
        or np.any(decay_rates < 0)
    ):
        raise ValueError(
            "Provide two finite non-negative rates, ordered (+1, -1)."
        )

    ground = basis(3, 1)

    return [
        np.sqrt(rate) * ground * basis(3, index).dag()
        for rate, index in zip(decay_rates, [0, 2])
    ]


def spin_one_dephasing_operator(
    single_quantum_coherence_time: float,
) -> Qobj:
    """Construct longitudinal spin-1 dephasing calibrated to single-quantum coherence."""
    coherence_time = single_quantum_coherence_time

    if not np.isfinite(coherence_time) or coherence_time <= 0:
        raise ValueError("coherence time must be finite and positive.")

    _, _, sz = spin_operators(1)

    return np.sqrt(2 / coherence_time) * sz


def spin_one_energy_relaxation_operators(
    hamiltonian: Qobj,
    decay_rates,
) -> list:
    """Construct excited-to-ground jumps with rates ordered by excited-state energy."""
    if (
        hamiltonian.dims != [[3], [3]]
        or not hamiltonian.isherm
        or not np.all(np.isfinite(hamiltonian.full()))
    ):
        raise ValueError(
            "hamiltonian must be a finite Hermitian single-spin-1 operator."
        )

    energies, states = hamiltonian.eigenstates()

    if np.isclose(
        energies[1],
        energies[0],
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError("The ground state must be nondegenerate.")

    # Reuse rate validation and the two-jump construction.
    bare_jumps = spin_one_relaxation_operators(decay_rates)

    # Map bare indices (0, 1, 2) to (lower excited, ground, upper excited).
    transformation = (
        states[1] * basis(3, 0).dag()
        + states[0] * basis(3, 1).dag()
        + states[2] * basis(3, 2).dag()
    )

    return [
        transformation * jump * transformation.dag()
        for jump in bare_jumps
    ]
