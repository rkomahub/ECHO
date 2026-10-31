import numpy as np

from echo_spin.dynamics.propagators import static_propagator
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
        })

    return {
        "physical_dimension": full.shape[0],
        "logical_dimension": dimension,
        "frame": "Exact interaction picture of the local electronic Hamiltonian.",
        "average": "Uniform over logical computational-basis inputs.",
        "records": records,
        "limits": [
            "Bare logical subspace must be invariant under local evolution.",
            "No controls, nuclei or noise.",
            "Population diagnostics do not quantify all phase or gate errors.",
        ],
    }
