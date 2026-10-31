import json
from pathlib import Path

import numpy as np

from echo_spin.electronic_reference import electronic_population_diagnostics
from echo_spin.scenario import longitudinal_scenario_from_dict


def test_electronic_populations_conserve_probability_and_resolve_exchange():
    """Resolve logical exchange while accounting separately for leakage."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/scenario.json"
    )
    scenario = longitudinal_scenario_from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
    result = electronic_population_diagnostics(scenario, [0.0, 1.0])
    initial, evolved = result["records"]

    np.testing.assert_allclose(
        initial["mean_stay_probability"], 1.0, atol=1e-12
    )
    for record in result["records"]:
        total = sum(
            record[key] for key in (
                "mean_stay_probability",
                "mean_exchange_probability",
                "mean_other_logical_transfer_probability",
                "mean_leakage_probability",
            )
        )
        np.testing.assert_allclose(total, 1.0, atol=1e-8, rtol=0.0)

    assert evolved["mean_exchange_probability"] > 1e-6
    assert (
        evolved["mean_leakage_probability"]
        < evolved["mean_exchange_probability"]
    )
