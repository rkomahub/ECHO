import json
from pathlib import Path

import numpy as np
import pytest

from echo_spin.scenario import (
    load_longitudinal_scenario,
    longitudinal_scenario_from_dict,
)

CONFIGURATION = (
    Path(__file__).resolve().parents[1]
    / "configs/registers/three_nv_trial.json"
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
