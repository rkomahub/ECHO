import json
from pathlib import Path

import numpy as np

from echo_spin.characterisation import longitudinal_characterisation_report
from echo_spin.decoupling.modulation import effective_zz_coupling
from echo_spin.protocols import build_ideal_gate_candidates, ideal_protocol_report
from echo_spin.representation import longitudinal_model_report
from echo_spin.scenario import longitudinal_scenario_from_dict


def test_candidates_have_expected_local_and_pair_sign_averages():
    """Distinguish free, global echo and selective interaction retention."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/registers/linear/n3/scenario.json"
    )
    specification = json.loads(path.read_text(encoding="utf-8"))
    characterisation = longitudinal_characterisation_report(
        longitudinal_scenario_from_dict(specification)
    )
    report = longitudinal_model_report(characterisation)
    candidates = build_ideal_gate_candidates(report)
    duration = candidates["duration_us"]

    assert candidates["target"].dims == [[2, 2, 2], [2, 2, 2]]
    np.testing.assert_allclose(
        duration * abs(report["couplings_rad_per_us"][0][1]),
        np.pi / 4,
    )

    for name, schedules in candidates["schedules"].items():
        for pulses in schedules:
            assert len(pulses) % 2 == 0
            local_average = effective_zz_coupling(
                1.0, duration, pulses, []
            )
            np.testing.assert_allclose(
                local_average,
                1.0 if name == "free" else 0.0,
                atol=1e-12,
            )

        for i, j in ((0, 1), (0, 2), (1, 2)):
            pair_average = effective_zz_coupling(
                1.0, duration, schedules[i], schedules[j]
            )
            expected = (
                1.0
                if name != "hadamard_selective" or (i, j) == (0, 1)
                else 0.0
            )
            np.testing.assert_allclose(
                pair_average, expected, atol=1e-12
            )

    saved = json.loads(json.dumps(
        ideal_protocol_report(report),
        allow_nan=False,
    ))
    protocols = saved["protocols"]

    assert protocols["free"]["individual_pulse_count"] == 0
    assert protocols["free"]["distinct_pulse_time_count"] == 0
    assert protocols["closed_global_hahn"]["individual_pulse_count"] == 6
    assert protocols["closed_global_hahn"]["distinct_pulse_time_count"] == 2
    assert all(
        item["closed_up_to_global_phase"]
        for item in protocols.values()
    )

    reconstructed_target = (
        np.asarray(saved["target_diagonal"]["real"])
        + 1j * np.asarray(saved["target_diagonal"]["imag"])
    )
    np.testing.assert_allclose(
        reconstructed_target,
        candidates["target"].diag(),
    )
