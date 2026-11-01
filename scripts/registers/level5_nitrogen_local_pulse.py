import json
from pathlib import Path

from qutip import basis, tensor

from echo_spin.nitrogen_reference import (
    build_nitrogen_reference,
    calibrate_local_nitrogen_square_pulse,
    nitrogen_magnetic_drive_operator,
    nitrogen_product_preparation,
)
from echo_spin.provenance import software_provenance
from echo_spin.scenario import longitudinal_scenario_from_dict


def main():
    """Compare nuclear preparations under one polarised-branch calibrated pulse."""
    root = Path("results/linear/n3")
    source = json.loads(
        (root / "level_05_controls/local_addressability.json").read_text()
    )
    nuclei = json.loads(
        Path("configs/registers/linear/n3/nitrogen_14.json").read_text()
    )
    specification = source["cases"]["gradient"]["specification"]
    scenario = longitudinal_scenario_from_dict(specification)
    reference = build_nitrogen_reference(scenario, nuclei)
    drive = nitrogen_magnetic_drive_operator(
        scenario, nuclei, source["field_profile_tesla"]
    )
    calibrated = calibrate_local_nitrogen_square_pulse(reference, drive)
    propagator = calibrated["propagator"]
    cases = {}

    for label, projection in (("polarised", 1), ("unpolarised", None)):
        initial = tensor(
            basis(3, 1).proj(),
            nitrogen_product_preparation([projection]),
        )
        evolved = propagator * initial * propagator.dag()
        populations = evolved.ptrace(0).diag().real.tolist()
        cases[label] = {
            "electron_populations_ms_plus1_0_minus1": populations,
            "joint_density_after_pulse": {
                "real": evolved.full().real.tolist(),
                "imag": evolved.full().imag.tolist(),
            },
        }
        print(label, populations)

    report = {
        "software": software_provenance(),
        "specification": specification,
        "nuclear_parameters": nuclei,
        "field_profile_tesla": source["field_profile_tesla"],
        "site": 0,
        "calibration": {
            key: value for key, value in calibrated.items()
            if key != "propagator"
        },
        "cases": cases,
        "limit": "Single local pulse; no intersite interaction or gate-fidelity claim.",
    }
    print("Calibration:", report["calibration"])
    output = root / "level_05_controls/nitrogen_local_pulse.json"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
