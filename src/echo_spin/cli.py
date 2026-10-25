import argparse
import json
from pathlib import Path

from echo_spin.nv.registers import load_register_configuration
from echo_spin.provenance import software_provenance
from echo_spin.scenario import load_longitudinal_scenario


def main(argv=None) -> int:
    """Report a resolved register or an ideal longitudinal scenario as JSON."""
    parser = argparse.ArgumentParser(
        description="Inspect an NV register or longitudinal scenario."
    )
    parser.add_argument(
        "configuration",
        type=Path,
        help="Register or scenario JSON file.",
    )
    parser.add_argument(
        "--scenario",
        action="store_true",
        help="Load a complete ideal longitudinal scenario.",
    )
    arguments = parser.parse_args(argv)

    try:
        scenario = None
        if arguments.scenario:
            scenario = load_longitudinal_scenario(arguments.configuration)
            register = scenario.register
        else:
            register = load_register_configuration(arguments.configuration)

        report = {
            "number_of_nv": register.number_of_nv,
            "units": {
                "positions": "m",
                "magnetic_fields": "T",
                "local_frames": "dimensionless",
            },
            "positions": register.positions.tolist(),
            "local_frames": register.local_frames.tolist(),
            "magnetic_fields": register.magnetic_fields.tolist(),
        }

        if scenario is not None:
            report["units"].update({
                "gradient": "T/m",
                "D": "rad/us",
                "E": "rad/us",
                "gamma_e": "rad/us/T",
                "bare_detuning": "rad/us",
            })
            report["specification"] = scenario.specification
            report["software"] = software_provenance()
            report["scope"] = {
                "preparation": (
                    "Optional for unitary gate evaluation; specified logical product "
                    "density matrices are validated but not propagated by this command."
                ),
                "nuclear_parameters": "Outside the selected model.",
                "uncertainty": (
                    "Declared distributions are validated; this command does not "
                    "sample parameters or execute an uncertainty ensemble."
                ),
                "hardware_bounds": (
                    "Not specified for instantaneous ideal control."
                ),
                "experimental_feasibility": "Not assessed.",
                "individual_addressability": "Assumed, not established.",
            }
            report["validation"] = {
                "geometry_and_frames": "Passed.",
                "affine_field": "Passed.",
                "site_parameters": "Passed.",
                "task_and_control_conventions": "Passed.",
                "parameter_provenance": "Required entries present; sources not verified.",
                "crystallographic_feasibility": "Not assessed.",
                "hardware_feasibility": "Not assessed.",
                "physical_model_validity": "Not established by input validation.",
            }
            report["environment"] = scenario.environment
            report["environment_validation"] = (
                "Record structure and finite JSON values checked; "
                "section-specific physics and execution support not established."
            )
            report["constraint_checks"] = scenario.constraint_checks

        output = json.dumps(report, indent=2, allow_nan=False)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
