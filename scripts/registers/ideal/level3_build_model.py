import argparse
import json
from pathlib import Path

from echo_spin.provenance import software_provenance
from echo_spin.representation import longitudinal_model_report


def main():
    """Save a longitudinal model report from Level 2 characterisation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("characterisation_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--assess-dipolar",
        action="store_true",
        help="Include the static reference diagnostic, limited to N<=3.",
    )
    arguments = parser.parse_args()

    try:
        with arguments.characterisation_report.open(encoding="utf-8") as stream:
            source = json.load(stream)

        report = longitudinal_model_report(
            source,
            assess_dipolar=arguments.assess_dipolar,
        )
        report["source_level_2_software"] = source.get("software")
        report["software"] = software_provenance()

        text = json.dumps(report, indent=2, allow_nan=False)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
