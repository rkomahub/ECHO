import argparse
import json
from pathlib import Path

from echo_spin.characterisation import longitudinal_characterisation_report
from echo_spin.provenance import software_provenance
from echo_spin.scenario import longitudinal_scenario_from_dict


def main():
    """Save Level 2 characterisation from a previously exported Level 1 report."""
    parser = argparse.ArgumentParser(
        description="Characterise an ideal longitudinal NV scenario."
    )
    parser.add_argument("scenario_report", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    try:
        with arguments.scenario_report.open(encoding="utf-8") as stream:
            source = json.load(stream)

        if not isinstance(source, dict) or "specification" not in source:
            raise ValueError("Input must be a complete Level 1 scenario report.")

        scenario = longitudinal_scenario_from_dict(source["specification"])
        report = longitudinal_characterisation_report(scenario)
        report["source_level_1_software"] = source.get("software")
        report["software"] = software_provenance()

        text = json.dumps(report, indent=2, allow_nan=False)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
