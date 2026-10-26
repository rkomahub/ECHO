import argparse
import json
from pathlib import Path

from echo_spin.protocols import ideal_protocol_report
from echo_spin.provenance import software_provenance


def main():
    """Save ideal candidate protocols from a Level 3 model report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_report", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    try:
        with arguments.model_report.open(encoding="utf-8") as stream:
            source = json.load(stream)

        report = ideal_protocol_report(source)
        report["software"] = software_provenance()

        text = json.dumps(report, indent=2, allow_nan=False)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
