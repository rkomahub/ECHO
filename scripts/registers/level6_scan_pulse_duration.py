import argparse
import json
from pathlib import Path

from echo_spin.comparisons import scan_pulse_durations
from echo_spin.provenance import software_provenance


def main():
    """Save a finite-pulse duration comparison from a protocol report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--durations-us", nargs="+", type=float, required=True
    )
    parser.add_argument("--steps", type=int, default=128)
    arguments = parser.parse_args()

    try:
        source = json.loads(
            arguments.protocol_report.read_text(encoding="utf-8")
        )
        report = scan_pulse_durations(
            source, arguments.durations_us, steps=arguments.steps
        )
        report["software"] = software_provenance()

        text = json.dumps(report, indent=2, allow_nan=False)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
