import argparse
import json
from pathlib import Path

from echo_spin.evaluation import finite_evaluation_report
from echo_spin.provenance import software_provenance


def main():
    """Save a finite-pulse evaluation from a Level 4 protocol report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--pulse-duration-us", type=float, required=True
    )
    parser.add_argument(
        "--shape", choices=("square", "sine"), default="square"
    )
    parser.add_argument("--steps", type=int, default=128)
    arguments = parser.parse_args()

    try:
        source = json.loads(
            arguments.protocol_report.read_text(encoding="utf-8")
        )
        report = finite_evaluation_report(
            source,
            arguments.pulse_duration_us,
            shape=arguments.shape,
            steps=arguments.steps,
        )
        report["software"] = software_provenance()

        text = json.dumps(report, indent=2, allow_nan=False)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
