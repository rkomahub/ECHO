import argparse
import json
from pathlib import Path

from echo_spin.comparisons import scan_amplitude_errors
from echo_spin.provenance import software_provenance


def main():
    """Save a systematic amplitude-error scan at fixed pulse duration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--errors", nargs="+", type=float, required=True)
    parser.add_argument("--pulse-duration-us", type=float, required=True)
    parser.add_argument(
        "--shape", choices=("square", "sine"), default="square"
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--phase-cycle", choices=("x", "xy4", "xy8_targets"), default="x"
    )
    arguments = parser.parse_args()

    try:
        source = json.loads(
            arguments.protocol_report.read_text(encoding="utf-8")
        )
        report = scan_amplitude_errors(
            source,
            arguments.errors,
            arguments.pulse_duration_us,
            shape=arguments.shape,
            steps=arguments.steps,
            phase_cycle=arguments.phase_cycle,
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
