import argparse
import json
from pathlib import Path

from echo_spin.nv.registers import load_register_configuration


def main(argv=None) -> int:
    """Inspect a saved NV register and write its resolved configuration as JSON."""
    parser = argparse.ArgumentParser(
        description="Inspect an NV register configuration."
    )
    parser.add_argument(
        "configuration",
        type=Path,
        help="Register JSON file using metres and tesla.",
    )
    arguments = parser.parse_args(argv)

    try:
        register = load_register_configuration(arguments.configuration)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

    report = {
        "number_of_nv": register.number_of_nv,
        "units": {
            "positions": "m",
            "magnetic_fields": "T",
        },
        "positions": register.positions.tolist(),
        "local_frames": register.local_frames.tolist(),
        "magnetic_fields": register.magnetic_fields.tolist(),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
