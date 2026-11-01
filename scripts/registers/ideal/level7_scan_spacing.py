import argparse
import json
from pathlib import Path

from echo_spin.comparisons import scan_pair_separations
from echo_spin.provenance import software_provenance


def main():
    """Save a prescribed spacing scan inherited from a Level 6 evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--separations-nm",
        nargs="+",
        type=float,
        required=True,
        help="Requested target-pair separations in nanometres.",
    )
    arguments = parser.parse_args()

    try:
        with arguments.evaluation_report.open(encoding="utf-8") as stream:
            source = json.load(stream)

        specification = source[
            "source_implementation"
        ]["source_protocol_report"]["source_model"]["specification"]

        records = scan_pair_separations(
            specification,
            [value * 1e-9 for value in arguments.separations_nm],
        )
        evaluated = sum(
            record["status"] == "evaluated" for record in records
        )

        report = {
            "report_version": 1,
            "comparison": "uniform_geometry_scaling",
            "source_specification": specification,
            "source_level_6_software": source.get("software"),
            "software": software_provenance(),
            "requested_separations_nm": arguments.separations_nm,
            "evaluated_count": evaluated,
            "rejected_count": len(records) - evaluated,
            "records": records,
            "conventions": {
                "anchor": "First target site.",
                "geometry_change": "Uniform scaling of all relative positions.",
                "fields": "Original affine-field specification retained.",
                "duration": "Recomputed for the target pair at every point.",
                "rejections": "Retained with reasons; no silent removal.",
            },
            "limits": [
                "Deterministic ideal longitudinal calculations only.",
                "No physical addressing or finite-pulse response.",
                "No uncertainty ensemble or numerical optimisation.",
                "Unit selective fidelity is expected in this ideal model.",
                "Recorded geometry provenance describes the baseline; "
                "each record explicitly supplies its scale and new positions.",
            ],
        }

        text = json.dumps(report, indent=2, allow_nan=False)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
