import argparse
import csv
import json
from pathlib import Path

import numpy as np

from echo_spin.provenance import software_provenance


def main():
    """Summarise saved gate evaluations and spacing comparisons without simulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_report", type=Path)
    parser.add_argument("spacing_report", type=Path)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()

    try:
        with arguments.evaluation_report.open(encoding="utf-8") as stream:
            evaluation = json.load(stream)
        with arguments.spacing_report.open(encoding="utf-8") as stream:
            spacing = json.load(stream)

        protocol_report = evaluation[
            "source_implementation"
        ]["source_protocol_report"]
        specification = protocol_report["source_model"]["specification"]

        if specification != spacing["source_specification"]:
            raise ValueError("Reports describe different baseline specifications.")

        a, b = protocol_report["target_pair"]
        geometry = specification["register"]
        if "positions" in geometry:
            positions = np.asarray(geometry["positions"], dtype=float)
        else:
            from echo_spin.nv.registers import register_configuration_from_dict
            positions = register_configuration_from_dict(geometry).positions
        baseline_distance = float(np.linalg.norm(positions[b] - positions[a]))

        rows = []
        groups = [{
            "dataset": "baseline",
            "requested_pair_separation_m": baseline_distance,
            "duration_us": evaluation["duration_us"],
            "status": "evaluated",
            "reason": None,
            "protocols": evaluation["protocols"],
        }]
        groups.extend(
            {"dataset": "spacing_scan", **record}
            for record in spacing["records"]
        )

        for group in groups:
            protocols = group.get("protocols", {})
            for name, result in protocols.items() if protocols else [("", {})]:
                resources = result.get("resources", {})
                rows.append({
                    "dataset": group["dataset"],
                    "separation_nm": group["requested_pair_separation_m"] * 1e9,
                    "status": group["status"],
                    "reason": group["reason"],
                    "protocol": name,
                    "duration_us": group.get("duration_us"),
                    "average_gate_fidelity": result.get("average_gate_fidelity"),
                    "individual_pulse_count": resources.get(
                        "individual_pulse_count"
                    ),
                    "distinct_pulse_time_count": resources.get(
                        "distinct_pulse_time_count"
                    ),
                })

        summary = {
            "report_version": 1,
            "source_evaluation": str(arguments.evaluation_report),
            "source_spacing_scan": str(arguments.spacing_report),
            "specification": specification,
            "assessment": evaluation["assessment"],
            "software": software_provenance(),
            "rows": rows,
            "conclusion_scope": "Deterministic ideal longitudinal control only.",
            "benchmarking_status": "Not performed.",
        }
        text = json.dumps(summary, indent=2, allow_nan=False)

        output = arguments.output_directory
        paths = [output / "summary.json", output / "comparison.csv"]
        if any(path.exists() for path in paths):
            raise FileExistsError("Summary outputs already exist.")

        output.mkdir(parents=True, exist_ok=True)
        with paths[0].open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
        with paths[1].open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
