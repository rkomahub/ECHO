import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from echo_spin.characterisation import characterise_local_electrons
from echo_spin.plotting import plot_local_transition_spectra
from echo_spin.provenance import software_provenance
from echo_spin.scenario import longitudinal_scenario_from_dict


def read_json(path):
    """Read a saved numerical report."""
    return json.loads(path.read_text(encoding="utf-8"))


def add_carriers(figure, carriers):
    """Mark the three intended carrier frequencies on every site panel."""
    colours = ["tab:red", "tab:orange", "tab:purple"]
    for axis in figure.axes:
        for site, frequency in enumerate(carriers):
            axis.axvline(
                frequency / (2 * np.pi),
                color=colours[site],
                linestyle="--",
                linewidth=0.8,
                label=f"Carrier {site}",
            )
    figure.axes[0].legend(fontsize=7, ncol=3)


def main():
    """Present saved physical spectra, finite controls and nuclear gate results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    output = arguments.output_directory
    if output.exists():
        parser.error("Choose a new output directory.")

    root = Path("results/linear/n3")
    electronic_path = (
        root / "physical/electronic/level_06_evaluation/"
        "shared_selective_gate.json"
    )
    nuclear_path = (
        root / "physical/electron_nitrogen/level_06_evaluation/"
        "nitrogen_selective_gate_frames.json"
    )
    spectra_path = (
        root / "physical/electron_nitrogen/level_02_characterisation/"
        "nitrogen_spectra.json"
    )
    comparison_path = (
        root / "comparisons/level_06_evaluation/"
        "ideal_physical_comparison.json"
    )

    electronic = read_json(electronic_path)
    nuclear = read_json(nuclear_path)
    nuclear_spectra = read_json(spectra_path)
    comparison = read_json(comparison_path)
    trial = electronic["source_local_trial"]
    specification = trial["cases"]["gradient"]["specification"]

    # Static 3x3 diagonalisation only; no time propagation.
    spectra = characterise_local_electrons(
        longitudinal_scenario_from_dict(specification),
        field_profile_tesla=trial["field_profile_tesla"],
    )
    for spectrum in spectra:
        for key in ("zeeman_vector_rad_per_us", "energies_rad_per_us"):
            spectrum[key] = np.asarray(spectrum[key]).tolist()
    electronic_carriers = {}
    for event in electronic["schedule"]["events"]:
        for tone in event["tones"]:
            electronic_carriers[tone["intended_site"]] = (
                tone["carrier_rad_per_us"]
            )
    electronic_carriers = [
        electronic_carriers[site] for site in range(3)
    ]
    nuclear_carriers = [
        item["carrier_rad_per_us"] for item in nuclear["calibrations"]
    ]
    finest = max(
        nuclear["results"], key=lambda record: record["steps_per_event"]
    )
    nuclear_metrics = finest["frames"]["calibrated_carriers"]
    electronic_metrics = comparison["records"]["hadamard_selective"]["physical"]

    # Save the resolved plotting inputs before producing figures.
    output.mkdir(parents=True)
    plot_data = {
        "software": software_provenance(),
        "source_reports": [
            str(electronic_path), str(nuclear_path),
            str(spectra_path), str(comparison_path),
        ],
        "electronic_spectra": spectra,
        "electronic_carriers_rad_per_us": electronic_carriers,
        "nuclear_carriers_rad_per_us": nuclear_carriers,
        "nuclear_steps_per_event": finest["steps_per_event"],
        "nuclear_metrics": nuclear_metrics,
        "electronic_metrics": electronic_metrics,
    }
    (output / "plot_data.json").write_text(
        json.dumps(plot_data, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    for name, lines, carriers, title in (
        (
            "electronic_addressing",
            spectra,
            electronic_carriers,
            "Electronic transitions and selective-gate carriers",
        ),
        (
            "nitrogen_addressing",
            nuclear_spectra["spectra"],
            nuclear_carriers,
            "Electron–nitrogen transitions and calibrated carriers",
        ),
    ):
        figure = plot_local_transition_spectra(
            lines,
            title=title,
            frequency_range_mhz=(2495, 2605),
            strength_label="Strength\n(rad/µs)²",
        )
        add_carriers(figure, carriers)
        figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
        plt.close(figure)

    schedule = electronic["schedule"]
    figure, axis = plt.subplots(figsize=(9, 2.8), constrained_layout=True)
    colours = {"X": "tab:blue", "Y": "tab:orange"}
    for event in schedule["events"]:
        start = event["start_us"]
        width = event["end_us"] - start
        for tone in event["tones"]:
            site = tone["intended_site"]
            axis.broken_barh(
                [(start, width)],
                (site - 0.28, 0.56),
                facecolors=colours[tone["logical_axis"]],
            )
    axis.set_yticks([0, 1, 2], ["Target 0", "Target 1", "Spectator 2"])
    axis.set_ylim(-0.6, 2.6)
    axis.set_xlim(0, schedule["duration_us"])
    axis.set_xlabel("Elapsed time (µs)")
    axis.set_title("Finite shared-tone control schedule")
    axis.legend(handles=[
        Patch(color=colours[phase], label=f"{phase} phase")
        for phase in ("X", "Y")
    ])
    axis.grid(axis="x", alpha=0.2)
    figure.savefig(output / "finite_control_timeline.pdf", bbox_inches="tight")
    plt.close(figure)

    labels = ["Ideal", "Electronic", "¹⁴N polarised", "¹⁴N unpolarised"]
    fidelity = [
        comparison["records"]["hadamard_selective"]["ideal_fidelity"],
        electronic_metrics["average_gate_fidelity"],
        nuclear_metrics["polarised"]["average_gate_fidelity"],
        nuclear_metrics["unpolarised"]["average_gate_fidelity"],
    ]
    leakage = [
        electronic_metrics["average_leakage"],
        nuclear_metrics["polarised"]["average_leakage"],
        nuclear_metrics["unpolarised"]["average_leakage"],
    ]
    figure, axes = plt.subplots(
        1, 2, figsize=(9, 3.6), constrained_layout=True
    )
    axes[0].bar(range(4), fidelity, color=[
        "0.65", "tab:blue", "tab:green", "tab:orange"
    ])
    axes[0].set_xticks(range(4), labels, rotation=25, ha="right")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Whole-register average gate fidelity")
    for index, value in enumerate(fidelity):
        axes[0].text(
            index, value + 0.015, f"{value:.4f}",
            ha="center", fontsize=8,
        )
    axes[1].semilogy(range(3), leakage, "o", color="tab:red")
    axes[1].set_xticks(range(3), labels[1:], rotation=25, ha="right")
    axes[1].set_ylabel("Average leakage probability")
    axes[1].grid(axis="y", which="both", alpha=0.2)
    figure.savefig(output / "model_preparation_comparison.pdf", bbox_inches="tight")
    plt.close(figure)

    print(f"Saved four PDFs and plotting inputs in {output}")


if __name__ == "__main__":
    main()
