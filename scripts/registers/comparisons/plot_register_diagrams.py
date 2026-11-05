import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from echo_spin.characterisation import characterise_longitudinal_scenario
from echo_spin.scenario import longitudinal_scenario_from_dict


def read_json(path):
    """Read a saved numerical report."""
    return json.loads(path.read_text(encoding="utf-8"))


def dominant_branches(spectrum, first, last):
    """Select the strongest transition from each ground-like nuclear level."""
    branches = []
    for lower in range(3):
        candidates = [
            line for line in spectrum["transitions"]
            if line["lower_eigenstate"] == lower
            and first <= line["upper_eigenstate"] < last
        ]
        if not candidates:
            raise ValueError("Missing electronic transition branch.")
        branches.append(max(candidates, key=lambda line: line["strength"]))
    if len({line["upper_eigenstate"] for line in branches}) != 3:
        raise ValueError("Electronic branches do not identify three distinct levels.")
    return branches


def main():
    """Draw the physical spin network and local nitrogen-resolved energy levels."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    output = arguments.output_directory
    if output.exists():
        parser.error("Choose a new output directory.")

    root = Path("results/linear/n3")
    spectra_path = (
        root / "physical/electron_nitrogen/"
        "level_02_characterisation/nitrogen_spectra.json"
    )
    gate_path = (
        root / "physical/electron_nitrogen/"
        "level_06_evaluation/nitrogen_selective_gate_frames.json"
    )
    spectra_report = read_json(spectra_path)
    gate = read_json(gate_path)
    scenario = longitudinal_scenario_from_dict(
        spectra_report["specification"]
    )
    if scenario.register.number_of_nv != 3:
        parser.error("This presentation script expects the three-site chain.")

    interactions = characterise_longitudinal_scenario(scenario)
    positions = scenario.register.positions * 1e9
    fields = scenario.register.magnetic_fields * 1e3
    couplings = interactions["couplings_rad_per_us"] / (2 * np.pi) * 1e3
    carriers = [
        item["carrier_rad_per_us"] for item in gate["calibrations"]
    ]

    output.mkdir(parents=True)
    (output / "diagram_data.json").write_text(
        json.dumps({
            "source_spectra": str(spectra_path),
            "source_gate": str(gate_path),
            "positions_nm": positions.tolist(),
            "fields_mT": fields.tolist(),
            "longitudinal_couplings_kHz": couplings.tolist(),
            "carrier_rad_per_us": carriers,
            "energy_layout": "Schematic band spacing; numerical labels are exact energies.",
        }, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    # Network: physical site positions; curved edges distinguish all pairs.
    figure, axis = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    x = positions[:, 0]
    spacing = x[1] - x[0]

    for i, j, radius in ((0, 1, 0.0), (1, 2, 0.0), (0, 2, -0.45)):
        axis.add_patch(FancyArrowPatch(
            (x[i], 0), (x[j], 0),
            arrowstyle="-",
            connectionstyle=f"arc3,rad={radius}",
            linewidth=2.4 if (i, j) == (0, 1) else 1.3,
            color="tab:red" if (i, j) == (0, 1) else "0.4",
            shrinkA=24, shrinkB=24,
        ))
        height = 0.13 * spacing if j - i == 1 else 0.64 * spacing
        axis.text(
            (x[i] + x[j]) / 2, height,
            rf"$J_{{{i}{j}}}/2\pi={couplings[i, j]:.3f}$ kHz",
            ha="center", fontsize=10,
        )

    axis.scatter(
        x, np.zeros(3), s=1100,
        c=["tab:red", "tab:red", "tab:blue"],
        edgecolors="white", linewidths=1.5, zorder=3,
    )
    for site in range(3):
        axis.text(
            x[site], 0, f"NV {site}",
            color="white", ha="center", va="center", zorder=4,
        )
        axis.text(
            x[site], -0.22 * spacing,
            rf"$B_z={fields[site, 2]:g}$ mT",
            ha="center",
        )
        axis.text(
            x[site], -0.37 * spacing,
            "Target" if site < 2 else "Spectator",
            ha="center", fontsize=9,
        )
    axis.set_xticks(x)
    axis.set_xlabel("Position along chain (nm)")
    axis.set_yticks([])
    axis.set_xlim(x[0] - 0.45 * spacing, x[-1] + 0.45 * spacing)
    axis.set_ylim(-0.55 * spacing, 0.9 * spacing)
    axis.set_title("Dipolar network: longitudinal design coefficients")
    for spine in ("left", "right", "top"):
        axis.spines[spine].set_visible(False)
    figure.savefig(output / "spin_network.pdf", bbox_inches="tight")
    plt.close(figure)

    # Exact energy labels on a schematic manifold layout.
    figure, axes = plt.subplots(
        1, 3, figsize=(12, 5.8), constrained_layout=True
    )
    band_names = [
        r"$m_s\simeq0$",
        r"$m_s\simeq-1$",
        r"$m_s\simeq+1$",
    ]
    centres = [0.0, 1.35, 2.7]

    for site, (axis, spectrum) in enumerate(zip(
        axes, spectra_report["spectra"]
    )):
        energies = np.asarray(
            spectrum["energies_relative_to_ground_rad_per_us"]
        ) / (2 * np.pi)
        lower_branches = dominant_branches(spectrum, 3, 6)
        upper_branches = dominant_branches(spectrum, 6, 9)

        columns = {index: index for index in range(3)}
        for line in lower_branches + upper_branches:
            columns[line["upper_eigenstate"]] = line["lower_eigenstate"]

        y = np.zeros(9)
        for band, centre in enumerate(centres):
            indices = np.arange(3 * band, 3 * band + 3)
            values = energies[indices]
            spread = max(float(np.ptp(values)), 1e-12)
            y[indices] = centre + 0.24 * (
                values - values.mean()
            ) / spread

            axis.axhspan(
                centre - 0.25, centre + 0.37,
                color="0.96", zorder=0,
            )
            axis.text(
                0.5, centre + 0.28, band_names[band],
                ha="center", fontsize=11, color="0.3",
            )

        for level in range(9):
            x_level = 0.17 + 0.33 * columns[level]
            colour = "0.45" if level >= 6 else "black"
            axis.hlines(
                y[level], x_level - 0.105, x_level + 0.105,
                color=colour, linewidth=1.6, zorder=3,
            )
            axis.text(
                x_level, y[level] + 0.045,
                f"{energies[level]:.3f}",
                ha="center", va="bottom",
                fontsize=8, color=colour,
            )

        # Show only the driven logical branches; retain upper levels
        # without arrows that cross the intermediate manifold.
        for line in lower_branches:
            lower = line["lower_eigenstate"]
            upper = line["upper_eigenstate"]
            x_arrow = 0.17 + 0.33 * columns[lower]
            selected = np.isclose(
                line["frequency_rad_per_us"], carriers[site],
                atol=1e-5, rtol=0,
            )
            colour = "tab:red" if selected else "tab:blue"
            axis.annotate(
                "",
                xy=(x_arrow, y[upper] - 0.015),
                xytext=(x_arrow, y[lower] + 0.015),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": colour,
                    "lw": 1.8 if selected else 1.1,
                },
                zorder=2,
            )
            frequency = line["frequency_rad_per_us"] / (2 * np.pi)
            axis.text(
                x_arrow + 0.025,
                (y[lower] + y[upper]) / 2,
                f"{frequency:.3f} MHz",
                rotation=90, ha="left", va="center",
                fontsize=8, color=colour,
            )

        axis.set_title(
            f"NV {site}: "
            rf"$B_z={fields[site, 2]:g}$ mT",
            fontsize=12,
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(-0.35, 3.15)
        axis.axis("off")

    figure.suptitle(
        "Local electron–¹⁴N energy structure\n"
        r"Labels: $(E-E_{\mathrm{ground}})/(2\pi)$ in MHz; "
        "vertical spacing schematic",
        fontsize=12,
    )
    figure.savefig(
        output / "nitrogen_energy_levels.pdf", bbox_inches="tight"
    )
    plt.close(figure)
    print(f"Saved both diagrams and their input summary in {output}")


if __name__ == "__main__":
    main()
