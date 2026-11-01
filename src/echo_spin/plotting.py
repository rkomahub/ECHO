import matplotlib.pyplot as plt
import numpy as np

from echo_spin.decoupling.modulation import modulation_function


def plot_scenario_report(report):
    """Plot geometry, site fields and pair distances from a saved CLI report."""
    positions = np.asarray(report["positions"], dtype=float) * 1e9
    frames = np.asarray(report["local_frames"], dtype=float)
    fields = np.asarray(report["magnetic_fields"], dtype=float) * 1e3
    number_of_nv = len(positions)
    sites = np.arange(number_of_nv)

    specification = report.get("specification", {})
    pair = specification.get("task", {}).get("target_pair")
    constraints = report.get("environment", {}).get("constraints", {})
    bounds = (
        constraints["parameters"]
        if constraints.get("status") == "specified"
        else {}
    )

    # Geometry: equal coordinate scales preserve spatial proportions.
    geometry = plt.figure(figsize=(7, 6), constrained_layout=True)
    axis = geometry.add_subplot(111, projection="3d")

    span = np.ptp(positions, axis=0).max()
    span = span if span > 0 else 1.0
    centre = (positions.min(axis=0) + positions.max(axis=0)) / 2

    axis.scatter(*positions.T, color="black", s=35, label="NV sites")

    for index, position in enumerate(positions):
        axis.text(
            *(position + 0.025 * span),
            str(index),
        )

    nv_axes = frames[:, :, 2]
    axis.quiver(
        *positions.T,
        *nv_axes.T,
        length=0.15 * span,
        normalize=True,
        color="tab:blue",
        label="NV axes (direction only)",
    )

    field_norms = np.linalg.norm(fields, axis=1)
    nonzero = field_norms > 0
    if np.any(nonzero):
        directions = fields[nonzero] / field_norms[nonzero, None]
        axis.quiver(
            *positions[nonzero].T,
            *directions.T,
            length=0.10 * span,
            normalize=True,
            color="tab:orange",
            label="Field directions (not magnitude)",
        )

    if pair is not None:
        endpoints = positions[list(pair)]
        axis.plot(
            *endpoints.T,
            color="tab:red",
            linewidth=3,
            label=f"Target pair {tuple(pair)}",
        )

    for coordinate, setter in enumerate(
        (axis.set_xlim, axis.set_ylim, axis.set_zlim)
    ):
        setter(
            centre[coordinate] - 0.65 * span,
            centre[coordinate] + 0.65 * span,
        )

    axis.set_box_aspect((1, 1, 1))
    axis.set(
        xlabel="Laboratory x (nm)",
        ylabel="Laboratory y (nm)",
        zlabel="Laboratory z (nm)",
        title=f"Register geometry: N = {number_of_nv}",
    )
    axis.legend(loc="upper left", fontsize="small")

    # Field components expose gradients and orientation differences.
    field_figure, axis = plt.subplots(
        figsize=(7, 4), constrained_layout=True
    )
    for component, label in enumerate(("Bx", "By", "Bz")):
        axis.plot(
            sites,
            fields[:, component],
            marker="o",
            label=label,
        )

    axis.plot(
        sites,
        field_norms,
        "k--",
        marker=".",
        label="|B|",
    )
    if "maximum_field_t" in bounds:
        axis.axhline(
            bounds["maximum_field_t"] * 1e3,
            color="tab:red",
            linestyle=":",
            label="Maximum |B| at sites",
        )

    axis.set(
        xlabel="Site index",
        ylabel="Magnetic field (mT)",
        title="Laboratory-frame fields at NV sites",
    )
    axis.set_xticks(sites)
    axis.grid(alpha=0.25)
    axis.legend()

    # Distances are geometric diagnostics, not coupling strengths.
    differences = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    masked = np.ma.array(
        distances,
        mask=np.eye(number_of_nv, dtype=bool),
    )

    distance_figure, axis = plt.subplots(
        figsize=(6, 5), constrained_layout=True
    )
    colour_map = plt.get_cmap("viridis").copy()
    colour_map.set_bad("lightgrey")

    upper = distances.max() if distances.max() > 0 else 1.0
    image = axis.imshow(
        masked,
        cmap=colour_map,
        vmin=0.0,
        vmax=upper,
    )
    distance_figure.colorbar(image, ax=axis, label="Separation (nm)")

    if number_of_nv <= 12:
        for i in sites:
            for j in sites:
                if i != j:
                    axis.text(
                        j,
                        i,
                        f"{distances[i, j]:.1f}",
                        ha="center",
                        va="center",
                        color=(
                            "white"
                            if distances[i, j] < upper / 2
                            else "black"
                        ),
                        fontsize=8,
                    )

    if "minimum_separation_m" in bounds:
        minimum = bounds["minimum_separation_m"] * 1e9
        rows, columns = np.where(
            (~np.eye(number_of_nv, dtype=bool))
            & (distances < minimum)
        )
        if len(rows):
            axis.scatter(
                columns,
                rows,
                marker="x",
                color="red",
                label="Below minimum separation",
            )
            axis.legend()
        title = f"Pair distances; required minimum = {minimum:g} nm"
    else:
        title = "Pair distances; no minimum bound specified"

    axis.set(
        xlabel="Site index",
        ylabel="Site index",
        title=title,
    )
    axis.set_xticks(sites)
    axis.set_yticks(sites)

    return {
        "geometry": geometry,
        "fields": field_figure,
        "distances": distance_figure,
    }


def plot_characterisation_report(report):
    """Plot signed couplings, ideal gate times and isolated electronic lines."""
    couplings = np.asarray(report["couplings_rad_per_us"], dtype=float)
    times = np.asarray(report["ideal_gate_times_us"], dtype=float)
    number_of_nv = len(couplings)
    sites = np.arange(number_of_nv)
    diagonal = np.eye(number_of_nv, dtype=bool)

    # rad/us -> cycles/us -> kHz.
    coupling_khz = couplings * 1000 / (2 * np.pi)
    limit = np.max(np.abs(coupling_khz))
    limit = limit if limit > 0 else 1.0

    coupling_figure, axis = plt.subplots(
        figsize=(6, 5), constrained_layout=True
    )
    colour_map = plt.get_cmap("RdBu_r").copy()
    colour_map.set_bad("lightgrey")
    image = axis.imshow(
        np.ma.array(coupling_khz, mask=diagonal),
        cmap=colour_map,
        vmin=-limit,
        vmax=limit,
    )
    coupling_figure.colorbar(image, ax=axis, label="J / (2π) (kHz)")
    axis.set(
        xlabel="Site index",
        ylabel="Site index",
        title="Signed longitudinal couplings",
    )
    axis.set_xticks(sites)
    axis.set_yticks(sites)

    time_figure, axis_time = plt.subplots(
        figsize=(6, 5), constrained_layout=True
    )
    time_map = plt.get_cmap("viridis").copy()
    time_map.set_bad("lightgrey")
    valid_times = np.isfinite(times) & ~diagonal
    upper = np.max(times[valid_times]) if np.any(valid_times) else 1.0
    image = axis_time.imshow(
        np.ma.array(times, mask=~valid_times),
        cmap=time_map,
        vmin=0.0,
        vmax=upper,
    )
    time_figure.colorbar(image, ax=axis_time, label="Ideal gate time (µs)")
    axis_time.set(
        xlabel="Site index",
        ylabel="Site index",
        title="Ideal pair gate times; grey = undefined",
    )
    axis_time.set_xticks(sites)
    axis_time.set_yticks(sites)

    if number_of_nv <= 12:
        for i in sites:
            for j in sites:
                if i == j:
                    continue
                axis.text(
                    j, i, f"{coupling_khz[i, j]:.3g}",
                    ha="center", va="center", fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
                axis_time.text(
                    j, i,
                    f"{times[i, j]:.3g}" if valid_times[i, j] else "∞",
                    ha="center", va="center", fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )

    spectrum_figure = plot_local_transition_spectra(
        report["local_electrons"],
        title="Isolated electronic transitions: diagnostic local Sx",
    )

    spectrum_figure = plot_local_transition_spectra(
        report["local_electrons"],
        title="Isolated electronic transitions: diagnostic local Sx",
    )

    return {
        "couplings": coupling_figure,
        "ideal_gate_times": time_figure,
        "electronic_transitions": spectrum_figure,
    }


def plot_dipolar_comparison(report, component="real"):
    """Compare saved projected and longitudinal dipolar matrices on one scale."""
    if component not in ("real", "imag"):
        raise ValueError("component must be 'real' or 'imag'.")

    diagnostic = report["dipolar_projection"]
    if "matrices_rad_per_us" not in diagnostic:
        raise ValueError(
            "Regenerate the model report with saved dipolar matrices."
        )

    saved = diagnostic["matrices_rad_per_us"]
    projected = (
        np.asarray(saved["projected"]["real"])
        + 1j * np.asarray(saved["projected"]["imag"])
    )
    longitudinal = (
        np.asarray(saved["longitudinal"]["real"])
        + 1j * np.asarray(saved["longitudinal"]["imag"])
    )
    matrices = (
        projected,
        longitudinal,
        projected - longitudinal,
    )

    number_of_nv = len(report["subsystem_dimensions"])
    dimension = 2 ** number_of_nv
    labels = [
        f"|{index:0{number_of_nv}b}⟩"
        for index in range(dimension)
    ]
    values = [
        getattr(matrix, component)
        for matrix in matrices
    ]
    limit = max(float(np.max(np.abs(value))) for value in values)
    limit = limit if limit > 0 else 1.0

    figure, axes = plt.subplots(
        1, 3,
        figsize=(16, 5),
        constrained_layout=True,
    )
    titles = (
        "Projected dipolar Hamiltonian",
        "Longitudinal dipolar approximation",
        "Difference: projected − longitudinal",
    )

    for axis, value, title in zip(axes, values, titles):
        image = axis.imshow(
            value,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_title(title)
        axis.set_xlabel("Input basis state")
        axis.set_ylabel("Output basis state")
        axis.set_xticks(range(dimension))
        axis.set_yticks(range(dimension))
        axis.set_xticklabels(labels, rotation=90, fontsize=8)
        axis.set_yticklabels(labels, fontsize=8)

    figure.colorbar(
        image,
        ax=axes.tolist(),
        label=f"{component.capitalize()} matrix element (rad/µs)",
        shrink=0.85,
    )
    figure.suptitle(
        f"Dipolar terms only — {component} part; "
        "static comparison, not a dynamical error estimate"
    )
    return figure


def plot_protocol_modulations(report):
    """Plot saved sign schedules and pulse events for every candidate."""
    protocols = report["protocols"]
    duration = report["duration_us"]
    target_pair = set(report["target_pair"])

    figure, axes = plt.subplots(
        len(protocols),
        1,
        sharex=True,
        squeeze=False,
        figsize=(10, max(3, 2.5 * len(protocols))),
        constrained_layout=True,
    )

    for axis, (name, protocol) in zip(axes[:, 0], protocols.items()):
        schedules = protocol["pulse_times_us"]

        for site, pulse_times in enumerate(schedules):
            pulses = np.asarray(pulse_times, dtype=float)
            boundaries = np.unique(np.concatenate(([0.0, duration], pulses)))
            midpoints = boundaries[:-1] + np.diff(boundaries) / 2
            signs = modulation_function(midpoints, pulses)

            offset = 3 * site
            colour = "tab:red" if site in target_pair else "tab:blue"

            axis.stairs(
                offset + signs,
                boundaries,
                color=colour,
                linewidth=1.5,
            )
            axis.axhline(
                offset, color="grey", linewidth=0.5, alpha=0.4
            )

            for time in pulses:
                axis.plot(
                    [time, time],
                    [offset - 1.2, offset + 1.2],
                    color=colour,
                    linestyle=":",
                    linewidth=0.8,
                )

            # Show the final control sign after any closing pulse.
            final_sign = modulation_function([duration], pulses)[0]
            axis.plot(
                duration,
                offset + final_sign,
                marker="o",
                color=colour,
                markersize=4,
                clip_on=False,
            )

        axis.set_yticks([3 * site for site in range(len(schedules))])
        axis.set_yticklabels([f"Site {site}" for site in range(len(schedules))])
        axis.set_ylim(-1.5, 3 * (len(schedules) - 1) + 1.5)
        axis.set_title(
            f"{name}: {protocol['individual_pulse_count']} pulses, "
            f"{protocol['distinct_pulse_time_count']} distinct pulse times"
        )
        axis.set_xlim(0, duration)

    axes[-1, 0].set_xlabel("Time (µs)")
    figure.suptitle(
        "Ideal sign modulation: ±1 around each site baseline\n"
        "Red: target pair; blue: spectators; dotted lines: pulses; "
        "dots: final signs"
    )
    return figure


def plot_gate_evaluation(report):
    """Plot saved whole-register fidelities, resources and residual phases."""
    protocols = report["protocols"]
    names = list(protocols)
    labels = [name.replace("_", "\n") for name in names]
    positions = np.arange(len(names))

    performance, axes = plt.subplots(
        1, 2, figsize=(11, 4), constrained_layout=True
    )

    fidelities = [
        protocols[name]["average_gate_fidelity"]
        for name in names
    ]
    axes[0].bar(positions, fidelities, color="tab:blue")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Whole-register average gate fidelity")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels)
    for position, fidelity in zip(positions, fidelities):
        axes[0].text(
            position, fidelity + 0.02, f"{fidelity:.6g}",
            ha="center",
        )

    individual = [
        protocols[name]["resources"]["individual_pulse_count"]
        for name in names
    ]
    distinct = [
        protocols[name]["resources"]["distinct_pulse_time_count"]
        for name in names
    ]
    axes[1].bar(
        positions - 0.18, individual, width=0.36,
        label="Individual pulses",
    )
    axes[1].bar(
        positions + 0.18, distinct, width=0.36,
        label="Distinct pulse times",
    )
    axes[1].set_ylabel("Count, including closing pulses")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    performance.suptitle(
        f"Ideal register gates — common duration "
        f"{report['duration_us']:.6g} µs"
    )

    local_phases = [
        np.asarray(
            protocols[name]["phases"]["local_generator_phases_rad"]
        )
        for name in names
    ]
    pair_phases = [
        np.asarray(
            protocols[name]["phases"][
                "residual_pair_generator_phases_rad"
            ]
        )
        for name in names
    ]
    limit = max(
        float(np.max(np.abs(values)))
        for values in local_phases + pair_phases
    )
    limit = limit if limit > 0 else 1.0

    phases, axes = plt.subplots(
        2, len(names),
        squeeze=False,
        figsize=(4 * len(names), 7),
        constrained_layout=True,
    )
    colour_map = plt.get_cmap("RdBu_r").copy()
    colour_map.set_bad("lightgrey")

    for column, name in enumerate(names):
        number_of_nv = len(local_phases[column])
        sites = np.arange(number_of_nv)

        axes[0, column].imshow(
            local_phases[column][None, :],
            cmap=colour_map,
            vmin=-limit,
            vmax=limit,
            aspect="auto",
        )
        axes[0, column].set_title(name.replace("_", " "))
        axes[0, column].set_xticks(sites)
        axes[0, column].set_yticks([])
        axes[0, column].set_xlabel("Site index")
        axes[0, column].set_ylabel("Local residual φᵢ")

        # Show each pair once: i < j.
        image = axes[1, column].imshow(
            np.ma.array(
                pair_phases[column],
                mask=np.tri(number_of_nv, dtype=bool),
            ),
            cmap=colour_map,
            vmin=-limit,
            vmax=limit,
        )
        axes[1, column].set_xticks(sites)
        axes[1, column].set_yticks(sites)
        axes[1, column].set_xlabel("Site j")
        axes[1, column].set_ylabel("Site i")
        axes[1, column].set_title("Residual pair coefficients Δφᵢⱼ")

    phases.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="Generator coefficient (rad)",
        shrink=0.85,
    )
    phases.suptitle(
        "Unwrapped residual phases — common colour scale\n"
        "Grey: duplicate or self-pairs; no local corrections"
    )

    return {
        "performance": performance,
        "residual_phases": phases,
    }


def plot_spacing_comparison(report):
    """Plot saved gate durations and fidelities while marking rejected spacings."""
    records = report["records"]
    evaluated = sorted(
        (record for record in records if record["status"] == "evaluated"),
        key=lambda record: record["requested_pair_separation_m"],
    )
    rejected = [
        record["requested_pair_separation_m"] * 1e9
        for record in records
        if record["status"] == "rejected"
    ]

    figure, axes = plt.subplots(
        1, 2, figsize=(11, 4), constrained_layout=True
    )
    distances = [
        record["requested_pair_separation_m"] * 1e9
        for record in evaluated
    ]

    axes[0].plot(
        distances,
        [record["duration_us"] for record in evaluated],
        "o-",
        label="Calculated gate duration",
    )
    axes[0].set_ylabel("Target-pair gate duration (µs)")
    axes[0].set_title("Duration recomputed at each separation")

    names = sorted({
        name
        for record in evaluated
        for name in record["protocols"]
    })
    markers = ("o", "s", "^")
    styles = ("-", "--", ":")

    for index, name in enumerate(names):
        available = [
            record for record in evaluated
            if name in record["protocols"]
        ]
        axes[1].plot(
            [
                record["requested_pair_separation_m"] * 1e9
                for record in available
            ],
            [
                record["protocols"][name]["average_gate_fidelity"]
                for record in available
            ],
            marker=markers[index % len(markers)],
            linestyle=styles[index % len(styles)],
            label=name.replace("_", " "),
        )

    axes[1].set_ylabel("Whole-register average gate fidelity")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Ideal control performance")

    for axis in axes:
        axis.set_xlabel("Target-pair separation (nm)")
        axis.grid(alpha=0.25)
        if rejected:
            axis.plot(
                rejected,
                [0.03] * len(rejected),
                transform=axis.get_xaxis_transform(),
                linestyle="none",
                marker="x",
                color="red",
                label="Rejected; vertical position is arbitrary",
            )
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize="small")

    if not evaluated:
        for axis in axes:
            axis.text(
                0.5, 0.5, "No evaluated configurations",
                transform=axis.transAxes,
                ha="center",
            )

    figure.suptitle(
        "Uniform geometry scaling — deterministic ideal longitudinal model"
    )
    return figure


def plot_pulse_duration_comparison(report):
    """Plot finite-pulse gate errors and required peak Rabi frequencies."""
    figure, axes = plt.subplots(
        1, 2, figsize=(11, 4), constrained_layout=True
    )
    styles = {"square": "-", "sine": "--"}
    colors = {
        "free": "C0",
        "closed_global_hahn": "C1",
        "hadamard_selective": "C2",
    }

    for shape, style in styles.items():
        records = sorted(
            (
                record for record in report["records"]
                if record["status"] == "evaluated"
                and record["pulse_shape"] == shape
            ),
            key=lambda record: record["pulse_duration_us"],
        )
        if not records:
            continue

        durations_ns = [
            record["pulse_duration_us"] * 1000
            for record in records
        ]
        for name in records[0]["protocols"]:
            errors = [
                max(
                    1 - record["protocols"][name]["average_gate_fidelity"],
                    1e-15,
                )
                for record in records
            ]
            axes[0].loglog(
                durations_ns,
                errors,
                marker="o",
                linestyle=style,
                color=colors.get(name),
                label=f"{name.replace('_', ' ')}: {shape}",
            )

        axes[1].loglog(
            durations_ns,
            [
                record["peak_rabi_frequency_rad_per_us"] / (2 * np.pi)
                for record in records
            ],
            marker="o",
            linestyle=style,
            label=shape,
        )

    axes[0].set_ylabel(r"Whole-register gate error $1-F_{\rm avg}$")
    axes[0].set_title("Display floor: $10^{-15}$")
    axes[1].set_ylabel("Peak Rabi frequency (MHz)")
    axes[1].set_title("Equal duration and rotation area")

    for axis in axes:
        axis.set_xlabel("Pulse duration (ns)")
        axis.grid(alpha=0.25, which="both")
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize="small")

    rejected = sum(
        record["status"] == "rejected"
        for record in report["records"]
    )
    figure.suptitle(
        f"Finite independent qubit controls — "
        f"T = {report['duration_us']:.6g} µs; "
        f"{rejected} rejected shape-duration cases"
    )
    return figure


def plot_amplitude_error_comparison(report):
    """Plot whole-register infidelity against systematic amplitude error."""
    records = sorted(
        report["records"], key=lambda record: record["amplitude_error"]
    )
    errors_percent = [
        100 * record["amplitude_error"] for record in records
    ]
    figure, axis = plt.subplots(
        figsize=(7, 4.5), constrained_layout=True
    )

    for name in records[0]["protocols"]:
        axis.semilogy(
            errors_percent,
            [
                max(record["protocols"][name]["infidelity"], 1e-15)
                for record in records
            ],
            "o-",
            label=name.replace("_", " "),
        )

    axis.axvline(0, color="grey", linestyle=":", linewidth=1)
    axis.set_xlabel(r"Amplitude error $100\varepsilon$ (%)")
    axis.set_ylabel(r"Whole-register gate infidelity $1-F_{\rm avg}$")
    axis.set_title(
        f"{report['pulse_shape'].capitalize()} pulses: "
        f"{1000 * report['pulse_duration_us']:g} ns; "
        f"T = {report['duration_us']:.6g} µs"
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    figure.text(
        0.5, 0.01,
        "Independent addressing; fixed timing; display floor 10⁻¹⁵",
        ha="center", fontsize="small",
    )
    return figure


def plot_phase_cycle_comparison(reports):
    """Compare selective-gate amplitude sensitivity across pulse phase cycles."""
    if not reports:
        raise ValueError("Provide at least one labelled scan report.")

    reference = next(iter(reports.values()))
    for report in reports.values():
        for key in (
            "source_protocol_report",
            "pulse_shape",
            "pulse_duration_us",
            "duration_us",
        ):
            if report[key] != reference[key]:
                raise ValueError(f"Incompatible scan setting: {key}.")

    figure, axis = plt.subplots(
        figsize=(7, 4.5), constrained_layout=True
    )

    for label, report in reports.items():
        records = sorted(
            report["records"],
            key=lambda record: record["amplitude_error"],
        )
        axis.semilogy(
            [100 * record["amplitude_error"] for record in records],
            [
                max(
                    record["protocols"]["hadamard_selective"]["infidelity"],
                    1e-15,
                )
                for record in records
            ],
            "o-",
            label=label,
        )

    axis.set_xlabel(r"Amplitude error $100\varepsilon$ (%)")
    axis.set_ylabel(r"Whole-register gate infidelity $1-F_{\rm avg}$")
    axis.set_title(
        f"Selective recoupling — {reference['pulse_shape']} pulses, "
        f"{1000 * reference['pulse_duration_us']:g} ns"
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    return figure


def plot_electronic_populations(report):
    """Plot logical population transfer and sampled electronic leakage."""
    records = report["records"]
    times = [record["time_us"] for record in records]
    figure, axes = plt.subplots(
        1, 2, figsize=(10, 4), constrained_layout=True
    )

    for key, label in (
        ("mean_exchange_probability", "Logical exchange"),
        ("mean_other_logical_transfer_probability", "Other logical transfer"),
    ):
        axes[0].plot(
            times, [record[key] for record in records], label=label
        )

    axes[0].axhline(
        0, color="grey", linestyle="--", label="Longitudinal prediction"
    )
    axes[0].set_ylabel("Mean population-transfer probability")
    axes[0].set_title("Dynamics within the logical subspace")
    axes[0].legend(fontsize="small")

    axes[1].plot(
        times,
        [record["mean_leakage_probability"] for record in records],
        ".",
        label="Sampled leakage",
    )
    axes[1].set_ylabel("Mean leakage probability")
    axes[1].set_title("Population outside the logical subspace")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].legend(fontsize="small")

    for axis in axes:
        axis.set_xlabel("Time (µs)")
        axis.grid(alpha=0.25)

    figure.suptitle("Full electronic free evolution — no control pulses")
    return figure


def plot_local_transition_spectra(
    spectra,
    *,
    title="Local transition lines",
    frequency_range_mhz=None,
    strength_label="Strength",
):
    """Plot saved transition frequencies and squared coupling matrix elements."""
    if not spectra:
        raise ValueError("Provide at least one local spectrum.")

    figure, axes = plt.subplots(
        len(spectra),
        1,
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(8, max(3, 1.5 * len(spectra))),
        constrained_layout=True,
    )
    for spectrum, axis in zip(spectra, axes[:, 0]):
        frequencies = np.array([
            line["frequency_rad_per_us"] / (2 * np.pi)
            for line in spectrum["transitions"]
        ])
        strengths = np.array([
            line["strength"] for line in spectrum["transitions"]
        ])
        if frequency_range_mhz is not None:
            lower, upper = frequency_range_mhz
            selected = (frequencies >= lower) & (frequencies <= upper)
            frequencies, strengths = frequencies[selected], strengths[selected]
            axis.set_xlim(lower, upper)

        axis.vlines(frequencies, 0, strengths, color="tab:blue")
        axis.set_ylabel(f"Site {spectrum['site']}\n{strength_label}")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)

    axes[-1, 0].set_xlabel("Transition frequency (MHz)")
    figure.suptitle(title)
    return figure
