import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def fit_sector(lengths, values, errors):
    """Fit a positive logical sector to a zero-offset exponential."""
    def model(length, amplitude, decay):
        """Return the sector's amplitude-times-exponential reference."""
        return amplitude * decay**length

    parameters, _ = curve_fit(
        model,
        lengths,
        values,
        sigma=errors,
        absolute_sigma=True,
        p0=[0.95, 0.95],
        bounds=([0.0, 0.0], [1.0, 1.0]),
        maxfev=20000,
    )

    residuals = values - model(lengths, *parameters)
    reduced_chi_squared = (
        np.sum((residuals / errors) ** 2)
        / (len(lengths) - 2)
    )
    return parameters, residuals, reduced_chi_squared


parser = argparse.ArgumentParser()
parser.add_argument("archive")
args = parser.parse_args()

archive_path = Path(args.archive)

with np.load(archive_path, allow_pickle=False) as archive:
    if archive["clifford_family"].item() != "local":
        raise ValueError("This analysis requires a local-Clifford archive.")

    lengths = archive["lengths"]
    labels = list(archive["readout_labels"].astype(str))
    samples = archive["readout_samples"]

sector_names = [
    "nv1_logical_z",
    "nv2_logical_z",
    "logical_zz",
]
columns = [labels.index(name) for name in sector_names]

selected = lengths > 0
lengths = lengths[selected]
data = samples[selected][:, :, columns]

if len(np.unique(lengths)) < 3:
    raise ValueError("At least three distinct positive lengths are required.")

means = data.mean(axis=1)
errors = data.std(axis=1, ddof=1) / np.sqrt(data.shape[1])

if np.any(errors <= 0):
    raise ValueError("Positive circuit standard errors are required.")

fits = [
    fit_sector(lengths, means[:, column], errors[:, column])
    for column in range(3)
]

# Resample complete circuit vectors, preserving correlations between sectors.
rng = np.random.default_rng(123)
bootstrap_parameters = np.full((500, 3, 2), np.nan)

for repetition in range(500):
    indices = rng.integers(
        0,
        data.shape[1],
        size=(len(lengths), data.shape[1]),
    )
    resampled = data[
        np.arange(len(lengths))[:, None],
        indices,
    ].mean(axis=1)

    for column in range(3):
        try:
            parameters, _, _ = fit_sector(
                lengths,
                resampled[:, column],
                errors[:, column],
            )
            bootstrap_parameters[repetition, column] = parameters
        except RuntimeError:
            pass

figure, axis = plt.subplots(figsize=(7, 4.5))
grid = np.linspace(lengths.min(), lengths.max(), 200)

for column, name in enumerate(sector_names):
    parameters, residuals, chi_squared = fits[column]
    amplitude, decay = parameters

    boot = bootstrap_parameters[:, column]
    valid = np.all(np.isfinite(boot), axis=1)
    boot = boot[valid]

    if len(boot) == 0:
        raise RuntimeError(f"No successful bootstrap fits for {name}.")

    interval = np.percentile(boot[:, 1], [2.5, 97.5])
    boundary_fraction = np.mean(
        np.any(
            (boot < 1e-6) | (boot > 1 - 1e-6),
            axis=1,
        )
    )

    print(
        f"{name}: amplitude={amplitude:.6f}, p={decay:.6f}, "
        f"bootstrap interval=[{interval[0]:.6f}, {interval[1]:.6f}], "
        f"reduced chi-squared={chi_squared:.4f}, "
        f"boundary fraction={boundary_fraction:.3f}, "
        f"successful bootstrap fits={len(boot)}/500"
    )
    print("Residuals =", residuals)

    points = axis.errorbar(
        lengths,
        means[:, column],
        yerr=errors[:, column],
        fmt="o",
        capsize=3,
        label=name,
    )
    axis.plot(
        grid,
        amplitude * decay**grid,
        color=points.lines[0].get_color(),
    )

axis.set_xlabel("Clifford length")
axis.set_ylabel("Unnormalized logical sector")
axis.set_title("Exploratory local RB sector fits")
axis.legend()
figure.tight_layout()

output = (
    Path("results/figures")
    / f"{archive_path.stem}_local_sectors.png"
)
output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output, dpi=180)
plt.close(figure)
print("Saved figure:", output)
