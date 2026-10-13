import argparse
import json

import numpy as np

parser = argparse.ArgumentParser(
    description="Inspect saved physical RB data without rerunning pulses."
)
parser.add_argument("archive")
args = parser.parse_args()

with np.load(args.archive, allow_pickle=False) as archive:
    metadata = json.loads(archive["metadata_json"].item())
    lengths = archive["lengths"]
    samples = archive["samples"]
    labels = list(archive["readout_labels"].astype(str))
    readouts = archive["readout_samples"]
    gate_counts = archive["gate_counts"]
    recovery_counts = archive["recovery_counts"]
    words = json.loads(archive["catalog_words_json"].item())

if readouts.shape != samples.shape + (len(labels),):
    raise ValueError("Readout dimensions do not match the circuit samples.")

indices = {label: index for index, label in enumerate(labels)}

reconstructed_return = (
    readouts[:, :, indices["joint_logical_survival"]]
    + readouts[:, :, indices["nv1_logical_z"]]
    + readouts[:, :, indices["nv2_logical_z"]]
    + readouts[:, :, indices["logical_zz"]]
) / 4

np.testing.assert_allclose(
    reconstructed_return,
    samples,
    atol=1e-9,
    rtol=0.0,
)

compiler_counts = np.array([
    [
        sum(index < 4 for index in word),
        sum(index == 4 for index in word),
    ]
    for word in words
])

print("Clifford family:", metadata["clifford_family"])
print("Dephasing included:", metadata["dephasing_included"])
print("Circuits per length:", samples.shape[1])
print(
    "Exact compiler mean counts (single-qubit, entangling):",
    compiler_counts.mean(axis=0),
)

selected_labels = [
    "nv1_ground_return",
    "nv2_ground_return",
    "joint_logical_survival",
    "nv1_logical_z",
    "nv2_logical_z",
    "logical_zz",
    "joint_ground_return",
]

for row, length in enumerate(lengths):
    print(f"\nClifford length {int(length)}:")

    for label in selected_labels:
        values = readouts[row, :, indices[label]]
        standard_error = (
            values.std(ddof=1) / np.sqrt(len(values))
        )
        print(
            f"  {label}: {values.mean():.6f} "
            f"+/- {standard_error:.6f}"
        )

    print(
        "  Total primitives including recovery:",
        (gate_counts[row] + recovery_counts[row]).sum(axis=1),
    )
