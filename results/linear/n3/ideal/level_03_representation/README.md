# Level 3 — Dynamical representation

## Selected model

One logical electronic qubit per NV, with site zero the leftmost
tensor factor:

H = c I + sum_i delta_i Z_i / 2 + sum_(i<j) J_ij Z_i Z_j.

Hamiltonian coefficients are in rad/us. Time is in microseconds.
For three NV centres, the Hilbert-space dimension is eight.

## Saved output

`model_report.json` contains:

- the original scenario specification;
- subsystem dimensions and tensor-order conventions;
- longitudinal coefficients and the Hamiltonian diagonal;
- Pauli rotation and bare logical-embedding conventions;
- the approximation assessment;
- optional static dipolar-projection diagnostics;
- software provenance.

## Reconstruction

Pass the decoded report to
`echo_spin.representation.build_longitudinal_model`.

The returned object supplies:

- `system`: logical subsystem description;
- `hamiltonian`: the drift Qobj;
- `pauli_operators`: embedded X, Y and Z for every site;
- `assessment`: the ideal-model assumptions and omissions.

The saved report retains the optional physical-reference diagnostics;
the model builder does not repeat that assessment.

## Assessment limits

This is an ideal control-design baseline, not a validated physical
reduction. The static diagnostic excludes local electronic energies,
nuclei, drives and noise.

Omitted in-subspace terms and coupling to excluded electronic states
are measured separately. Their norms are angular frequencies,
not gate-error probabilities.

Identical aligned sites can support resonant transverse exchange.
A large electronic splitting alone does not justify dropping it.

The static reference diagnostic is limited to three electronic
qutrits. Larger logical registers remain supported without that
diagnostic.

## Handoff to Level 4

Use the reconstructed Hamiltonian and Pauli operators, together with
the saved target pair and control conventions, to construct ideal
pulse schedules. Preserve the physical-assessment limitations.