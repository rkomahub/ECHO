# Level 7 — Uniform spacing comparison

## Outputs

- `spacing_scan.json`: requested separations, resolved candidate
  specifications, evaluated metrics, resources, rejection reasons,
  and software provenance.
- `figures/spacing_comparison.pdf`: target-pair gate duration and
  whole-register fidelity versus separation.

## Comparison conditions

All relative positions are scaled uniformly about the first
target site. Register size, orientations, site parameters,
field specification, and control strategies remain fixed.

Each candidate is revalidated and recalculated.
The gate duration is determined separately at every spacing.
Rejected candidates are retained without fabricated metric values.

## Result

For this zero-bare-detuning ideal model, couplings scale as d^-3
and gate duration as d^3. Accumulated phases, fidelities, and
pulse counts remain unchanged within numerical precision.

This scan establishes no optimal physical spacing.

## Limits and extensions

Finite pulses, shared-drive response, decoherence, uncertainty
ensembles, and fabrication feasibility are not evaluated.

Other register sizes, geometry comparisons, field scans, and
constrained optimisation remain future extensions.