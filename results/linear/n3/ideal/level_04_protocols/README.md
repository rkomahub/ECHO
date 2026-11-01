# Level 4 — Candidate control protocols

## Outputs

- `protocol_report.json`: source model, target pair, common duration,
  whole-register target diagonal, per-site pulse times, exact sign
  averages, closure checks, pulse counts, and software provenance.
- `figures/modulations.pdf`: ideal sign modulation for each candidate.

## Candidates

- Free evolution preserves local terms and all interactions.
- Closed global Hahn echo cancels static local Z terms while
  preserving all ZZ interactions.
- Hadamard selective recoupling cancels local Z terms and unwanted
  ZZ interactions while retaining the selected pair.

All candidates use the same target-pair gate duration.
Closing pulses are included in schedules and resource counts.
A diagonal pair-sign average of one is not a self-interaction.

## Handoff

The next stage receives the saved source model, pulse schedules,
pulse-axis and angle conventions, duration, and target operation.

## Limits

These schedules assume instantaneous independently addressed pulses.
Cancellation is established for the static commuting longitudinal model.
Physical selectivity and neglected dynamics remain unvalidated.
No gate propagator or fidelity is calculated at this level.