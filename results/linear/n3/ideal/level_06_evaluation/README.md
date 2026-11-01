# Level 6 — Whole-register gate evaluation

## Outputs

- `evaluation_report.json`: target and propagated operations, fidelity,
  residual generator phases, resources, numerical checks, source
  implementation, and software provenance.
- `figures/performance.pdf`: whole-register fidelity and pulse counts.
- `figures/residual_phases.pdf`: local and pair residual coefficients.

Reports generated after the final consistency update also contain
`phase_prediction_error`.

## Numerical checks

- Unitarity of each propagated operation.
- Agreement with exact schedule-integrated phases, including global phase.
- Agreement of selective recoupling with the whole-register target.

## Interpretation

Fidelity averages over pure states of the complete logical register.
The target acts on sites (0, 1) and requires spectator identity.

Local coefficients multiply Z_i; corresponding rotation angles are
twice these coefficients. Pair residuals subtract the desired target
phase. All phase coefficients are unwrapped.

## Limits

Independent instantaneous addressing is assumed despite coincident
local transition frequencies. Nuclear dynamics, physical leakage,
finite pulses and shared-drive response are not represented.

Unit fidelity demonstrates ideal schedule correctness, not
experimentally achievable fidelity.