# Level 5 — Ideal control implementation

## Output

`implementation_report.json` contains:

- the source protocol and model reports;
- chronological pulse-event times;
- addressed sites at each event;
- free-evolution intervals;
- event-operator counts and timeline checks;
- pulse conventions and software provenance.

## Reconstruction

Pass `source_protocol_report` to
`echo_spin.implementation.implement_ideal_protocols`.

The returned objects include the drift Hamiltonian, whole-register
target, pulse propagators, and free-evolution intervals.

Simultaneous local pulses are combined into one event operator.
Individual local pulse counts remain in the source protocol report.

## Timeline convention

Free interval 0, pulse event 0, free interval 1, and so on.
There is one more free interval than pulse events.
A closing pulse at the final time is followed by a zero-duration
interval.

## Handoff to Level 6

Propagate the complete laboratory-frame operation in chronological
order, including closing pulses. Compare against the saved target,
allowing a global phase but no local corrections.

## Limits

This implementation uses instantaneous independently addressed
X-pi rotations. Finite pulses, physical shared-drive response,
nuclear dynamics and leakage are not simulated here.

The Level 4 modulation figure already visualises these ideal
schedules; no additional Level 5 figure is required.