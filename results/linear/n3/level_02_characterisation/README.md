# Level 2 — Register characterisation

## Numerical output

`characterisation_report.json` contains:

- pair distances and local-frame electronic dipolar tensors;
- signed longitudinal couplings, projection-induced local shifts,
  and the constant Hamiltonian offset;
- ideal pair gate times and zero-coupling pairs;
- isolated electronic energies and local-Sx transition strengths;
- inter-site spectral-overlap diagnostics;
- input specification, software provenance, and modelling limits.

## Figures

- `figures/couplings.pdf`: signed J/(2π), in kHz.
- `figures/ideal_gate_times.pdf`: ideal pair gate times, in microseconds.
- `figures/electronic_transitions.pdf`: isolated-site transition lines
  in MHz, with dimensionless local-Sx strengths.

## Limits

Spectra exclude nuclei and inter-site shifts.
The longitudinal projection is assumed rather than validated.
Spectral separation does not establish physical addressability.
Ideal gate times are not achieved physical gate fidelities.