# Theory

## system.py

`SpinSystem` represents a generic composite quantum spin system. For each subsystem with spin quantum number \(s_i\), the local Hilbert-space dimension is

\[
d_i = 2s_i + 1
\]

For a system containing \(N\) subsystems, the total Hilbert-space dimension is

\[
\dim \mathcal H
=
\prod_{i=1}^{N} d_i
\]

The implementation supports both identical-spin and mixed-spin systems.

## operators.py

For a spin \(s\), the library constructs the angular-momentum operators \(S_x\), \(S_y\), and \(S_z\).

A local operator acting on subsystem \(i\) is embedded into the composite Hilbert space as

\[
O_i =
I_1 \otimes \cdots \otimes O \otimes \cdots \otimes I_N
\]

This allows operators to act on arbitrary subsystems in both uniform-spin and mixed-spin systems.