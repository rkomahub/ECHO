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

For a spin \(s\), the library constructs the angular-momentum operators \(S_x\), \(S_y\), and \(S_z\). A local operator acting on subsystem \(i\) is embedded into the composite Hilbert space as

\[
O_i =
I_1 \otimes \cdots \otimes O \otimes \cdots \otimes I_N
\]

This allows operators to act on arbitrary subsystems in both uniform-spin and mixed-spin systems.

## hamiltonians.py

The electronic ground-state Hamiltonian of an NV center is implemented as

\[
H_{\mathrm e}
=
D S_z^2
+
\boldsymbol{\omega}_{e}\cdot\mathbf S
\]

where \(D\) is the zero-field splitting and \(\boldsymbol{\omega}_{e}\) is the electronic Larmor-frequency vector. All quantities appearing in a Hamiltonian must use compatible frequency/energy units. For example, if \(D\) is expressed in GHz, the components of \(\boldsymbol{\omega}_e\) must be expressed in GHz as well The NV electronic ground state is modeled as a spin-1 subsystem.