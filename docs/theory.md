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

where \(D\) is the zero-field splitting and \(\boldsymbol{\omega}_{e}\) is the electronic Larmor-frequency vector. All quantities appearing in a Hamiltonian must use compatible frequency/energy units. For example, if \(D\) is expressed in GHz, the components of \(\boldsymbol{\omega}_e\) must be expressed in GHz as well. The NV electronic ground state is modeled as a spin-1 subsystem.

### Nuclear contribution

The \(^{14}\mathrm N\) nuclear spin is modeled as \(I=1\), with Hamiltonian

\[
H_{\mathrm n}
=
Q I_z^2
+
\boldsymbol{\omega}_{n}\cdot\mathbf I
\]

where \(Q\) is the nuclear quadrupole parameter and \(\boldsymbol{\omega}_{n}\) is the nuclear Larmor-frequency vector.

### Hyperfine interaction

The interaction between the NV electronic spin and the \(^{14}\mathrm N\) nuclear spin is implemented as

\[
H_{\mathrm{hf}}
=
\mathbf S\cdot A\cdot\mathbf I
=
\sum_{\alpha,\beta}
S_\alpha A_{\alpha\beta}I_\beta
\]

where \(A\) is the \(3\times3\) hyperfine tensor.

### Complete single-NV Hamiltonian

The complete single-NV Hamiltonian is constructed using the previously defined electronic, nuclear, and hyperfine contributions.

\[
H_0
=
H_{\mathrm e}
+
H_{\mathrm n}
+
H_{\mathrm{hf}}
\]

### NV register

A register containing \(N\) NV centers is represented using the subsystem ordering

\[
(S_0,I_0,S_1,I_1,\ldots,S_{N-1},I_{N-1})
\]

The Hamiltonian of the non-interacting register is

\[
H_{\mathrm{register}}
=
\sum_{i=0}^{N-1}H_{\mathrm{NV}}^{(i)}
\]

For \(^{14}\mathrm{NV}\), every electronic and nuclear subsystem has spin 1, giving total Hilbert-space dimension

\[
\dim\mathcal H = 3^{2N}=9^N
\]

## interactions.py

The magnetic dipole-dipole interaction between two spins is implemented as

\[
H_{ij}^{\mathrm{dip}}
=
J_{ij}
\left[
\mathbf S_i\cdot\mathbf S_j
-
3(\mathbf S_i\cdot\hat{\mathbf r}_{ij})
(\mathbf S_j\cdot\hat{\mathbf r}_{ij})
\right]
\]

where \(J_{ij}\) is the coupling strength and \(\hat{\mathbf r}_{ij}\) is the unit vector connecting the two spins.