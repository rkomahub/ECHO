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

### Interacting NV register

The interacting NV-register Hamiltonian is constructed as

\[
H
=
\sum_i H_{\mathrm{NV}}^{(i)}
+
\sum_{i<j} H_{ij}^{\mathrm{dip}}
\]

where the dipolar terms currently couple the electronic spins of different NV centers.

Each pair \(i<j\) is included exactly once.

### Dipolar geometry

Given spin positions \(\mathbf r_i\), the relative direction and coupling are calculated as

\[
\hat{\mathbf r}_{ij}
=
\frac{\mathbf r_j-\mathbf r_i}
{|\mathbf r_j-\mathbf r_i|}
\]

and for a user-defined coupling prefactor  \(C\) 

\[
J_{ij}
=
\frac{C}{|\mathbf r_j-\mathbf r_i|^3}
\]

### Position-based NV interactions

The geometry of an NV register can be specified through the NV positions. From the positions, ECHO computes the relative directions and dipolar couplings automatically using the inverse-cube dependence

\[
J_{ij} \propto \frac{1}{r_{ij}^{3}}
\]

## frames.py

Cartesian frame rotations are represented by three-dimensional rotation matrices.

For a rotation by an angle \(\beta\) around the \(y\) axis,

\[
R_y(\beta)
=
\begin{pmatrix}
\cos\beta & 0 & \sin\beta\\
0 & 1 & 0\\
-\sin\beta & 0 & \cos\beta
\end{pmatrix}
\]

A vector transforms as

\[
\mathbf v' = R\mathbf v
\]

This machinery is used to express magnetic fields and NV axes in local crystal frames.

### Misaligned NV frames

For an NV whose local crystal frame is rotated by an angle \(\beta\), laboratory-frame vectors are transformed into the local frame as

\[
\mathbf v' = R(-\beta)\mathbf v
\]

The standard NV Hamiltonian can then be evaluated directly using the rotated magnetic-field vector.

## basis.py

The electronic Hamiltonian is diagonalized to obtain its energy eigenstates.

Given a unitary matrix \(U\) whose columns are an ordered set of dressed eigenstates, the dressed spin operators are defined as

\[
\widetilde S_\alpha
=
U S_\alpha U^\dagger
\]

so that they have the usual spin-matrix representation in the dressed basis.

### Dressed-state ordering

The eigenstates of \(H_e\) are generally mixtures of the bare
\(|+1\rangle\), \(|0\rangle\), and \(|-1\rangle\) states.

ECHO associates dressed eigenstates with these spin labels by maximizing
their overlap with the bare spin-1 basis. This defines the ordering used
to construct the dressed operators \(\widetilde S_x\),
\(\widetilde S_y\), and \(\widetilde S_z\).

### Secular dipolar interaction

The dipole-dipole interaction is expressed in the product basis of the
dressed electronic eigenstates.

When the dipolar coupling is much smaller than the electronic transition
frequencies, rapidly oscillating off-diagonal terms are neglected.

The secular interaction is therefore obtained by retaining the diagonal
part of the dipolar Hamiltonian in the dressed basis.

### Effective dressed-spin coupling

After the secular approximation, the interaction is projected onto the
longitudinal dressed-spin operator

\[
\widetilde S'_{z,1}\widetilde S_{z,2}
\]

to extract the effective coupling \(g\).

The residual between the complete secular Hamiltonian and the effective
\(ZZ\)-type interaction is retained to quantify the validity of the
approximation.

## controls.py

The driven Hamiltonian is written as

\[
H_{\mathrm{driven}}(t)
=
H^{\mathrm{free}}
+
H^{\mathrm{mw}}(t)
\]

The microwave contribution is constructed from slowly varying control amplitudes \(\Omega_{i,x}(t)\), \(\Omega_{i,y}(t)\), carrier frequencies \(\omega_i\), and phases \(\xi_i\).

The control operator includes both dressed electronic-spin operators and the weaker nuclear-spin contribution weighted by

\[
\widetilde\gamma
=
\frac{\gamma_n}{\gamma_e}
\]