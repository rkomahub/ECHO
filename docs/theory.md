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

### Projection onto an effective subspace

A full spin Hamiltonian can be restricted to a selected effective subspace. If the
columns of \(P\) are the selected basis states, the projected operator is

\[
O_{\mathrm{eff}}=P^\dagger O P
\]

For the two-NV gate, this allows the full two-qutrit Hamiltonian to be reduced to the
four-dimensional computational subspace formed by the two microwave-addressed
transitions.

The full spin-1 description is retained separately so that leakage into off-resonant
levels can later be studied explicitly.

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

## joas.py

The noiseless model is assembled as
\[
H_{\mathrm{driven}}(t)
=
H^{\mathrm{free}}
+
H^{\mathrm{mw}}(t)
\]
with
\[
H^{\mathrm{free}}
=
\sum_i H_{\mathrm{NV},i}^{0}
+
H^{\mathrm{int}}
\]
where \(H^{\mathrm{int}}\) is the effective secular interaction between NV centers.

## dynamics/propagators.py

For a time-independent Hamiltonian \(H\), the propagator is

\[
U(t)=e^{-iHt}.
\]

A rotating frame generated by \(G\) is defined through

\[
V(t)=e^{-iGt},
\]

so that the propagator in the rotating frame is

\[
U_{\mathrm{rot}}(t)=V^\dagger(t)U(t).
\]

For a time-dependent Hamiltonian, the propagator is the time-ordered exponential
\[
U(t,t_0)=\mathcal{T}\exp\left[-i\int_{t_0}^{t}H(t')\,dt'\right]
\]
ECHO approximates this evolution by dividing the interval into \(N\) steps of duration \(\Delta t\) and multiplying the corresponding short-time propagators in chronological order
\[
U(t,t_0)\approx e^{-iH(t_N)\Delta t}\cdots e^{-iH(t_1)\Delta t}
\]

For numerical propagation with microwave driving, the Hamiltonian can be transformed at each time step into a rotating frame generated by \(G\)
\[
V(t)=e^{-iGt}
\]
The rotating-frame Hamiltonian is
\[
H_{\mathrm{rot}}(t)
=
V^\dagger(t)
[H_{\mathrm{driven}}(t)-G]
V(t)
\]
This removes the fast evolution generated by \(G\) and allows the remaining Hamiltonian to vary more slowly.

### Transformed Hamiltonian

During a dynamical-decoupling sequence, the Hamiltonian in each free-evolution
interval can be transformed by the cumulative pulse propagator

\[
H_k
=
U_k H_{\mathrm{free}} U_k^\dagger
\]

This toggling-frame description will be used to study refocusing of single-qubit
detunings while retaining the desired two-qubit interaction.

## controls/pulse.py

Microwave control amplitudes can be represented by finite-duration pulse envelopes. A sine-shaped pulse of duration \(T\) and peak amplitude \(\Omega_{\max}\) is defined as

\[
\Omega(t)=\Omega_{\max}\sin\left(\frac{\pi t}{T}\right),
\qquad 0\leq t\leq T
\]

and the amplitude is zero outside the pulse interval. This envelope can be used as the time-dependent amplitude entering the microwave control Hamiltonian. For a sine-shaped \(\pi\) pulse, the Rabi frequency is related to the pulse duration by

\[
\Omega_{\mathrm{Rabi}}=\frac{\pi}{t_\pi}
\]

The required peak amplitude is therefore

\[
\Omega_{\max}
=
\frac{\pi}{2}\Omega_{\mathrm{Rabi}}
=
\frac{\pi^2}{2t_\pi}
\]

## gates/gates.py

The target two-qubit entangling operation is the \(\sqrt{ZZ}\) gate. In the computational basis

\[
\{|00\rangle,|01\rangle,|10\rangle,|11\rangle\}
\]

the target unitary is defined, up to a global phase, as

\[
U_{\sqrt{ZZ}}
=
\operatorname{diag}(1,i,i,1)
\]

This gate will be used as the target operation when validating dynamically decoupled dipolar evolution.

## control/rotations.py

In the instantaneous-pulse approximation, an ideal rotation around axis
\(\alpha\in\{x,y\}\) is

\[
R_\alpha(\theta)
=
\exp\left(-i\frac{\theta}{2}\sigma_\alpha\right)
\]

A refocusing pulse corresponds to \(\theta=\pi\).

## dynamics/sequences.py

Ideal dynamical-decoupling sequences can be described in the toggling frame. After a
sequence of instantaneous pulses, the cumulative pulse propagator determines the
Hamiltonian during the following free-evolution interval.

If \(P_k\) denotes the \(k\)-th pulse, the cumulative transformation after \(k\) pulses is

\[
U_k=P_kP_{k-1}\cdots P_1
\]

Using the convention
\[
|\psi_{\mathrm{togg}}\rangle=U_k^\dagger|\psi_{\mathrm{lab}}\rangle,
\]
the corresponding toggling-frame Hamiltonian is
\[
H_k=U_k^\dagger H_{\mathrm{free}}U_k.
\]

The laboratory-frame propagator is recovered by multiplying the toggling-frame propagator on the left by the final cumulative pulse operator.
The complete evolution is obtained by chronologically composing the propagators generated
during the individual free-evolution intervals. This representation allows refocusing of
single-qubit detunings and, later, selective retention of desired spin-spin interactions.

### Tensor-product and reduced-basis ordering conventions

ECHO uses the standard tensor-product convention in which, for a two-qubit
system,

\[
\mathcal{H}
=
\mathcal{H}_0\otimes\mathcal{H}_1
\]

an operator acting on qubit \(0\) is represented as

\[
O_0 = O\otimes I
\]

whereas an operator acting on qubit \(1\) is

\[
O_1 = I\otimes O
\]

The reduced two-qubit basis used in the Joas gate derivation follows a different
label ordering. It is

\[
\{
|+1,0\rangle,
|0,0\rangle,
|+1,-1\rangle,
|0,-1\rangle
\}
\]

where the first label denotes NV1 and the second label denotes NV2.

In this ordering, the NV1 state changes between adjacent basis states, whereas
the NV2 state changes between pairs. Consequently, when this basis is represented
using the generic ECHO two-qubit tensor convention,

\[
\mathrm{NV1}\longleftrightarrow \text{qubit }1,
\qquad
\mathrm{NV2}\longleftrightarrow \text{qubit }0
\]

Thus, within the Joas reduced model,

\[
U_{\mathrm{NV1}} = I\otimes U_1,
\qquad
U_{\mathrm{NV2}} = U_2\otimes I
\]

This mapping is specific to the ordering chosen for the Joas reduced basis and
does not modify ECHO's generic tensor-product convention. Explicit tests are used
to verify the mapping before constructing the gate dynamics.

### Effective dressed-state interaction

For two NV centers with different crystallographic orientations, the microscopic
dipolar interaction must account for the rotation between their local frames, e.g.
\[
\mathbf{S}'_1 = R(\beta)\mathbf{S}_1.
\]

After diagonalizing the single-NV Hamiltonians and applying the secular
approximation, the interaction reduces to
\[
H_{\mathrm{int}}^{\mathrm{eff}}
=
g(\mathbf r_{12},\beta,\mathbf B)\,
\widetilde S'_{z,1}\otimes\widetilde S_{z,2}.
\]

The geometric rotation is therefore already incorporated in the derivation of
the effective coupling \(g\) and the dressed operators. In the Joas
reproduction, \(g\) is taken directly from the reported effective dipolar
coupling, so no additional rotation of \(\widetilde S_{z,1}\) is applied.
Applying \(R(\beta)\) again would double-count the geometric transformation.

For a future geometry-resolved model, the rotation should instead be introduced
at the microscopic dipolar-Hamiltonian level, before dressing and secularization.

## dephasing.py

Pure dephasing is modeled through an exponential coherence decay
\[
    p(t)=e^{-t/T_2}.
\]
For a single qubit, the corresponding quantum channel is written in Kraus form as
\[
    \mathcal{E}_{T_2}(\rho)
    =
    K_0 \rho K_0^\dagger
    +
    K_1 \rho K_1^\dagger,
\]
with
\[
    K_0=\sqrt{\frac{1+p}{2}}\,I,
    \qquad
    K_1=\sqrt{\frac{1-p}{2}}\,\sigma_z.
\]
The populations are unchanged, while the coherences decay as
\[
    \rho_{01}(t)=e^{-t/T_2}\rho_{01}(0).
\]

For two qubits with independent dephasing, the channel is constructed from
the tensor products of the individual Kraus operators,
\[
    \mathcal{E}_{12}(\rho)
    =
    \sum_{i,j}
    (K_i^{(1)}\otimes K_j^{(2)})
    \rho
    (K_i^{(1)}\otimes K_j^{(2)})^\dagger.
\]

### Stochastic Dephasing Noise

A fluctuating longitudinal field can be modeled as a stochastic frequency
shift $\delta\omega(t)$ coupled to the qubit through

$$
H_{\mathrm{noise}}(t)
=
\frac{\delta\omega(t)}{2}\sigma_z
$$

The noise is modeled using an Ornstein--Uhlenbeck process with stationary
correlation function

$$
\left\langle
\delta\omega(t)\delta\omega(0)
\right\rangle
=
\sigma^2 e^{-|t|/\tau_c}
$$

where $\sigma$ determines the noise amplitude and $\tau_c$ its correlation
time. Since the noise couples longitudinally through $\sigma_z$, it produces
random phase accumulation without directly changing the qubit populations.