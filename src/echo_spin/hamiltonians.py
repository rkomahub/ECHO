import numpy as np
from collections.abc import Sequence

from qutip import Qobj

from echo_spin.operators import embed_operator, spin_operators
from echo_spin.system import SpinSystem

def electronic_nv_hamiltonian(
    system: SpinSystem,
    electron_site: int,
    D: float,
    omega_e: Sequence[float],
) -> Qobj:
    """Construct the electronic Hamiltonian of an NV center.

    The Hamiltonian is

        H_e = D Sz^2 + omega_e · S

    where D is the zero-field splitting and omega_e is the
    electronic Larmor-frequency vector.

    Parameters
    ----------
    system : SpinSystem
        Composite spin system containing the NV electron.
    electron_site : int
        Site corresponding to the electronic spin.
    D : float
        Zero-field splitting.
    omega_e : sequence of float
        Cartesian components (omega_x, omega_y, omega_z).

    Returns
    -------
    Qobj
        Electronic NV Hamiltonian acting on the full Hilbert space.
    """
    if system.spins[electron_site] != 1:
        raise ValueError("An NV electronic spin must have spin quantum number 1.")

    if len(omega_e) != 3:
        raise ValueError("omega_e must contain three Cartesian components.")

    sx, sy, sz = spin_operators(1)

    sx = embed_operator(sx, electron_site, system)
    sy = embed_operator(sy, electron_site, system)
    sz = embed_operator(sz, electron_site, system)

    omega_x, omega_y, omega_z = omega_e

    return (
        D * sz**2
        + omega_x * sx
        + omega_y * sy
        + omega_z * sz
    )

def nuclear_nv_hamiltonian(
    system: SpinSystem,
    nuclear_site: int,
    Q: float,
    omega_n: Sequence[float],
) -> Qobj:
    """Construct the nuclear Hamiltonian of an NV center.

    The Hamiltonian is

        H_n = Q Iz^2 + omega_n · I

    where Q is the nuclear quadrupole parameter and omega_n is the
    nuclear Larmor-frequency vector.
    """
    if system.spins[nuclear_site] != 1:
        raise ValueError(
            "A 14N nuclear spin must have spin quantum number 1."
        )

    if len(omega_n) != 3:
        raise ValueError(
            "omega_n must contain three Cartesian components."
        )

    ix, iy, iz = spin_operators(1)

    ix = embed_operator(ix, nuclear_site, system)
    iy = embed_operator(iy, nuclear_site, system)
    iz = embed_operator(iz, nuclear_site, system)

    omega_x, omega_y, omega_z = omega_n

    return (
        Q * iz**2
        + omega_x * ix
        + omega_y * iy
        + omega_z * iz
    )

def hyperfine_hamiltonian(
    system: SpinSystem,
    electron_site: int,
    nuclear_site: int,
    A: Sequence[Sequence[float]],
) -> Qobj:
    """Construct the electron-nuclear hyperfine Hamiltonian.

    The Hamiltonian is

        H_hf = sum_{a,b} S_a A_ab I_b

    where A is the 3x3 hyperfine tensor.
    """
    if electron_site == nuclear_site:
        raise ValueError(
            "Electron and nuclear spins must occupy different sites."
        )

    if system.spins[electron_site] != 1:
        raise ValueError(
            "An NV electronic spin must have spin quantum number 1."
        )

    if system.spins[nuclear_site] != 1:
        raise ValueError(
            "A 14N nuclear spin must have spin quantum number 1."
        )

    A = np.asarray(A, dtype=float)

    if A.shape != (3, 3):
        raise ValueError("The hyperfine tensor A must have shape (3, 3).")

    sx, sy, sz = spin_operators(1)
    ix, iy, iz = spin_operators(1)

    electron_operators = [
        embed_operator(sx, electron_site, system),
        embed_operator(sy, electron_site, system),
        embed_operator(sz, electron_site, system),
    ]

    nuclear_operators = [
        embed_operator(ix, nuclear_site, system),
        embed_operator(iy, nuclear_site, system),
        embed_operator(iz, nuclear_site, system),
    ]

    hamiltonian = 0

    for alpha in range(3):
        for beta in range(3):
            hamiltonian += (
                A[alpha, beta]
                * electron_operators[alpha]
                * nuclear_operators[beta]
            )

    return hamiltonian

def nv_hamiltonian(
    system: SpinSystem,
    electron_site: int,
    nuclear_site: int,
    D: float,
    omega_e: Sequence[float],
    Q: float,
    omega_n: Sequence[float],
    A: Sequence[Sequence[float]],
) -> Qobj:
    """Construct the complete single-NV Hamiltonian.

    The Hamiltonian is

        H_0 = H_e + H_n + H_hf

    with

        H_e  = D Sz^2 + omega_e · S
        H_n  = Q Iz^2 + omega_n · I
        H_hf = S · A · I
    """
    return (
        electronic_nv_hamiltonian(
            system=system,
            electron_site=electron_site,
            D=D,
            omega_e=omega_e,
        )
        + nuclear_nv_hamiltonian(
            system=system,
            nuclear_site=nuclear_site,
            Q=Q,
            omega_n=omega_n,
        )
        + hyperfine_hamiltonian(
            system=system,
            electron_site=electron_site,
            nuclear_site=nuclear_site,
            A=A,
        )
    )