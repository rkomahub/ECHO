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