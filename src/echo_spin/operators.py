from qutip import Qobj, jmat, qeye, tensor

from echo_spin.system import SpinSystem


def spin_operators(spin: float) -> tuple[Qobj, Qobj, Qobj]:
    """Return the angular-momentum operators Sx, Sy and Sz.

    Parameters
    ----------
    spin : float
        Spin quantum number.

    Returns
    -------
    tuple[Qobj, Qobj, Qobj]
        The operators Sx, Sy and Sz.
    """
    sx = jmat(spin, "x")
    sy = jmat(spin, "y")
    sz = jmat(spin, "z")

    return sx, sy, sz


def embed_operator(
    operator: Qobj,
    site: int,
    system: SpinSystem,
) -> Qobj:
    """Embed a local operator into the full Hilbert space.

    Parameters
    ----------
    operator : Qobj
        Operator acting on one subsystem.
    site : int
        Index of the subsystem on which the operator acts.
    system : SpinSystem
        Composite spin system.

    Returns
    -------
    Qobj
        Operator acting on the full Hilbert space.
    """
    if site < 0 or site >= system.number_of_spins:
        raise IndexError("Site index is outside the spin system.")

    if operator.shape != (
        system.dimensions[site],
        system.dimensions[site],
    ):
        raise ValueError(
            "Operator dimension does not match the selected subsystem."
        )

    factors = [
        qeye(d)
        for d in system.dimensions
    ]

    factors[site] = operator

    return tensor(factors)