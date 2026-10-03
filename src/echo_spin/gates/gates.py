import numpy as np
from qutip import Qobj


def sqrt_zz_gate() -> Qobj:
    """Return the ideal two-qubit sqrt(ZZ) gate."""
    return Qobj(
        np.diag([1.0, 1.0j, 1.0j, 1.0]),
        dims=[[2, 2], [2, 2]],
    )