import math
from collections.abc import Sequence


class SpinSystem:
    """Representation of a quantum system composed of arbitrary spins.

    Parameters
    ----------
    spins : sequence of float
        Spin quantum numbers of the subsystems.
        Example: [1, 1] represents two spin-1 systems.

    Attributes
    ----------
    spins : tuple of float
        Spin quantum numbers.
    dimensions : tuple of int
        Local Hilbert-space dimensions.
    dimension : int
        Total Hilbert-space dimension.
    """

    def __init__(self, spins: Sequence[float]):
        if len(spins) == 0:
            raise ValueError("At least one spin must be specified.")

        self.spins = tuple(spins)

        for spin in self.spins:
            if spin < 0:
                raise ValueError("Spin quantum numbers must be non-negative.")

            two_spin = 2 * spin

            if not float(two_spin).is_integer():
                raise ValueError(
                    "Spin quantum numbers must be integer or half-integer."
                )

        self.dimensions = tuple(
            int(2 * spin + 1)
            for spin in self.spins
        )

        self.dimension = math.prod(self.dimensions)

    @property
    def number_of_spins(self) -> int:
        """Return the number of spin subsystems."""
        return len(self.spins)

    def __repr__(self) -> str:
        return (
            f"SpinSystem(spins={self.spins}, "
            f"dimensions={self.dimensions}, "
            f"dimension={self.dimension})"
        )