from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from qutip import Qobj, qeye

from echo_spin.core.system import SpinSystem
from echo_spin.nv.fields import register_magnetic_fields
from echo_spin.nv.frames import validate_local_frame
from echo_spin.nv.geometry import register_local_frames, validate_positions
from echo_spin.nv.hamiltonians import nv_hamiltonian
from echo_spin.nv.interactions import (
    dipolar_geometry,
    dipolar_interaction,
)


def nv_register_hamiltonian(
    system: SpinSystem,
    nv_parameters: Sequence[dict],
    dipolar_couplings=None,
    dipolar_directions=None,
    positions=None,
    coupling_prefactor=None,
    local_frames=None,
) -> Qobj:
    """Construct the Hamiltonian of a register of non-interacting NV centers.

    Each NV center is represented by two consecutive spin-1 subsystems:

        electron site = 2 * i
        nuclear site  = 2 * i + 1

    Parameters
    ----------
    system : SpinSystem
        Composite spin system containing all NV electrons and nuclei.
    nv_parameters : sequence of dict
        Parameters of each NV center. Each dictionary must contain
        D, omega_e, Q, omega_n and A.

    Returns
    -------
    Qobj
        Hamiltonian of the complete NV register.
    """
    number_of_nv = len(nv_parameters)

    if local_frames is not None:
        if len(local_frames) != number_of_nv:
            raise ValueError("Provide one local frame per NV.")

        local_frames = [
            validate_local_frame(frame)
            for frame in local_frames
        ]

    if system.number_of_spins != 2 * number_of_nv:
        raise ValueError(
            "An NV register requires two spin subsystems per NV center."
        )

    for spin in system.spins:
        if spin != 1:
            raise ValueError(
                "NV electronic and 14N nuclear spins must both have spin 1."
            )

    hamiltonian = 0

    for i, parameters in enumerate(nv_parameters):
        electron_site = 2 * i
        nuclear_site = 2 * i + 1

        hamiltonian += nv_hamiltonian(
            system=system,
            electron_site=electron_site,
            nuclear_site=nuclear_site,
            D=parameters["D"],
            omega_e=parameters["omega_e"],
            Q=parameters["Q"],
            omega_n=parameters["omega_n"],
            A=parameters["A"],
            E=parameters.get("E", 0.0),
        )

    if positions is not None:
        if coupling_prefactor is None:
            raise ValueError(
                "coupling_prefactor must be provided when positions are used."
            )

        if (
            dipolar_couplings is not None
            or dipolar_directions is not None
        ):
            raise ValueError(
                "Provide either positions or explicit dipolar data, not both."
            )

        dipolar_couplings, dipolar_directions = dipolar_geometry(
            positions=positions,
            coupling_prefactor=coupling_prefactor,
        )

    if dipolar_couplings is not None or dipolar_directions is not None:
        if dipolar_couplings is None or dipolar_directions is None:
            raise ValueError(
                "Both dipolar_couplings and dipolar_directions must be provided."
            )

        hamiltonian += nv_dipolar_interaction(
            system=system,
            couplings=dipolar_couplings,
            directions=dipolar_directions,
            local_frames=local_frames,
        )

    return hamiltonian

def nv_dipolar_interaction(
    system: SpinSystem,
    couplings,
    directions,
    local_frames=None,
) -> Qobj:
    """Construct pairwise electronic dipolar interactions in local NV frames.

    Parameters
    ----------
    system : SpinSystem
        Register ordered as electron, nucleus, electron, nucleus, ...
    couplings : array-like
        (N, N) array of distance-dependent dipolar prefactors in
        angular-frequency units. Only entries i < j are used.
    directions : array-like
        (N, N, 3) array of laboratory-frame pair directions.
        Nonzero-coupling directions are normalized internally.
    local_frames : sequence of array-like, optional
        One right-handed orthonormal 3x3 matrix per NV, with local
        axes as laboratory-coordinate columns. Defaults to identities.

    Returns
    -------
    Qobj
        Sum of electronic pair interactions on the full register.

    Notes
    -----
    Each pair uses the tensor R_i.T (I - 3 r_hat r_hat.T) R_j.
    Local Hamiltonians and controls must use the same local frames.
    No secular or logical-subspace approximation is applied.
    """
    if system.number_of_spins == 0 or system.number_of_spins % 2 != 0:
        raise ValueError(
            "An NV register requires electron-nucleus subsystem pairs."
        )

    number_of_nv = system.number_of_spins // 2

    couplings = np.asarray(couplings, dtype=float)
    directions = np.asarray(directions, dtype=float)

    if (
        couplings.shape != (number_of_nv, number_of_nv)
        or not np.all(np.isfinite(couplings))
    ):
        raise ValueError("couplings must be a finite array of shape (N, N).")

    if (
        directions.shape != (number_of_nv, number_of_nv, 3)
        or not np.all(np.isfinite(directions))
    ):
        raise ValueError(
            "directions must be a finite array of shape (N, N, 3)."
        )

    if local_frames is None:
        frames = [np.eye(3) for _ in range(number_of_nv)]
    else:
        if len(local_frames) != number_of_nv:
            raise ValueError("Provide one local frame per NV.")

        frames = [
            validate_local_frame(frame)
            for frame in local_frames
        ]

    interaction = 0.0 * qeye(
        [int(2 * spin + 1) for spin in system.spins]
    )

    for i in range(number_of_nv):
        for j in range(i + 1, number_of_nv):
            # A zero coupling needs no direction or pair construction.
            if couplings[i, j] == 0.0:
                continue

            interaction += dipolar_interaction(
                system=system,
                site_i=2 * i,
                site_j=2 * j,
                coupling=couplings[i, j],
                direction=directions[i, j],
                frame_i=frames[i],
                frame_j=frames[j],
            )

    return interaction


@dataclass
class NVRegisterConfiguration:
    """Store ordered laboratory positions in metres, local frames and fields in tesla."""

    positions: np.ndarray
    local_frames: np.ndarray
    magnetic_fields: np.ndarray

    def __post_init__(self):
        """Validate and copy the static register arrays."""
        self.positions = validate_positions(self.positions)
        number_of_nv = len(self.positions)

        if len(self.local_frames) != number_of_nv:
            raise ValueError("Provide one local frame per NV.")

        self.local_frames = np.stack([
            validate_local_frame(frame)
            for frame in self.local_frames
        ])

        if np.iscomplexobj(self.magnetic_fields):
            raise ValueError("magnetic_fields must be real.")

        fields = np.asarray(self.magnetic_fields, dtype=float)
        if (
            fields.shape != (number_of_nv, 3)
            or not np.all(np.isfinite(fields))
        ):
            raise ValueError("magnetic_fields must be a finite (N, 3) array.")

        self.magnetic_fields = fields.copy()

    @property
    def number_of_nv(self) -> int:
        """Return the number of sites in the register."""
        return len(self.positions)

    @classmethod
    def from_geometry(
        cls,
        positions,
        nv_axes,
        transverse_references,
        field_at_reference,
        gradient=None,
        reference_position=(0.0, 0.0, 0.0),
        source_free=True,
    ):
        """Assemble a static register from laboratory geometry and an affine field."""
        positions = validate_positions(positions)

        return cls(
            positions=positions,
            local_frames=register_local_frames(
                positions, nv_axes, transverse_references
            ),
            magnetic_fields=register_magnetic_fields(
                positions=positions,
                field_at_reference=field_at_reference,
                gradient=gradient,
                reference_position=reference_position,
                source_free=source_free,
            ),
        )
