import numpy as np
from qutip import Qobj

from echo_spin.core.basis import (
    operator_from_basis,
    operator_in_basis,
)
from echo_spin.core.operators import embed_operator, spin_operators
from echo_spin.core.system import SpinSystem
from echo_spin.nv.frames import validate_local_frame
from echo_spin.nv.geometry import validate_positions


def dipolar_interaction(
    system: SpinSystem,
    site_i: int,
    site_j: int,
    coupling: float,
    direction,
    frame_i=None,
    frame_j=None,
) -> Qobj:
    """Construct dipolar coupling between spins represented in their local frames."""
    if site_i == site_j:
        raise ValueError("Dipolar interaction requires two different sites.")

    for site in (site_i, site_j):
        if (
            isinstance(site, (bool, np.bool_))
            or not isinstance(site, (int, np.integer))
            or not 0 <= site < system.number_of_spins
        ):
            raise ValueError("Interacting sites must be valid spin indices.")

    if np.iscomplexobj(coupling):
        raise ValueError("coupling must be a finite real scalar.")

    coupling_array = np.asarray(coupling, dtype=float)
    if coupling_array.shape != () or not np.isfinite(coupling_array):
        raise ValueError("coupling must be a finite real scalar.")

    if np.iscomplexobj(direction):
        raise ValueError("direction must be real.")

    direction = np.asarray(direction, dtype=float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("direction must contain three finite components.")

    # Normalize safely even when the supplied vector has very large components.
    scale = float(np.max(np.abs(direction)))
    if scale == 0.0:
        raise ValueError("direction must be non-zero.")

    scaled_direction = direction / scale
    r_hat = scaled_direction / np.linalg.norm(scaled_direction)

    rotation_i = (
        np.eye(3) if frame_i is None else validate_local_frame(frame_i)
    )
    rotation_j = (
        np.eye(3) if frame_j is None else validate_local_frame(frame_j)
    )

    # direction is expressed in the common laboratory frame.
    geometric_tensor = np.eye(3) - 3.0 * np.outer(r_hat, r_hat)

    # Local tensor: A_ab = sum_mu,nu R_i[mu,a] T[mu,nu] R_j[nu,b].
    local_tensor = np.einsum(
        "ma,mn,nb->ab",
        rotation_i,
        geometric_tensor,
        rotation_j,
        optimize=False,
    )

    operators_i = [
        embed_operator(operator, site_i, system)
        for operator in spin_operators(system.spins[site_i])
    ]
    operators_j = [
        embed_operator(operator, site_j, system)
        for operator in spin_operators(system.spins[site_j])
    ]

    interaction = 0.0 * operators_i[0]

    for a in range(3):
        for b in range(3):
            interaction += (
                local_tensor[a, b]
                * operators_i[a]
                * operators_j[b]
            )

    return float(coupling_array) * interaction


def dipolar_geometry(
    positions,
    coupling_prefactor: float,
):
    """Compute pairwise dipolar couplings and directions from positions.

    Parameters
    ----------
    positions : array-like
        Cartesian positions with shape (N, 3).
    coupling_prefactor : float
        Constant C defining J_ij = C / r_ij^3.

    Returns
    -------
    couplings : ndarray
        NxN matrix of dipolar coupling strengths.
    directions : ndarray
        NxNx3 array of unit vectors pointing from i to j.
    """
    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")

    number_of_spins = len(positions)

    couplings = np.zeros((number_of_spins, number_of_spins))
    directions = np.zeros((number_of_spins, number_of_spins, 3))

    for i in range(number_of_spins):
        for j in range(i + 1, number_of_spins):
            displacement = positions[j] - positions[i]
            distance = np.linalg.norm(displacement)

            if distance == 0.0:
                raise ValueError(
                    "Two spins cannot occupy the same position."
                )

            direction = displacement / distance
            coupling = coupling_prefactor / distance**3

            couplings[i, j] = coupling
            couplings[j, i] = coupling

            directions[i, j] = direction
            directions[j, i] = -direction

    return couplings, directions


def secular_interaction(
    interaction: Qobj,
    dressed_basis: list[Qobj],
) -> Qobj:
    """Return the diagonal secular part of an interaction Hamiltonian.

    The interaction is expressed in the dressed energy basis, all
    off-diagonal matrix elements are removed, and the result is
    transformed back to the original representation.
    """
    dressed_interaction = operator_in_basis(
        interaction,
        dressed_basis,
    )

    diagonal = np.diag(
        np.diag(dressed_interaction.full())
    )

    secular_dressed = Qobj(
        diagonal,
        dims=dressed_interaction.dims,
    )

    return operator_from_basis(
        secular_dressed,
        dressed_basis,
    )


def extract_effective_zz_coupling(
    secular_hamiltonian: Qobj,
    sz_i: Qobj,
    sz_j: Qobj,
) -> tuple[complex, Qobj]:
    """Project a secular interaction onto an effective Sz_i Sz_j coupling.

    Returns
    -------
    coupling : complex
        Effective coupling strength g.
    residual : Qobj
        Difference between the secular Hamiltonian and g Sz_i Sz_j.
    """
    zz_operator = sz_i * sz_j

    numerator = (zz_operator.dag() * secular_hamiltonian).tr()
    denominator = (zz_operator.dag() * zz_operator).tr()

    if np.isclose(denominator, 0.0):
        raise ValueError("The effective ZZ operator has zero norm.")

    coupling = numerator / denominator

    effective_hamiltonian = coupling * zz_operator
    residual = secular_hamiltonian - effective_hamiltonian

    return coupling, residual


def relative_operator_residual(
    operator: Qobj,
    residual: Qobj,
) -> float:
    """Return the relative Hilbert-Schmidt norm of a residual."""
    norm = operator.norm("fro")

    if np.isclose(norm, 0.0):
        raise ValueError("Cannot normalize by a zero operator.")

    return residual.norm("fro") / norm


def register_dipolar_tensors(
    positions,
    local_frames,
    coupling_prefactor: float,
) -> np.ndarray:
    """Return local dipolar tensors in rad/us for site positions in metres."""
    positions = validate_positions(positions)
    number_of_nv = len(positions)

    if len(local_frames) != number_of_nv:
        raise ValueError("Provide one local frame per NV.")

    frames = [validate_local_frame(frame) for frame in local_frames]

    if np.iscomplexobj(coupling_prefactor):
        raise ValueError("coupling_prefactor must be a finite real scalar.")

    prefactor = np.asarray(coupling_prefactor, dtype=float)
    if prefactor.shape != () or not np.isfinite(prefactor):
        raise ValueError("coupling_prefactor must be a finite real scalar.")

    couplings, directions = dipolar_geometry(
        positions, float(prefactor)
    )
    tensors = np.zeros((number_of_nv, number_of_nv, 3, 3))

    for i in range(number_of_nv):
        for j in range(i + 1, number_of_nv):
            direction = directions[i, j]
            geometry = np.eye(3) - 3.0 * np.outer(direction, direction)
            local_tensor = np.einsum(
                "ma,mn,nb->ab",
                frames[i],
                geometry,
                frames[j],
                optimize=False,
            )
            tensors[i, j] = couplings[i, j] * local_tensor
            tensors[j, i] = tensors[i, j].T

    if not np.all(np.isfinite(tensors)):
        raise ValueError("Dipolar tensors must contain finite values.")

    return tensors
