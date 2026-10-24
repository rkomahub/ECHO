import json

import numpy as np
import pytest

from echo_spin.nv.registers import load_register_configuration


@pytest.mark.parametrize("use_lattice", [False, True])
def test_load_register_configuration_preserves_geometry_and_fields(
    tmp_path, use_lattice
):
    """Recover the same physical register from explicit positions or a lattice."""
    parameters = {
        "nv_axes": [[0, 0, 1], [0, 0, 1]],
        "transverse_references": [[1, 0, 0], [1, 0, 0]],
        "field_at_reference": [0, 0, 0.01],
    }
    if use_lattice:
        parameters["lattice"] = {
            "shape": [2, 1, 1],
            "lattice_vectors": [
                [15e-9, 0, 0],
                [0, 15e-9, 0],
                [0, 0, 15e-9],
            ],
        }
    else:
        parameters["positions"] = [[0, 0, 0], [15e-9, 0, 0]]

    path = tmp_path / "register.json"
    path.write_text(json.dumps(parameters), encoding="utf-8")
    register = load_register_configuration(path)

    np.testing.assert_allclose(
        register.positions,
        [[0, 0, 0], [15e-9, 0, 0]],
        atol=1e-22,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        register.local_frames, np.tile(np.eye(3), (2, 1, 1))
    )
    np.testing.assert_allclose(
        register.magnetic_fields, [[0, 0, 0.01], [0, 0, 0.01]]
    )


@pytest.mark.parametrize(
    "parameters",
    [{}, {"positions": [], "lattice": {}}],
)
def test_load_register_configuration_rejects_ambiguous_geometry(
    tmp_path, parameters
):
    """Require exactly one spatial description in a configuration."""
    path = tmp_path / "register.json"
    path.write_text(json.dumps(parameters), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        load_register_configuration(path)
