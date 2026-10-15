"""Coordinate and signed-data regressions for the shared cube viewer."""

import tempfile

import numpy as np
from ase import Atoms
from ase.io.cube import read_cube, write_cube
from cubehandler import Cube
from cubehandler.cube import ANG_TO_BOHR

from aiidalab_widgets_empa.cube import CubeViewer, CubePlaneCut2D


def sample_cube():
    # Distinct origins and non-orthogonal axes expose coordinate swaps.
    shape = (8, 10, 12)
    cell = np.array([[4.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.3, 0.5, 6.0]])
    origin = np.array([-2.0, -3.0, -4.0])
    ijk = np.indices(shape).transpose(1, 2, 3, 0)
    xyz = origin + ijk @ (cell / np.array(shape)[:, None])
    data = xyz[..., 0] + 2 * xyz[..., 1] + 3 * xyz[..., 2]
    atoms = Atoms("H", positions=[origin + cell.sum(axis=0) / 2], cell=cell)
    return Cube(
        ase_atoms=atoms, origin=origin * ANG_TO_BOHR, cell=cell * ANG_TO_BOHR, data=data
    )


def test_origin_and_skewed_grid_interpolation():
    cube = sample_cube()
    widget = CubePlaneCut2D()
    widget.show_2d_checkbox.value = False
    widget.cube = cube
    indices = np.array([[[2.5, 3.5, 4.5]]])
    xyz = cube.origin / ANG_TO_BOHR + indices @ (
        cube.ase_atoms.cell.array / np.array(cube.data.shape)[:, None]
    )
    np.testing.assert_allclose(widget._cart_to_indexcoords(xyz), indices, atol=1e-12)
    widget.center_txt.value = " ".join(map(str, xyz[0, 0]))
    widget.width.value = 0.2
    widget.height.value = 0.2
    grid, _, values = widget._compute_slice_values()
    np.testing.assert_allclose(
        values, grid[..., 0] + 2 * grid[..., 1] + 3 * grid[..., 2], atol=1e-10
    )
    widget.center_txt.value = "100 100 100"
    assert np.isnan(widget._compute_slice_values()[2]).all()
    widget.close()


def test_viewer_preserves_volume_origin_and_signed_values(monkeypatch):
    captured = []
    original = write_cube

    def record(handle, atoms, data, **kwargs):
        with tempfile.TemporaryFile(mode="w+") as buffer:
            original(buffer, atoms, data, **kwargs)
            buffer.seek(0)
            captured.append(read_cube(buffer))
        original(handle, atoms, data, **kwargs)

    monkeypatch.setattr("ase.io.cube.write_cube", record)
    viewer = CubeViewer()
    cube = sample_cube()
    viewer.set_cube(cube)
    np.testing.assert_allclose(
        captured[0]["origin"], cube.origin / ANG_TO_BOHR, atol=1e-6
    )
    np.testing.assert_allclose(
        captured[0]["atoms"].positions, cube.ase_atoms.positions, atol=1e-6
    )
    np.testing.assert_allclose(captured[0]["data"], cube.data, atol=1e-5)
    assert viewer.slice2d.slice_png.startswith(b"\x89PNG")
    assert not viewer.slice2d.periodic
    viewer.close()
