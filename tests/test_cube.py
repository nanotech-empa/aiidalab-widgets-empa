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


def test_slice_controls_and_invalid_edit_clear_the_old_image():
    viewer = CubeViewer()
    viewer.set_cube(sample_cube())
    cut = viewer.slice2d
    cut.plane_preset.value = "1 0 0"
    cut.apply_plane_preset.click()
    cut.offset_txt.value = ".3"
    cut.show_contours.value = True
    cut.auto_color_range.value = False
    cut.vmin.value, cut.vmax.value = -30, 30
    cut.show_2d_checkbox.value = False
    assert cut.image.layout.display == "none"
    cut.show_2d_checkbox.value = True
    assert cut.image.layout.display == "" and cut.slice_png
    assert cut.return_settings()["normal"] == [1.0, 0.0, 0.0]
    viewer._export_slice()
    assert "Download 2D slice PNG" in viewer.slice_download.value
    cut.center_txt.value = "incomplete"
    assert not cut.image.value and not cut.slice_png
    assert "valid plane" in cut.error.value
    cut.center_on_atoms.click()
    assert cut.image.value and not cut.error.value
    viewer.viewer3d.show_plane_checkbox.value = True
    cut.res.value = 32
    assert viewer.viewer3d._plane_component is not None
    viewer.viewer3d.show_plane_checkbox.value = False
    viewer.set_cube(None)
    assert viewer.viewer3d._cube_component is None
    assert not cut.image.value
    viewer.close()


def test_render_callback_contains_only_selected_view_settings():
    submitted = []
    viewer = CubeViewer(on_render=submitted.append)
    viewer.render_button.click()
    assert "camera" in viewer.render_message.value
    viewer.set_cube(sample_cube())
    viewer.viewer3d.viewer._camera_orientation = [
        20.0,
        0,
        0,
        0,
        0,
        20.0,
        0,
        0,
        0,
        0,
        20.0,
        0,
        0,
        0,
        0,
        1.0,
    ]
    rows = viewer.viewer3d.isovalues
    rows.add_isovalue()
    rows.isovalues.children[-1].value = 0.005
    rows.isovalues.children[-1].color_widget.value = "#123456"
    viewer.width.value, viewer.height.value = 640, 480
    viewer.background.value = "#ABCDEF"
    viewer.opacity.value = 0.7
    viewer.tabs.selected_index = 2
    viewer.render_button.click()
    assert submitted[0]["iso"][-1] == [0.005, "#123456"]
    assert submitted[0]["background"] == "#ABCDEF"
    assert (submitted[0]["width"], submitted[0]["height"], submitted[0]["opacity"]) == (
        640,
        480,
        0.7,
    )
    rows.remove_isovalue()
    viewer.tabs.selected_index = 0
    viewer.viewer3d.update_plot()
    while rows.isovalues.children:
        rows.remove_isovalue()
    viewer.render_button.click()
    assert "at least one" in viewer.render_message.value
    assert len(submitted) == 1
    viewer.close()


def test_cube_switch_keeps_user_isovalues_within_new_range(tmp_path):
    viewer = CubeViewer()
    cube = sample_cube()
    # Use the actual file import path rather than the internal Cube setter.
    path = tmp_path / "cube.cube"
    with path.open("w") as handle:
        write_cube(handle, cube.ase_atoms, cube.data, origin=cube.origin / ANG_TO_BOHR)
    viewer.set_content(path.read_text())
    viewer.viewer3d.isovalues.isovalues.children[0].value = 0.01
    other = cube.copy()
    other.data[:] = -0.1
    viewer.set_cube(other)
    assert all(
        row.value == -0.1 for row in viewer.viewer3d.isovalues.isovalues.children
    )
    other = cube.copy()
    other.data[:] = 0.5
    viewer.set_cube(other)
    assert all(row.value == 0.5 for row in viewer.viewer3d.isovalues.isovalues.children)
    viewer.close()
