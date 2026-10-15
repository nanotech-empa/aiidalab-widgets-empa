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


def test_saved_view_restores_camera_surfaces_and_slice_in_3d_tab():
    viewer = CubeViewer()
    viewer.set_cube(sample_cube())
    viewer.viewer3d.viewer._camera_orientation = (np.eye(4) * 20).ravel().tolist()
    viewer.slice2d.offset_txt.value = ".3"
    viewer.slice2d.normal_txt.value = "1 0 0"
    viewer.viewer3d.show_plane_checkbox.value = True
    viewer.background.value = "#123456"
    snapshot = viewer.view_settings()
    viewer.viewer3d.viewer._camera_orientation = np.eye(4).ravel().tolist()
    viewer.slice2d.offset_txt.value = "0"
    viewer.slice2d.normal_txt.value = "0 0 1"
    viewer.viewer3d.show_plane_checkbox.value = False
    viewer.background.value = "#FFFFFF"
    viewer.viewer3d.isovalues.isovalues.children[0].value = 0.01
    viewer.apply_view_settings(snapshot)
    assert viewer.view_settings() == snapshot
    assert viewer.tabs.selected_index == 0
    assert (
        viewer.slice2d.plane_controls
        in viewer.viewer3d.children[0].children[1].children
    )
    assert viewer.slice2d.children == (viewer.slice2d.plot_controls,)
    assert viewer.viewer3d._plane_component is not None
    viewer.close()


def test_named_view_collection_keeps_multiple_views_and_reopens():
    from aiidalab_widgets_empa.cube_views import SavedCubeViews

    records, loaded, submitted = {}, [], []

    def save(name, identifier):
        identifier = identifier or str(len(records) + 1)
        records[identifier] = dict(id=identifier, name=name, label=name)
        return identifier

    def make_widget():
        return SavedCubeViews(
            save=save,
            list_views=lambda: list(records.values()),
            load=loaded.append,
            remove=records.pop,
            render=submitted.append,
        )

    widget = make_widget()
    widget.save_button.click()
    assert "name" in widget.message.value and not records
    for name in ("Top", "Side"):
        widget.name.value = name
        widget.save_button.click()
    assert len(records) == 2 and not submitted
    widget.close()
    widget = make_widget()
    assert len(widget.saved.options) == 2
    widget.saved.value = ("1",)
    widget.load_button.click()
    assert loaded == ["1"]
    widget.name.value = "New top"
    widget.update_button.click()
    assert records["1"]["name"] == "New top" and len(records) == 2
    widget.saved.value = ("1", "2")
    assert widget.load_button.disabled
    widget.render_button.click()
    assert submitted == [["1", "2"]]
    widget.remove_button.click()
    assert not records
    widget.refresh_button.click()
    assert not widget.saved.options and widget.render_button.disabled
    widget.close()
