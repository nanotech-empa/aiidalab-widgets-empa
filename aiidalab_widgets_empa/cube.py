"""Shared cube viewer extracted from aiidalab-empa-surfaces.

Based on surfaces_tools/widgets/cube.py at 452b8ed178b7070fc616cf65bc62e9e0817410a6.
Copyright (c) 2017 Empa and NCCR MARVEL Switzerland. MIT licensed;
see CUBE_VIEWER_LICENSE.txt. This module has no AiiDA/CP2K dependency.
"""

import io
import logging
import re
import tempfile

import ase.io.cube
import ipywidgets as ipw
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nglview
import numpy as np
import traitlets as tl
from cubehandler import Cube
from cubehandler.cube import ANG_TO_BOHR
from scipy.ndimage import map_coordinates
from .cube_views import render_plane_settings

logger = logging.getLogger(__name__)
ERR_EXPECT_3_FLOATS = "Expected 3 finite floats"


def _orthonormal_basis_from_normal(normal):
    normal = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(normal)
    if norm < 1.0e-12:
        n_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        n_hat = normal / norm

    trial = (
        np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(n_hat[0]) < 0.9
        else np.array([0.0, 1.0, 0.0], dtype=float)
    )
    u_hat = np.cross(n_hat, trial)
    u_hat /= np.linalg.norm(u_hat)
    v_hat = np.cross(n_hat, u_hat)
    v_hat /= np.linalg.norm(v_hat)
    return n_hat, u_hat, v_hat


def make_plane_mesh(
    center, normal, width, height, offset=0.0, colors_2d=None, max_grid_n=64
):
    n_hat, u_hat, v_hat = _orthonormal_basis_from_normal(normal)
    center = np.asarray(center, dtype=float) + float(offset) * n_hat
    width = float(width)
    height = float(height)

    if colors_2d is None:
        grid_n = 50
        colors = None
    else:
        grid_n = min(int(max_grid_n), colors_2d.shape[0] - 1)
        color_i = np.linspace(0, colors_2d.shape[0] - 1, grid_n, dtype=int)
        color_j = np.linspace(0, colors_2d.shape[1] - 1, grid_n, dtype=int)
        colors = colors_2d[np.ix_(color_i, color_j)]

    us = np.linspace(-width / 2, width / 2, grid_n + 1)
    vs = np.linspace(-height / 2, height / 2, grid_n + 1)
    vertices = []
    mesh_colors = []

    for i in range(grid_n):
        for j in range(grid_n):
            u0, u1 = us[i], us[i + 1]
            v0, v1 = vs[j], vs[j + 1]
            p00 = center + u0 * u_hat + v0 * v_hat
            p10 = center + u1 * u_hat + v0 * v_hat
            p11 = center + u1 * u_hat + v1 * v_hat
            p01 = center + u0 * u_hat + v1 * v_hat
            color = (0.6, 0.6, 0.6) if colors is None else colors[j, i]
            if not np.isfinite(color).all():
                continue

            for point in (p00, p10, p11, p00, p11, p01):
                vertices.extend(point.tolist())
                mesh_colors.extend(color)

    return vertices, mesh_colors


class OneIsovalue(ipw.HBox):
    def __init__(self, structure=None):
        self.isovalue_widget = ipw.BoundedFloatText(
            value=1e-3,
            min=-1e-1,
            max=1e-1,
            step=1e-5,
            description="Isovalue",
        )
        self.color_widget = ipw.ColorPicker(
            concise=False, description="Pick a color", value="#00FFFF", disabled=False
        )
        super().__init__([self.isovalue_widget, self.color_widget])

    # proxy: così min/max/value su OneIsovalue agiscono sul BoundedFloatText
    @property
    def isomin(self):
        return self.isovalue_widget.min

    @isomin.setter
    def isomin(self, v):
        self.isovalue_widget.min = v

    @property
    def isomax(self):
        return self.isovalue_widget.max

    @isomax.setter
    def isomax(self, v):
        self.isovalue_widget.max = v

    @property
    def value(self):
        return self.isovalue_widget.value

    @value.setter
    def value(self, v):
        self.isovalue_widget.value = v


class IsovaluesWidget(ipw.VBox):
    iso_min = tl.Float(1.0e-5, allow_none=True)
    iso_max = tl.Float(1.0e-1, allow_none=True)

    def __init__(self):
        self.isovalues = ipw.VBox()

        # Add isovalue button.
        self.info = ipw.HTML(f"min: {self.iso_min:.1e}, max: {self.iso_max:.1e}")
        self.add_isovalue_button = ipw.Button(
            description="Add isovalue",
            layout={"width": "initial"},
            button_style="success",
        )
        self.add_isovalue_button.on_click(self.add_isovalue)

        # Remove isovalue button.
        self.remove_isovalue_button = ipw.Button(
            description="Remove isovalue",
            layout={"width": "initial"},
            button_style="danger",
        )
        self.remove_isovalue_button.on_click(self.remove_isovalue)

        super().__init__(
            [
                self.isovalues,
                ipw.HBox(
                    [
                        self.info,
                        self.add_isovalue_button,
                        self.remove_isovalue_button,
                    ]
                ),
            ]
        )

    def return_isovalues_and_colors(self):
        return zip(
            *[
                (child.children[0].value, child.children[1].value)
                for child in self.isovalues.children
            ]
        )

    def set_range(self, vmin=-0.001, vmax=0.001):
        default = min(0.001, vmax)
        self.info.value = f"min: {vmin:.1e}, max: {vmax:.1e}"
        with self.hold_trait_notifications():
            self.iso_min = vmin
            self.iso_max = vmax
        if abs(vmin) > abs(vmax):
            default = max(-0.001, vmin)

        for isovalue in self.isovalues.children:
            field = isovalue.children[0]
            field.min = min(field.min, vmin)
            field.max = max(field.max, vmax)
            field.min, field.max = vmin, vmax
            if not (vmin <= isovalue.children[0].value <= vmax):
                isovalue.children[0].value = default

    @tl.observe("iso_min", "iso_max")
    def _observe_range(self, _=None):
        self.set_range(vmin=self.iso_min, vmax=self.iso_max)

    def add_isovalue(self, _=None):
        new = OneIsovalue()
        new.isomin = min(new.isomin, self.iso_min)
        new.isomax = max(new.isomax, self.iso_max)
        new.isomin, new.isomax = self.iso_min, self.iso_max
        new.value = (
            min(0.001, self.iso_max)
            if self.iso_max >= 1.0e-8
            else max(-0.001, self.iso_min)
        )
        self.isovalues.children += (new,)

    def remove_isovalue(self, _=None):
        self.isovalues.children = self.isovalues.children[:-1]

    def traits_to_link(self):
        return ["cube"]


class CubePlaneCut2D(ipw.VBox):
    """2D plane-cut viewer for cube data."""

    cube = tl.Instance(Cube, allow_none=True)

    def __init__(self, *, periodic=False):
        self.periodic = periodic
        self._last_r = None
        self._last_vals = None
        self._last_rgb = None
        self._sinv = None
        self._updating_color_range = False

        self.center_txt = ipw.Text(
            value="0.0 0.0 0.0",
            description="Center (A)",
            placeholder="x y z",
            layout=ipw.Layout(width="420px"),
            style={"description_width": "110px"},
        )
        self.normal_txt = ipw.Text(
            value="0 0 1",
            description="Normal",
            placeholder="nx ny nz",
            layout=ipw.Layout(width="420px"),
            style={"description_width": "110px"},
        )
        self.offset_txt = ipw.Text(
            value="0.0",
            description="Offset (A)",
            layout=ipw.Layout(width="220px"),
            style={"description_width": "110px"},
        )
        self.width = ipw.FloatText(
            value=10.0,
            description="Width (A)",
            layout=ipw.Layout(width="220px"),
            style={"description_width": "110px"},
        )
        self.height = ipw.FloatText(
            value=10.0,
            description="Height (A)",
            layout=ipw.Layout(width="220px"),
            style={"description_width": "110px"},
        )
        self.res = ipw.IntSlider(
            value=64,
            min=32,
            max=256,
            step=32,
            description="2D slice res.",
            continuous_update=False,
            layout=ipw.Layout(width="360px"),
            style={"description_width": "90px"},
        )
        self.show_contours = ipw.Checkbox(value=False, description="2D contours")
        self.auto_color_range = ipw.Checkbox(
            value=True,
            description="Auto color range",
        )
        self.auto_color_range.observe(self._on_auto_color_range_toggled, names="value")
        self.center_color_range = ipw.Checkbox(
            value=True, description="Centre automatic scale on zero"
        )
        self.vmin = ipw.FloatText(
            value=0.0,
            description="vmin",
            disabled=True,
            layout=ipw.Layout(width="150px"),
            style={"description_width": "60px"},
        )
        self.vmax = ipw.FloatText(
            value=0.0,
            description="vmax",
            disabled=True,
            layout=ipw.Layout(width="150px"),
            style={"description_width": "60px"},
        )
        self.show_2d_checkbox = ipw.Checkbox(value=True, description="Show 2D plot")
        self.show_2d_checkbox.observe(self._on_show_2d_toggled, names="value")
        self.plane_preset = ipw.Dropdown(
            options=[
                ("XY", "0 0 1"),
                ("XZ", "0 1 0"),
                ("YZ", "1 0 0"),
            ],
            description="Plane:",
            layout=ipw.Layout(width="180px"),
            style={"description_width": "55px"},
        )
        self.apply_plane_preset = ipw.Button(
            description="Apply",
            layout=ipw.Layout(width="80px"),
        )
        self.apply_plane_preset.on_click(self._on_apply_plane_preset)
        self.center_on_atoms = ipw.Button(
            description="Center on atoms",
            layout=ipw.Layout(width="140px"),
        )
        self.center_on_atoms.on_click(self._on_center_on_atoms)
        self.info = ipw.HTML("")
        self.error = ipw.HTML("")
        self._out = ipw.Output(layout=ipw.Layout(width="100%"))
        self.image = ipw.Image(format="png")
        self.slice_png = b""
        self.color_legend = ipw.HTML()
        self.color_controls = ipw.VBox(
            [
                self.auto_color_range,
                self.center_color_range,
                ipw.HBox([self.vmin, self.vmax]),
                self.color_legend,
            ]
        )

        self.plane_controls = ipw.VBox(
            [
                ipw.HTML(
                    "<b>2D plane cut</b> · "
                    + (
                        "periodic boundaries"
                        if self.periodic
                        else "isolated grid; outside areas are empty"
                    )
                ),
                ipw.HBox([self.center_txt]),
                ipw.HBox([self.normal_txt]),
                ipw.HBox(
                    [
                        self.plane_preset,
                        self.apply_plane_preset,
                        self.center_on_atoms,
                    ]
                ),
                ipw.HBox([self.offset_txt]),
                ipw.HBox([self.width, self.height]),
                self.res,
                self.color_controls,
                self.error,
            ]
        )
        self.plot_controls = ipw.VBox(
            [
                self.color_controls,
                self.show_contours,
                self.show_2d_checkbox,
                self.info,
                self.image,
                self._out,
            ]
        )
        super().__init__([self.plane_controls, self.plot_controls])

        for widget in (
            self.center_txt,
            self.normal_txt,
            self.offset_txt,
            self.width,
            self.height,
            self.res,
            self.show_contours,
            self.auto_color_range,
            self.center_color_range,
            self.vmin,
            self.vmax,
        ):
            widget.observe(self._on_any_change, names="value")

    def _vec3_is_valid(self, text):
        tokens = [token for token in re.split(r"[,\s]+", text.strip()) if token]
        if len(tokens) != 3:
            return False
        try:
            [float(token) for token in tokens]
        except ValueError:
            return False
        return True

    def _inputs_valid(self):
        if self.cube is None:
            return False
        if not self._vec3_is_valid(self.center_txt.value):
            return False
        if not self._vec3_is_valid(self.normal_txt.value):
            return False
        try:
            float(self.offset_txt.value)
            float(self.width.value)
            float(self.height.value)
            int(self.res.value)
        except (TypeError, ValueError):
            return False
        return True

    def _on_any_change(self, _=None):
        if self._updating_color_range:
            return
        if self._inputs_valid():
            self.error.value = ""
            self.plot_slice()
        else:
            self.error.value = "<span style='color:#b00'>Enter a valid plane before displaying a slice.</span>"
            self.image.value = b""
            self.slice_png = b""
            with self._out:
                self._out.clear_output(wait=True)

    def _on_show_2d_toggled(self, change):
        if change["new"]:
            self._out.layout.display = ""
            if self._inputs_valid():
                self.plot_slice()
        else:
            self._out.clear_output()
            self._out.layout.display = "none"
            self.image.layout.display = "none"
            if self._inputs_valid():
                self.plot_slice()

    def _on_apply_plane_preset(self, _=None):
        self.normal_txt.value = self.plane_preset.value

    def _on_center_on_atoms(self, _=None):
        if self.cube is None:
            return
        center = np.mean(self.cube.ase_atoms.get_positions(), axis=0)
        self.center_txt.value = f"{center[0]:.3f} {center[1]:.3f} {center[2]:.3f}"

    def _on_auto_color_range_toggled(self, change):
        use_auto = change["new"]
        self.vmin.disabled = use_auto
        self.vmax.disabled = use_auto
        self.center_color_range.disabled = not use_auto
        if self._inputs_valid():
            self.plot_slice()

    @staticmethod
    def _parse_vec3(text):
        values = np.fromstring(text.replace(",", " "), sep=" ", dtype=float)
        if values.size != 3 or not np.isfinite(values).all():
            raise ValueError(ERR_EXPECT_3_FLOATS)
        return values

    @tl.observe("cube")
    def _on_cube(self, _=None):
        if self.cube is None:
            self._sinv = None
            self.info.value = ""
            self.image.value = b""
            self.slice_png = b""
            with self._out:
                self._out.clear_output()
            return

        cell = np.array(self.cube.ase_atoms.cell).T
        self._sinv = np.linalg.inv(cell)
        center = np.mean(self.cube.ase_atoms.get_positions(), axis=0)
        self.center_txt.value = f"{center[0]:.3f} {center[1]:.3f} {center[2]:.3f}"

        nx, ny, nz = self.cube.data.shape
        dmin = float(np.min(self.cube.data))
        dmax = float(np.max(self.cube.data))
        lengths = np.linalg.norm(cell, axis=0)
        self.info.value = (
            f"Grid: {nx} x {ny} x {nz} | data in [{dmin:.3e}, {dmax:.3e}] "
            f"| cell lengths: {lengths[0]:.2f}, {lengths[1]:.2f}, "
            f"{lengths[2]:.2f} A"
        )
        mean_len = float(np.mean(lengths))
        self.width.value = mean_len
        self.height.value = mean_len
        self.plot_slice()

    def _build_plane_grid(self):
        center = self._parse_vec3(self.center_txt.value)
        normal = self._parse_vec3(self.normal_txt.value)
        n_hat, u_hat, v_hat = _orthonormal_basis_from_normal(normal)

        offset = float(self.offset_txt.value)
        center_shifted = center + offset * n_hat
        width = float(self.width.value)
        height = float(self.height.value)
        resolution = int(self.res.value)

        us = np.linspace(-width / 2, width / 2, resolution)
        vs = np.linspace(-height / 2, height / 2, resolution)
        uu, vv = np.meshgrid(us, vs)
        grid = (
            center_shifted[None, None, :]
            + uu[..., None] * u_hat
            + vv[..., None] * v_hat
        )
        extent = [-width / 2, width / 2, -height / 2, height / 2]
        return grid, extent

    def _cart_to_indexcoords(self, grid):
        nx, ny, nz = self.cube.data.shape
        positions = grid.reshape(-1, 3) - self.cube.origin / ANG_TO_BOHR
        fractional = (self._sinv @ positions.T).T
        indexes = np.empty_like(fractional)
        indexes[:, 0] = fractional[:, 0] * nx
        indexes[:, 1] = fractional[:, 1] * ny
        indexes[:, 2] = fractional[:, 2] * nz
        return indexes.reshape(grid.shape)

    def _compute_slice_values(self):
        grid, extent = self._build_plane_grid()
        index_coords = self._cart_to_indexcoords(grid)
        coords = np.stack(
            [index_coords[..., 0], index_coords[..., 1], index_coords[..., 2]], axis=0
        )
        values = map_coordinates(
            self.cube.data,
            coords,
            order=1,
            mode="wrap" if self.periodic else "constant",
            cval=np.nan,
            prefilter=False,
        )
        return grid, extent, values

    def _effective_color_range(self, values):
        if self.auto_color_range.value:
            finite = values[np.isfinite(values)]
            if self.center_color_range.value:
                limit = float(np.max(np.abs(finite))) if finite.size else 0.0
                limit = limit or 1.0
                return -limit, limit
            return (
                (float(finite.min()), float(finite.max()))
                if finite.size
                else (0.0, 1.0)
            )
        return float(self.vmin.value), float(self.vmax.value)

    def _store_slice_cache(self, grid, values, vmin, vmax):
        normalizer = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = mpl.colormaps["coolwarm"]
        self._last_r = grid
        self._last_vals = values
        self._last_rgb = cmap(normalizer(values))[..., :3]
        self._last_rgb[~np.isfinite(values)] = np.nan
        stops = ",".join(mcolors.to_hex(cmap(x)) for x in np.linspace(0, 1, 11))
        self.color_legend.value = (
            f'<div style="width:300px;height:12px;background:linear-gradient(to right,{stops})"></div>'
            f'<div style="width:300px;display:flex;justify-content:space-between">'
            f"<span>{vmin:.3g}</span><span>{(vmin + vmax) / 2:.3g}</span><span>{vmax:.3g}</span></div>"
            "<small>Cube values (a.u.). Plane colours are unaffected by lighting.</small>"
        )

    def plot_slice(self):
        if self.cube is None:
            return

        try:
            grid, extent, values = self._compute_slice_values()
            vmin, vmax = self._effective_color_range(values)
        except Exception:
            logger.debug("cube plane slice failed", exc_info=True)
            self.error.value = "<span style='color:#b00'>Error computing slice</span>"
            self.image.value = b""
            self.slice_png = b""
            with self._out:
                self._out.clear_output()
            return

        if self.auto_color_range.value:
            self._updating_color_range = True
            try:
                self.vmin.value = vmin
                self.vmax.value = vmax
            finally:
                self._updating_color_range = False

        self._store_slice_cache(grid, values, vmin, vmax)
        if not self.show_2d_checkbox.value:
            self._out.layout.display = "none"
            self.image.layout.display = "none"
            return

        self._out.layout.display = ""
        self.image.layout.display = ""
        with self._out:
            self._out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            image = ax.imshow(
                values,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("u (A)")
            ax.set_ylabel("v (A)")
            fig.colorbar(image, ax=ax)

            if self.show_contours.value and not np.isclose(vmin, vmax):
                contour_levels = np.linspace(vmin, vmax, 9)[1:-1]
                ax.contour(
                    values,
                    levels=contour_levels,
                    origin="lower",
                    extent=extent,
                    colors="black",
                    linewidths=1.2,
                    alpha=0.65,
                )
                ax.contour(
                    values,
                    levels=contour_levels,
                    origin="lower",
                    extent=extent,
                    colors="white",
                    linewidths=0.55,
                    alpha=0.95,
                )

            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            self.slice_png = buffer.getvalue()
            self.image.value = self.slice_png
            plt.close(fig)

    def current_plane_params(self):
        center = self._parse_vec3(self.center_txt.value)
        normal = self._parse_vec3(self.normal_txt.value)
        n_hat, _, _ = _orthonormal_basis_from_normal(normal)
        return (
            center,
            n_hat,
            float(self.offset_txt.value),
            float(self.width.value),
            float(self.height.value),
        )

    def return_settings(self):
        center, normal, offset, width, height = self.current_plane_params()
        return {
            "periodic": self.periodic,
            "center": center.tolist(),
            "normal": normal.tolist(),
            "offset": offset,
            "width": width,
            "height": height,
            "resolution": int(self.res.value),
            "show_2d": bool(self.show_2d_checkbox.value),
            "show_contours": bool(self.show_contours.value),
            "auto_color_range": bool(self.auto_color_range.value),
            "center_color_range": bool(self.center_color_range.value),
            "vmin": float(self.vmin.value),
            "vmax": float(self.vmax.value),
        }


class CubeArrayData3dViewerWidget(ipw.VBox):
    """Widget to view cube data in 3D and overlay a slicing plane."""

    cube = tl.Instance(Cube, allow_none=True)

    def __init__(self, **kwargs):
        self.structure = None
        self.viewer = nglview.NGLWidget()
        self.isovalues = IsovaluesWidget()
        self._loaded_cube = None
        self._structure_component = None
        self._cube_component = None
        self._plane_component = None
        self.slice2d = None
        self._external_plane_sync_cb = None
        self.lighting = dict(ambient=0.45, direct=0.55, gloss=0.15)
        self._apply_stage_lighting()

        self.show_isosurfaes_button = ipw.Button(
            description="Update isosurfaces",
            layout={"width": "initial"},
            button_style="success",
        )
        self.show_isosurfaes_button.on_click(lambda _: self.apply_isosurfaces())
        self.show_plane_checkbox = ipw.Checkbox(
            value=True, description="Show slicing plane in 3D"
        )
        self.show_plane_checkbox.observe(self._on_plane_toggle, names="value")
        self.plane_resolution = ipw.BoundedIntText(
            value=48,
            min=16,
            max=128,
            step=16,
            description="3D plane res.:",
            layout=ipw.Layout(width="180px"),
            style={"description_width": "100px"},
        )

        super().__init__(
            [
                self.viewer,
                self.isovalues,
                ipw.HBox(
                    [
                        self.show_isosurfaes_button,
                        self.show_plane_checkbox,
                        self.plane_resolution,
                    ]
                ),
            ],
            **kwargs,
        )

    @tl.observe("cube")
    def on_observe_cube(self, _=None):
        """Update object attributes when cube trait is modified."""
        if self.cube is None:
            self.structure = None
            self._loaded_cube = None
            self._remove_component("_structure_component")
            self._remove_component("_cube_component")
            self.hide_and_remove_plane()
            return

        self.isovalues.set_range(
            vmin=float(np.min(self.cube.data)), vmax=float(np.max(self.cube.data))
        )
        if not self.isovalues.isovalues.children:
            for sign, color in ((1, "#2563B7"), (-1, "#E3AE3B")):
                bound = self.isovalues.iso_max if sign > 0 else -self.isovalues.iso_min
                value = sign * min(0.03, max(0.0, bound) / 3)
                if (
                    bound > 0
                    and self.isovalues.iso_min <= value <= self.isovalues.iso_max
                ):
                    self.isovalues.add_isovalue()
                    row = self.isovalues.isovalues.children[-1]
                    row.value = value
                    row.color_widget.value = color
        self.structure = self.cube.ase_atoms
        self.update_plot()

    def _remove_component(self, attr):
        component = getattr(self, attr, None)
        if component is None:
            return
        try:
            self.viewer.remove_component(component.id)
        except Exception:
            logger.debug("remove_component failed", exc_info=True)
        setattr(self, attr, None)

    def update_plot(self):
        """Update the 3D plot."""
        if self._loaded_cube is self.cube and self._cube_component is not None:
            self.apply_isosurfaces()
            if self._external_plane_sync_cb:
                self._external_plane_sync_cb()
            return

        self._remove_component("_structure_component")
        self._remove_component("_cube_component")
        if self.structure is None or self.cube is None:
            return

        self._structure_component = self.viewer.add_component(
            nglview.ASEStructure(self.structure)
        )
        self._structure_component.clear()
        self._structure_component.add_ball_and_stick(
            roughness=1 - self.lighting["gloss"]
        )
        with tempfile.NamedTemporaryFile(mode="w") as tempf:
            ase.io.cube.write_cube(
                tempf,
                self.structure,
                self.cube.data,
                origin=self.cube.origin / ANG_TO_BOHR,
            )
            self._cube_component = self.viewer.add_component(tempf.name, ext="cube")
            self._cube_component.clear()
        self._loaded_cube = self.cube
        self.apply_isosurfaces()

        if self._external_plane_sync_cb:
            self._external_plane_sync_cb()

    def apply_isosurfaces(self):
        try:
            isovalues, colors = self.isovalues.return_isovalues_and_colors()
            self.set_cube_isosurf(isovalues, colors)
        except ValueError:
            pass
        except Exception:
            logger.debug("set_cube_isosurf failed", exc_info=True)

    def set_cube_isosurf(self, isovals, colors):
        """Set cube isosurfaces."""
        if self._cube_component is None:
            return
        self._cube_component.clear()
        for isov, col in zip(isovals, colors):
            try:
                self._cube_component.add_surface(
                    color=col,
                    isolevelType="value",
                    isolevel=isov,
                    roughness=1 - self.lighting["gloss"],
                    metalness=0,
                )
            except Exception:
                logger.debug("add_surface failed", exc_info=True)

    def _apply_stage_lighting(self):
        self.viewer.parameters = dict(
            ambientIntensity=self.lighting["ambient"],
            lightIntensity=self.lighting["direct"],
        )

    def set_lighting(self, settings):
        self.lighting = dict(settings)
        self._apply_stage_lighting()
        self.apply_isosurfaces()
        if self._structure_component is not None:
            self._structure_component.update_ball_and_stick(
                roughness=1 - settings["gloss"], metalness=0
            )

    def hide_and_remove_plane(self):
        if self._plane_component is None:
            return
        try:
            self.viewer.remove_component(self._plane_component.id)
        except Exception:
            logger.debug("remove plane component failed", exc_info=True)
        self._plane_component = None

    def _on_plane_toggle(self, change):
        if self._external_plane_sync_cb and change["new"]:
            self._external_plane_sync_cb()
            return
        if self._plane_component is None:
            return
        try:
            if change["new"]:
                self._plane_component.show()
            else:
                self._plane_component.hide()
        except Exception:
            logger.debug("plane show/hide failed", exc_info=True)

    def update_plane_mesh(self, *, center, normal, width, height, offset=0.0):
        if not self.show_plane_checkbox.value:
            self.hide_and_remove_plane()
            return

        self.hide_and_remove_plane()
        colors_2d = getattr(self.slice2d, "_last_rgb", None)

        try:
            vertices, colors = make_plane_mesh(
                center,
                normal,
                width,
                height,
                offset,
                colors_2d=colors_2d,
                max_grid_n=self.plane_resolution.value,
            )
            self._plane_component = self.viewer.shape.add_mesh(vertices, colors)
            index = self.viewer._ngl_component_ids.index(self._plane_component.id)
            # NGL's background buffer flag selects its NOLIGHT shader. Set it
            # AFTER attachment so the plane stays in the depth-tested model
            # group, rather than becoming a backdrop behind all the atoms.
            self.viewer._execute_js_code(
                f"this.stage.compList[{index}].reprList.forEach(item => "
                "item.repr.bufferList.forEach(buffer => "
                "buffer.setParameters({background: true})));"
                "this.stage.viewer.requestRender();"
            )
        except Exception:
            logger.debug("plane mesh update failed", exc_info=True)
            return


class CubeViewer(ipw.VBox):
    """Reusable, data-only 3D/2D viewer; applications submit render CalcJobs.

    ``on_render`` receives only serializable camera/isosurface settings. It must
    schedule rendering outside the notebook. No ORM objects are retained here.
    """

    def __init__(self, *, periodic=False, on_render=None, **kwargs):
        self.viewer3d = CubeArrayData3dViewerWidget()
        self.slice2d = CubePlaneCut2D(periodic=periodic)
        self.viewer3d.slice2d = self.slice2d
        self._cube_link = tl.dlink((self.viewer3d, "cube"), (self.slice2d, "cube"))
        self.viewer3d._external_plane_sync_cb = self._sync_plane
        for field in (
            self.slice2d.center_txt,
            self.slice2d.normal_txt,
            self.slice2d.offset_txt,
            self.slice2d.width,
            self.slice2d.height,
            self.slice2d.res,
            self.slice2d.auto_color_range,
            self.slice2d.center_color_range,
            self.slice2d.vmin,
            self.slice2d.vmax,
            self.viewer3d.plane_resolution,
        ):
            field.observe(self._sync_plane, names="value")
        self.viewer3d.viewer.layout = ipw.Layout(width="100%", height="450px")
        # NGL initially sizes its canvas from its parent's width. Give it a
        # dedicated flex column rather than the full viewer/controls row.
        canvas_column = ipw.VBox(
            [self.viewer3d.viewer],
            layout=ipw.Layout(min_width="320px", flex="1 1 480px"),
        )
        self.viewer3d.show_plane_checkbox.value = False
        plane_controls = ipw.VBox(
            [
                self.viewer3d.show_plane_checkbox,
                self.slice2d.plane_controls,
                self.viewer3d.plane_resolution,
            ],
            layout=ipw.Layout(width="440px"),
        )
        self.lighting_preset = ipw.Dropdown(
            options=[
                ("Soft (recommended)", "soft"),
                ("Matte", "matte"),
                ("Manual", "manual"),
            ],
            description="Lighting",
        )
        self.light_ambient = ipw.FloatSlider(
            value=0.45,
            min=0,
            max=1,
            step=0.05,
            description="Ambient",
            continuous_update=False,
        )
        self.light_direct = ipw.FloatSlider(
            value=0.55,
            min=0,
            max=1,
            step=0.05,
            description="Direct light",
            continuous_update=False,
        )
        self.light_gloss = ipw.FloatSlider(
            value=0.15,
            min=0,
            max=1,
            step=0.05,
            description="Gloss",
            continuous_update=False,
        )
        self._updating_lighting = False
        self.lighting_preset.observe(self._on_lighting_preset, names="value")
        for field in (self.light_ambient, self.light_direct, self.light_gloss):
            field.observe(self._on_lighting_changed, names="value")
        lighting_controls = ipw.Accordion(
            children=[
                ipw.VBox(
                    [
                        self.lighting_preset,
                        self.light_ambient,
                        self.light_direct,
                        self.light_gloss,
                        ipw.HTML(
                            "<small>Lighting affects atoms and isosurfaces. The data plane keeps its colour scale.</small>"
                        ),
                    ]
                )
            ],
            selected_index=None,
        )
        lighting_controls.set_title(0, "Lighting")
        plane_controls.children += (lighting_controls,)
        self.viewer3d.children = (
            ipw.HBox(
                [canvas_column, plane_controls],
                layout=ipw.Layout(flex_flow="row wrap", align_items="flex-start"),
            ),
            self.viewer3d.isovalues,
            self.viewer3d.show_isosurfaes_button,
        )
        self.slice2d.children = (self.slice2d.plot_controls,)
        self.width = ipw.BoundedIntText(
            value=1600, min=320, max=4096, description="Width (px)"
        )
        self.height = ipw.BoundedIntText(
            value=1200, min=320, max=4096, description="Height (px)"
        )
        self.background = ipw.ColorPicker(value="#FFFFFF", description="Background")
        self.opacity = ipw.FloatSlider(
            value=0.65,
            min=0.05,
            max=1,
            step=0.05,
            description="Opacity",
            continuous_update=False,
        )
        self.render_button = ipw.Button(
            description="Render image",
            icon="image",
            button_style="primary",
            disabled=on_render is None,
        )
        self.render_message = ipw.HTML()
        self._render_callback = on_render
        self.render_button.on_click(self._render)
        self.background.observe(
            lambda change: setattr(self.viewer3d.viewer, "background", change["new"]),
            names="value",
        )
        self.slice_download = ipw.HTML()
        export_slice = ipw.Button(description="Export 2D slice", icon="download")
        export_slice.on_click(self._export_slice)
        self.tabs = ipw.Tab(
            children=[
                self.viewer3d,
                ipw.VBox([self.slice2d, export_slice, self.slice_download]),
                ipw.VBox(
                    [
                        ipw.HTML(
                            "<p>Render the selected cube, current camera, isosurfaces and visible slicing plane through the application. The flat 2D plot can also be exported separately.</p>"
                        ),
                        ipw.HBox([self.width, self.height]),
                        self.background,
                        self.opacity,
                        self.render_button,
                        self.render_message,
                    ]
                ),
            ]
        )
        for i, title in enumerate(("3D isosurfaces", "2D slice", "Render")):
            self.tabs.set_title(i, title)
        self.tabs.observe(self._resize, names="selected_index")
        super().__init__([self.tabs], **kwargs)

    def _on_lighting_preset(self, change):
        if self._updating_lighting or change["new"] == "manual":
            return
        self._set_lighting(
            dict(ambient=0.45, direct=0.55, gloss=0.15)
            if change["new"] == "soft"
            else dict(ambient=0.6, direct=0.4, gloss=0.0)
        )

    def _set_lighting(self, settings):
        self._updating_lighting = True
        try:
            for key, value in settings.items():
                getattr(self, "light_" + key).value = value
            self.lighting_preset.value = (
                "soft"
                if settings == dict(ambient=0.45, direct=0.55, gloss=0.15)
                else "matte"
                if settings == dict(ambient=0.6, direct=0.4, gloss=0.0)
                else "manual"
            )
        finally:
            self._updating_lighting = False
        self.viewer3d.set_lighting(settings)

    def _on_lighting_changed(self, _):
        if not self._updating_lighting:
            self._set_lighting(
                {
                    key: getattr(self, "light_" + key).value
                    for key in ("ambient", "direct", "gloss")
                }
            )

    def set_cube(self, cube):
        self.slice_download.value = ""
        self.viewer3d.cube = cube
        self.viewer3d.viewer.center()
        self._sync_plane()

    def set_content(self, content):
        self.set_cube(Cube.from_content(content))

    def _resize(self, change):
        if change["new"] == 0:
            self.viewer3d.viewer._remote_call("handleResize", target="Widget")

    def _sync_plane(self, _=None):
        if self.slice2d._updating_color_range:
            return
        if not self.slice2d._inputs_valid():
            self.viewer3d.hide_and_remove_plane()
            return
        center, normal, offset, width, height = self.slice2d.current_plane_params()
        self.viewer3d.update_plane_mesh(
            center=center, normal=normal, offset=offset, width=width, height=height
        )

    def render_settings(self):
        orientation = list(self.viewer3d.viewer._camera_orientation)
        if len(orientation) != 16 or not np.isfinite(orientation).all():
            raise ValueError("Open the 3D view and set the camera before rendering.")
        pairs = [
            [float(row.value), row.color_widget.value]
            for row in self.viewer3d.isovalues.isovalues.children
        ]
        if not pairs:
            raise ValueError("Add at least one isosurface before rendering.")
        settings = dict(
            orientation=orientation,
            iso=pairs,
            width=self.width.value,
            height=self.height.value,
            background=self.background.value,
            opacity=self.opacity.value,
            lighting=dict(self.viewer3d.lighting),
        )
        if self.viewer3d.show_plane_checkbox.value:
            settings["slice_plane"] = render_plane_settings(
                self.slice2d.return_settings()
            )
        return settings

    def view_settings(self):
        """Snapshot the view as plain data, including the linked slice plane."""
        return {
            "version": 1,
            "render": self.render_settings(),
            "plane": self.slice2d.return_settings(),
            "show_plane": self.viewer3d.show_plane_checkbox.value,
            "plane_resolution": self.viewer3d.plane_resolution.value,
        }

    def apply_view_settings(self, state):
        """Restore a saved view after its source cube has been loaded."""
        if state.get("version") != 1:
            raise ValueError("Unsupported visualization version")
        plane, render = state["plane"], state["render"]
        if plane["periodic"] != self.slice2d.periodic:
            raise ValueError("The saved view uses different boundary conditions")
        rows = self.viewer3d.isovalues
        if any(not rows.iso_min <= value <= rows.iso_max for value, _ in render["iso"]):
            raise ValueError("The saved isovalues are outside this cube's range")
        rows.isovalues.children = ()
        for value, color in render["iso"]:
            rows.add_isovalue()
            row = rows.isovalues.children[-1]
            row.value, row.color_widget.value = value, color
        self.viewer3d.apply_isosurfaces()
        for key in ("width", "height", "background", "opacity"):
            getattr(self, key).value = render[key]
        self._set_lighting(
            render.get("lighting", dict(ambient=0.45, direct=0.55, gloss=0.15))
        )
        cut = self.slice2d
        # Suppress intermediate slices while restoring the complete plane.
        cut._updating_color_range = True
        try:
            for key in ("center", "normal"):
                getattr(cut, key + "_txt").value = " ".join(map(str, plane[key]))
            cut.offset_txt.value = str(plane["offset"])
            # Older snapshots used asymmetric min/max autoscaling. Keep that
            # appearance until the user explicitly selects the centred scale.
            cut.center_color_range.value = plane.get("center_color_range", False)
            for attr, key in (
                ("width", "width"),
                ("height", "height"),
                ("res", "resolution"),
                ("show_2d_checkbox", "show_2d"),
                ("show_contours", "show_contours"),
                ("auto_color_range", "auto_color_range"),
                ("vmin", "vmin"),
                ("vmax", "vmax"),
            ):
                getattr(cut, attr).value = plane[key]
        finally:
            cut._updating_color_range = False
        cut._on_any_change()
        self.viewer3d.plane_resolution.value = state["plane_resolution"]
        self.viewer3d.show_plane_checkbox.value = state["show_plane"]
        self._sync_plane()
        orientation = list(render["orientation"])
        self.viewer3d.viewer._camera_orientation = orientation
        self.viewer3d.viewer._set_camera_orientation(orientation)
        self.tabs.selected_index = 0

    def _render(self, _=None):
        import html

        try:
            self.render_message.value = ""
            self._render_callback(self.render_settings())
        except Exception as exc:
            self.render_message.value = (
                '<p style="color:#a02020">' + html.escape(str(exc)) + "</p>"
            )

    def _export_slice(self, _=None):
        import base64

        if self.slice2d.slice_png:
            encoded = base64.b64encode(self.slice2d.slice_png).decode()
            self.slice_download.value = f'<a download="cube-slice.png" href="data:image/png;base64,{encoded}">Download 2D slice PNG</a>'

    def close(self):
        if hasattr(self, "_cube_link"):
            self._cube_link.unlink()
            self.viewer3d.viewer.close()
        super().close()
