"""Selected-sketch previews and cleanup must agree with model creation."""

from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET

import ipywidgets as ipw
import numpy as np
import pytest
from PIL import Image

from aiidalab_widgets_empa import CdxmlUploadWidget
from aiidalab_widgets_empa.cdxml_rendering import (
    _automatic_double_positions,
    _bond_segments,
    render_cdxml_png,
    set_png_widget,
)

DATA = Path(__file__).parent


def upload(widget, content):
    widget._on_file_upload({"new": ({"name": "drawing.cdxml", "content": content},)})


def drawing_positions(content):
    return np.asarray(
        [
            list(map(float, node.get("p").split()))
            for node in ET.fromstring(content).iter("n")
            if node.get("p")
        ]
    )


def test_preview_exists_before_create_model_and_updates_with_selection():
    from test_cdxml_selection import _multi_structure_cdxml

    widget = CdxmlUploadWidget()
    upload(widget, _multi_structure_cdxml().encode())
    first = bytes(widget.png_preview.value)
    assert first.startswith(b"\x89PNG")
    assert widget.structure is None
    assert max(int(widget.png_preview.width), int(widget.png_preview.height)) <= 300
    widget.structure_selector.value = 1
    assert bytes(widget.png_preview.value) != first
    assert widget.structure is None
    assert len(drawing_positions(widget._preview_cdxml)) == 3
    widget.preview_zoom.click()
    assert widget._preview_expanded
    assert max(int(widget.png_preview.width), int(widget.png_preview.height)) == min(
        900, max(Image.open(BytesIO(bytes(widget.png_preview.value))).size)
    )
    widget.preview_zoom.click()
    assert not widget._preview_expanded
    assert max(int(widget.png_preview.width), int(widget.png_preview.height)) <= 300


def test_cleanup_toggle_redraws_and_create_uses_the_same_prepared_geometry(monkeypatch):
    widget = CdxmlUploadWidget()
    upload(widget, (DATA / "mixed_rings_periodic.cdxml").read_bytes())
    original_png = bytes(widget.png_preview.value)
    original_positions = drawing_positions(widget._preview_cdxml)
    widget.symmetrize_geometry.value = True
    assert widget.structure is None
    assert "cleaned globally" in widget.output_message.value
    cleaned_positions = drawing_positions(widget._preview_cdxml)
    cleaned_png = bytes(widget.png_preview.value)
    assert cleaned_png != original_png
    assert not np.allclose(cleaned_positions, original_positions)
    # The preview must be a uniform scaling/rotation of the exact prepared
    # atoms; this checks all pair distances rather than just the visible edges.
    prepared = widget.atoms.positions.copy()
    preview_distances = np.linalg.norm(
        cleaned_positions[:, None] - cleaned_positions[None, :], axis=2
    )
    model_distances = np.linalg.norm(prepared[:, None] - prepared[None, :], axis=2)
    scale = preview_distances.max() / model_distances.max()
    assert np.allclose(preview_distances, model_distances * scale, atol=1e-5)
    preview_root = ET.fromstring(widget._preview_cdxml)
    for key in ("n", "b", "bracketedgroup", "crossingbond"):
        assert len(list(preview_root.iter(key))) == len(
            list(ET.fromstring(widget._selected_cdxml_content()).iter(key))
        )
    with monkeypatch.context() as context:

        def unexpected_conversion(*args, **kwargs):
            raise AssertionError("Create must use the geometry already shown")

        context.setattr(widget, "cdxml_to_ase_from_string", unexpected_conversion)
        widget.create_button.click()
        assert widget.structure is not None
        assert widget.structure.pbc.tolist() == [True, False, False]
    assert np.allclose(widget.atoms.positions, prepared)
    widget.symmetrize_geometry.value = False
    assert widget.structure is None
    assert bytes(widget.png_preview.value) == original_png
    assert np.allclose(drawing_positions(widget._preview_cdxml), original_positions)
    assert "Original CDXML geometry" in widget.output_message.value
    widget.create_button.click()
    assert widget.structure is not None


def test_bad_upload_clears_the_previous_preview():
    widget = CdxmlUploadWidget()
    upload(widget, (DATA / "benzene.cdxml").read_bytes())
    assert widget.png_preview.value
    upload(widget, b"<not-valid-xml")
    assert not widget.png_preview.value
    assert widget.preview_zoom.disabled
    assert "unavailable" in widget.preview_message.value
    assert widget.structure is None
    assert widget._conversion_signature is None


@pytest.mark.parametrize(
    "source,expected",
    [
        ((600, 400), (300, 200)),
        ((300, 900), (100, 300)),
        ((120, 80), (120, 80)),
    ],
)
def test_shared_image_sizing_preserves_bytes_and_ratio(source, expected):
    buffer = BytesIO()
    Image.new("RGB", source, "white").save(buffer, format="PNG")
    content = buffer.getvalue()
    widget = ipw.Image(format="png")
    set_png_widget(widget, content)
    assert bytes(widget.value) == content
    assert (int(widget.width), int(widget.height)) == expected
    assert (widget.layout.width, widget.layout.height) == tuple(
        f"{value}px" for value in expected
    )


def test_offset_double_bonds_follow_cdxml_screen_left_and_right():
    first, second = np.array([0.0, 0.0]), np.array([10.0, 0.0])
    left = _bond_segments(first, second, 2, "Left", 1.0)
    right = _bond_segments(first, second, 2, "Right", 1.0)
    assert np.allclose(left[0], [first, second])
    assert np.allclose(right[0], [first, second])
    assert all(point[1] < 0 for point in left[1])
    assert all(point[1] > 0 for point in right[1])
    assert left[1][0][0] > 0 and left[1][1][0] < 10


def test_automatic_ring_double_lines_point_inward():
    root = ET.fromstring((DATA / "benzene.cdxml").read_bytes())
    positions = {
        node.get("id"): np.asarray(list(map(float, node.get("p").split())))
        for node in root.iter("n")
    }
    bonds = list(root.iter("b"))
    sides = _automatic_double_positions(positions, bonds)
    centre = np.mean(list(positions.values()), axis=0)
    for bond in bonds:
        if bond.get("Order") != "2":
            continue
        first, second = positions[bond.get("B")], positions[bond.get("E")]
        inside = _bond_segments(first, second, 2, sides[bond.get("id")], 2.0)[1]
        assert np.linalg.norm(np.mean(inside, axis=0) - centre) < np.linalg.norm(
            (first + second) / 2 - centre
        )
    image = Image.open(BytesIO(render_cdxml_png(ET.tostring(root))))
    assert image.width > 50 and image.height > 50
    assert np.array(image)[:, :, :3].min() < 50


def test_cleanup_brackets_have_one_bond_padding_and_follow_the_same_cut_lines():
    # Bracket extent is a graphical choice, not a crossing bond midpoint.
    root = ET.fromstring((DATA / "mixed_rings_periodic.cdxml").read_bytes())
    for graphic in root.iter("graphic"):
        if graphic.get("BracketType") == "Square":
            values = list(map(float, graphic.get("BoundingBox").split()))
            values[1] -= 45
            values[3] -= 45
            graphic.set("BoundingBox", " ".join(map(str, values)))
    widget = CdxmlUploadWidget()
    upload(widget, ET.tostring(root))
    original = drawing_positions(widget._selected_cdxml_content())
    original = np.column_stack([original, np.zeros(len(original))])
    widget.symmetrize_geometry.value = True
    updated = drawing_positions(widget._preview_cdxml)
    updated = np.column_stack([updated, np.zeros(len(updated))])
    indices = {node.get("id"): index for index, node in enumerate(root.iter("n"))}
    lengths = [
        np.linalg.norm(
            updated[indices[bond.get("B")]] - updated[indices[bond.get("E")]]
        )
        for bond in root.iter("b")
    ]
    padding = np.median(lengths)
    boundaries = np.asarray(
        widget.transform_points(widget.atoms.positions, updated, widget.crossing_points)
    )
    cleaned_root = ET.fromstring(widget._preview_cdxml)
    cleaned_graphics = {
        graphic.get("id"): graphic for graphic in cleaned_root.iter("graphic")
    }
    for graphic in root.iter("graphic"):
        if graphic.get("BracketType") != "Square":
            continue
        corners = np.asarray(
            list(map(float, graphic.get("BoundingBox").split()))
        ).reshape((2, 2))
        corners = np.column_stack([corners, np.zeros(2)])
        expected = np.asarray(widget.transform_points(original, updated, corners))
        actual = np.asarray(
            list(
                map(
                    float,
                    cleaned_graphics[graphic.get("id")].get("BoundingBox").split(),
                )
            )
        ).reshape((2, 2))
        actual = np.column_stack([actual, np.zeros(2)])
        tangent = expected[1] - expected[0]
        tangent /= np.linalg.norm(tangent)
        projections = updated @ tangent
        assert actual[0] @ tangent == pytest.approx(
            projections.min() - padding, abs=1e-6
        )
        assert actual[1] @ tangent == pytest.approx(
            projections.max() + padding, abs=1e-6
        )
        normal = np.array([-tangent[1], tangent[0], 0.0])
        assert (
            min(
                abs((actual.mean(axis=0) - boundary) @ normal)
                for boundary in boundaries
            )
            < 1e-6
        )
        assert np.isfinite(actual).all()


@pytest.mark.parametrize("drawing_scale", [1.0, 5.0])
def test_independent_bracket_shifts_use_angstroms_and_preserve_the_model(drawing_scale):
    root = ET.fromstring((DATA / "mixed_rings_periodic.cdxml").read_bytes())
    for node in root.iter("n"):
        node.set(
            "p",
            " ".join(
                str(float(value) * drawing_scale) for value in node.get("p").split()
            ),
        )
    for graphic in root.iter("graphic"):
        if graphic.get("BracketType") == "Square":
            graphic.set(
                "BoundingBox",
                " ".join(
                    str(float(value) * drawing_scale)
                    for value in graphic.get("BoundingBox").split()
                ),
            )
    widget = CdxmlUploadWidget()
    upload(widget, ET.tostring(root))
    original_png = bytes(widget.png_preview.value)
    widget.symmetrize_geometry.value = True
    assert widget.bracket_controls.layout.display == "flex"
    baseline_png = bytes(widget.png_preview.value)
    widget.create_button.click()
    baseline = widget.structure.copy()
    boundaries = widget.crossing_points.copy()
    assert baseline.cell[0, 0] == pytest.approx(
        np.linalg.norm(boundaries[1] - boundaries[0])
    )

    def boxes():
        graphics = [
            graphic
            for graphic in ET.fromstring(widget._preview_cdxml).iter("graphic")
            if graphic.get("BracketType") == "Square"
        ]
        result = {
            graphic.get("id"): np.asarray(
                list(map(float, graphic.get("BoundingBox").split()))
            ).reshape((2, 2))
            for graphic in graphics
        }
        sides = sorted(
            result, key=lambda identifier: tuple(result[identifier].mean(axis=0))
        )
        return result, sides

    before, sides = boxes()
    physical = widget.atoms.positions[:, :2]
    drawn = drawing_positions(widget._preview_cdxml)
    unit_scale = (
        np.linalg.norm(drawn - drawn.mean(axis=0), axis=1).mean()
        / np.linalg.norm(physical - physical.mean(axis=0), axis=1).mean()
    )
    assert widget.bracket_shift_step.value == 0.1
    widget.preview_zoom.click()
    assert widget._preview_expanded
    widget.bracket_shift_buttons[("left", 1)].click()
    assert widget._preview_expanded
    after, _ = boxes()
    assert widget.bracket_shifts["left"].value == 0.1
    assert widget.bracket_shifts["right"].value == 0.0
    assert np.allclose(after[sides[1]], before[sides[1]])
    assert np.allclose(
        after[sides[0]] - before[sides[0]], [0.1 * unit_scale, 0.0], atol=1e-6
    )
    widget.bracket_shift_step.value = 0.05
    widget.bracket_shift_buttons[("right", -1)].click()
    after, _ = boxes()
    assert widget.bracket_shifts["right"].value == -0.05
    assert all(offset.step == 0.05 for offset in widget.bracket_shifts.values())
    assert np.allclose(
        after[sides[1]] - before[sides[1]], [-0.05 * unit_scale, 0.0], atol=1e-6
    )
    assert np.array_equal(widget.crossing_points, boundaries)
    widget.create_button.click()
    assert np.array_equal(widget.structure.positions, baseline.positions)
    assert np.array_equal(widget.structure.cell, baseline.cell)
    assert np.array_equal(widget.structure.pbc, baseline.pbc)
    widget.symmetrize_geometry.value = False
    assert bytes(widget.png_preview.value) == original_png
    assert widget.bracket_controls.layout.display == "none"
    widget.symmetrize_geometry.value = True
    widget.bracket_shift_resets["left"].click()
    assert widget.bracket_shifts["left"].value == 0
    assert widget.bracket_shifts["right"].value == -0.05
    widget.bracket_shift_resets["right"].click()
    assert bytes(widget.png_preview.value) == baseline_png
    widget.bracket_shifts["left"].value = 0.2
    upload(widget, ET.tostring(root))
    assert all(offset.value == 0 for offset in widget.bracket_shifts.values())
