"""Render CDXML drawing coordinates consistently across EMPA applications.

This module draws the supplied sketch without inferring chemistry or generating
new coordinates. PNG bytes remain independent of their on-screen display size.
"""

from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
from io import BytesIO

import numpy as np
from ase.data import chemical_symbols
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle


def _elements(root, name):
    return (item for item in root.iter() if item.tag.rsplit("}", 1)[-1] == name)


def _point(value):
    coordinates = tuple(float(value) for value in value.split())
    if len(coordinates) != 2 or not all(math.isfinite(value) for value in coordinates):
        raise ValueError("Invalid CDXML drawing coordinate.")
    return np.asarray(coordinates)


def _automatic_double_positions(positions, bonds):
    """Choose an inward offset for ring bonds without a drawing override."""
    # Import locally to avoid a cycle with the upload widget's preview import.
    from .cdxml import _bounded_faces, _signed_area

    ids = list(positions)
    indices = {identifier: index for index, identifier in enumerate(ids)}
    edges = [(indices[bond.get("B")], indices[bond.get("E")]) for bond in bonds]
    points = np.asarray([positions[identifier] for identifier in ids])
    faces = [
        face for face in _bounded_faces(points, edges) if _signed_area(points[face]) > 0
    ]
    result = {}
    for bond in bonds:
        first, second = indices[bond.get("B")], indices[bond.get("E")]
        adjacent = [
            face
            for face in faces
            if any(
                {left, right} == {first, second}
                for left, right in zip(face, face[1:] + face[:1])
            )
        ]
        if not adjacent:
            continue
        face = min(adjacent, key=lambda face: (len(face), tuple(sorted(face))))
        centre = np.mean(points[face], axis=0)
        delta = points[second] - points[first]
        # CDXML uses screen coordinates: +y goes down. Left looking B -> E
        # is therefore (dy, -dx), as opposed to the Cartesian left normal.
        left_normal = np.array([delta[1], -delta[0]])
        side = np.dot(left_normal, centre - (points[first] + points[second]) / 2)
        result[bond.get("id")] = "Left" if side > 0 else "Right"
    return result


def _bond_segments(first, second, order, position, spacing):
    """Return drawing segments, retaining one full line for offset doubles."""
    delta = second - first
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-12:
        return []
    left = np.array([delta[1], -delta[0]]) / distance
    if order in (2, 1.5) and position in ("Left", "Right"):
        offset = left * spacing * (1 if position == "Left" else -1)
        trim = delta * min(0.22, 0.6 * spacing / distance)
        return [(first, second), (first + trim + offset, second - trim + offset)]
    offsets = (
        (-spacing / 2, spacing / 2)
        if order in (2, 1.5)
        else ((-spacing, 0.0, spacing) if order == 3 else (0.0,))
    )
    return [(first + left * offset, second + left * offset) for offset in offsets]


def render_cdxml_png(content: bytes | str, *, highlighted_bonds=(), max_pixels=1200):
    """Draw a selected CDXML fragment as a proportionate PNG on white.

    Explicit DoublePosition overrides win; ring doubles without an override
    use an inward secondary line. Optional highlighted bond IDs are review
    suggestions, drawn orange and dashed, and excluded from ring placement.
    """
    root = ET.fromstring(content)
    nodes = {
        node.get("id"): node
        for node in _elements(root, "n")
        if node.get("id") and node.get("p")
    }
    if not nodes:
        raise ValueError("The CDXML contains no positioned atoms.")
    positions = {
        identifier: _point(node.get("p")) for identifier, node in nodes.items()
    }
    bonds = [
        bond
        for bond in _elements(root, "b")
        if bond.get("B") in positions and bond.get("E") in positions
    ]
    highlighted = set(highlighted_bonds)
    lengths = [
        np.linalg.norm(positions[bond.get("E")] - positions[bond.get("B")])
        for bond in bonds
        if bond.get("id") not in highlighted
    ]
    lengths = [length for length in lengths if length > 1e-12]
    bond_length = float(np.median(lengths)) if lengths else 14.4
    auto_positions = _automatic_double_positions(
        positions, [bond for bond in bonds if bond.get("id") not in highlighted]
    )
    graphics = {graphic.get("id"): graphic for graphic in _elements(root, "graphic")}
    brackets = []
    for graphic in graphics.values():
        if graphic.get("GraphicType", "").lower() == "bracket" or graphic.get(
            "BracketType"
        ):
            coordinates = [
                float(value) for value in graphic.get("BoundingBox", "").split()
            ]
            if len(coordinates) == 4 and all(
                math.isfinite(value) for value in coordinates
            ):
                brackets.append(
                    (np.asarray(coordinates[:2]), np.asarray(coordinates[2:]))
                )
    bounds = np.asarray(
        [*positions.values(), *(point for bracket in brackets for point in bracket)]
    )
    low = np.min(bounds, axis=0) - 0.55 * bond_length
    high = np.max(bounds, axis=0) + 0.55 * bond_length
    span = high - low
    pixels_per_unit = min(70.0 / bond_length, float(max_pixels) / float(np.max(span)))
    width, height = (max(1, math.ceil(value * pixels_per_unit)) for value in span)
    dpi = 150
    figure = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(high[1], low[1])
    axis.set_aspect("equal")
    axis.axis("off")
    declared_length = float(root.get("BondLength", bond_length)) or bond_length
    stroke = max(
        0.55,
        float(root.get("LineWidth", 0.6))
        / declared_length
        * bond_length
        * pixels_per_unit
        * 72
        / dpi,
    )
    font_size = (
        float(root.get("LabelSize", 10))
        / declared_length
        * bond_length
        * pixels_per_unit
        * 72
        / dpi
    )
    default_spacing = float(root.get("BondSpacing", 12)) / 100 * bond_length

    def line(first, second, color="#151515", style="-", linewidth=stroke):
        axis.plot(
            [first[0], second[0]],
            [first[1], second[1]],
            color=color,
            linestyle=style,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=1,
        )

    for bond in bonds:
        first, second = positions[bond.get("B")], positions[bond.get("E")]
        if bond.get("id") in highlighted:
            line(first, second, color="#d88413", style="--", linewidth=stroke * 1.3)
            continue
        try:
            order = 1.5 if bond.get("Order") == "A" else float(bond.get("Order", "1"))
        except ValueError:
            order = 1.0
        position = bond.get(
            "DoublePosition", auto_positions.get(bond.get("id"), "Center")
        )
        spacing = (
            float(bond.get("BondSpacing", root.get("BondSpacing", 12)))
            / 100
            * bond_length
        )
        display = bond.get("Display", "Solid")
        if order == 1 and display in (
            "WedgeBegin",
            "WedgeEnd",
            "HollowWedgeBegin",
            "HollowWedgeEnd",
        ):
            if display.endswith("End"):
                first, second = second, first
            delta = second - first
            norm = np.linalg.norm(delta)
            if norm:
                offset = np.array([delta[1], -delta[0]]) / norm * default_spacing
                axis.fill(
                    [first[0], second[0] + offset[0], second[0] - offset[0]],
                    [first[1], second[1] + offset[1], second[1] - offset[1]],
                    facecolor="white" if display.startswith("Hollow") else "#151515",
                    edgecolor="#151515",
                    linewidth=stroke,
                    zorder=1,
                )
            continue
        if order == 1 and display in ("WedgedHashBegin", "WedgedHashEnd", "Hash"):
            if display.endswith("End"):
                first, second = second, first
            delta = second - first
            norm = np.linalg.norm(delta)
            if norm:
                normal = np.array([delta[1], -delta[0]]) / norm
                for fraction in np.linspace(0.1, 0.9, 7):
                    middle = first + fraction * delta
                    offset = (
                        normal
                        * default_spacing
                        * (fraction if display != "Hash" else 0.65)
                    )
                    line(middle - offset, middle + offset)
            continue
        segments = _bond_segments(first, second, order, position, spacing)
        for index, (start, end) in enumerate(segments):
            dashed = display in ("Dash", "Dot", "DashDot") or (
                order == 1.5 and index == 1
            )
            line(
                start,
                end,
                style=":" if display == "Dot" else "--" if dashed else "-",
                linewidth=stroke * (2.5 if display == "Bold" else 1),
            )

    for identifier, node in nodes.items():
        position = positions[identifier]
        atomic_number = int(node.get("Element", "6"))
        label = "".join(item.text or "" for item in _elements(node, "s"))
        charge = int(node.get("Charge", "0"))
        if not label and (atomic_number != 6 or charge):
            label = chemical_symbols[atomic_number]
            if charge:
                label += (str(abs(charge)) if abs(charge) > 1 else "") + (
                    "+" if charge > 0 else "−"
                )
        if label:
            axis.text(
                *position,
                label,
                ha="center",
                va="center",
                fontsize=font_size,
                color="#151515",
                zorder=3,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
            )
        if node.get("Radical", "").lower() in ("doublet", "1"):
            radius = 0.035 * bond_length
            axis.add_patch(
                Circle(
                    position + np.array([0.16, -0.16]) * bond_length,
                    radius,
                    color="#151515",
                    zorder=4,
                )
            )
    for graphic in graphics.values():
        if graphic.get("SymbolType", "").lower() != "electron":
            continue
        for representation in _elements(graphic, "represent"):
            identifier = representation.get("object")
            if identifier in positions and nodes[identifier].get(
                "Radical", ""
            ).lower() not in ("doublet", "1"):
                point = positions[identifier] + np.array([0.16, -0.16]) * bond_length
                axis.plot(
                    *point, marker="o", markersize=stroke * 2, color="#151515", zorder=4
                )

    centres = [(first + second) / 2 for first, second in brackets]
    for index, (first, second) in enumerate(brackets):
        if len(brackets) > 1:
            other = max(
                (
                    point
                    for other_index, point in enumerate(centres)
                    if other_index != index
                ),
                key=lambda point: np.linalg.norm(point - centres[index]),
            )
            direction = other - centres[index]
            direction /= np.linalg.norm(direction) or 1.0
        else:
            direction = np.array([1.0, 0.0])
        lip = direction * 0.25 * bond_length
        line(first, second)
        line(first, first + lip)
        line(second, second + lip)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, facecolor="white")
    figure.clear()
    return buffer.getvalue()


def set_png_widget(image_widget, content: bytes, *, max_side=300):
    """Set PNG bytes and cap only the longest displayed side, without upscaling."""
    image_widget.value = bytes(content)
    if len(content) >= 24 and content.startswith(b"\x89PNG\r\n\x1a\n"):
        width, height = struct.unpack(">II", content[16:24])
        if width and height:
            scale = min(1.0, max_side / max(width, height))
            width, height = max(1, round(width * scale)), max(1, round(height * scale))
            image_widget.width, image_widget.height = str(width), str(height)
            image_widget.layout.width, image_widget.layout.height = (
                f"{width}px",
                f"{height}px",
            )
            return
    image_widget.width = image_widget.height = ""
    image_widget.layout.width = image_widget.layout.height = "auto"
