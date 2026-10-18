"""Import molecular and periodic structures from ChemDraw CDXML files."""

from __future__ import annotations

import hashlib
import html
import math
import uuid
import xml.etree.ElementTree as ET

import ase
import ipywidgets as ipw
import numpy as np
import traitlets as tr
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from scipy.optimize import least_squares, linear_sum_assignment

from .cdxml_graph import cdxml_root, read_cdxml_graph
from .cdxml_periodic import (
    PeriodicAmbiguity,
    RepeatUnit,
    build_repeat_model,
    identify_repeat,
)
from .cdxml_rendering import render_cdxml_png, set_png_widget
from .cdxml_sterics import MINIMUM_CONTACT_DISTANCES, relieve_steric_contacts


# ---------------- Utility functions ----------------
def normalize(v: np.ndarray) -> np.ndarray:
    """Return the normalized version of vector v, or zeros if near-zero norm."""
    v = np.array(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros(3)


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the rotation matrix that rotates vector a into vector b."""
    a, b = normalize(a), normalize(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3)
    vx = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))


def rotate_vector(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector v around given axis by 'angle' radians."""
    axis = normalize(axis)
    v = np.array(v)
    return (
        v * math.cos(angle)
        + np.cross(axis, v) * math.sin(angle)
        + axis * np.dot(axis, v) * (1 - math.cos(angle))
    )


def _signed_area(points: np.ndarray) -> float:
    """Return the signed area of a two-dimensional polygon."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _bounded_faces(
    positions: np.ndarray, edges: list[tuple[int, int]]
) -> list[list[int]]:
    """Find bounded faces in an existing straight-line planar embedding."""
    adjacency = {index: [] for index in range(len(positions))}
    for first, second in edges:
        adjacency[first].append(second)
        adjacency[second].append(first)

    ordered = {}
    for center, neighbors in adjacency.items():
        ordered[center] = sorted(
            neighbors,
            key=lambda neighbor: math.atan2(
                positions[neighbor, 1] - positions[center, 1],
                positions[neighbor, 0] - positions[center, 0],
            ),
        )

    visited = set()
    faces = []
    for first, second in edges:
        for start in ((first, second), (second, first)):
            if start in visited:
                continue
            face = []
            current = start
            while current not in visited:
                visited.add(current)
                left, right = current
                face.append(left)
                neighbors = ordered[right]
                if not neighbors:
                    break
                incoming = neighbors.index(left)
                current = (right, neighbors[(incoming - 1) % len(neighbors)])
                if current == start:
                    break
            if current == start and len(face) >= 3 and len(set(face)) == len(face):
                faces.append(face)

    if not faces:
        return []
    areas = [abs(_signed_area(positions[face])) for face in faces]
    exterior = int(np.argmax(areas))
    return [
        face
        for index, face in enumerate(faces)
        if index != exterior and 3 <= len(face) <= 12
    ]


def _segments_properly_cross(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    fourth: np.ndarray,
    tolerance: float = 1.0e-10,
) -> bool:
    """Return whether two line segments cross away from their endpoints."""

    def orientation(a, b, c):
        first = b - a
        second = c - a
        return float(first[0] * second[1] - first[1] * second[0])

    def lies_on_segment(start, end, point):
        if abs(orientation(start, end, point)) > tolerance:
            return False
        return bool(
            np.all(point >= np.minimum(start, end) - tolerance)
            and np.all(point <= np.maximum(start, end) + tolerance)
        )

    o1 = orientation(first, second, third)
    o2 = orientation(first, second, fourth)
    o3 = orientation(third, fourth, first)
    o4 = orientation(third, fourth, second)
    if o1 * o2 < -tolerance and o3 * o4 < -tolerance:
        return True
    return any(
        (
            lies_on_segment(first, second, third),
            lies_on_segment(first, second, fourth),
            lies_on_segment(third, fourth, first),
            lies_on_segment(third, fourth, second),
        )
    )


def _has_bond_crossings(positions: np.ndarray, edges: list[tuple[int, int]]) -> bool:
    """Check a straight-line graph embedding for non-adjacent bond crossings."""
    for edge_index, (first, second) in enumerate(edges):
        for third, fourth in edges[edge_index + 1 :]:
            if len({first, second, third, fourth}) < 4:
                continue
            if _segments_properly_cross(
                positions[first], positions[second], positions[third], positions[fourth]
            ):
                return True
    return False


def _top_level_cdxml_fragments(root: ET.Element) -> list[ET.Element]:
    """Return chemical fragments that are not nested in another fragment."""
    parents = {child: parent for parent in root.iter() for child in parent}
    fragments = []
    for fragment in root.iter("fragment"):
        parent = parents.get(fragment)
        while parent is not None and parent.tag != "fragment":
            parent = parents.get(parent)
        if parent is not None:
            continue
        if any(node.get("p") for node in fragment.iter("n")):
            fragments.append(fragment)
    return fragments


def _element_bounding_box(
    element: ET.Element,
) -> tuple[float, float, float, float] | None:
    """Return an element bounding box, falling back to its point position."""
    if bounding_box := element.get("BoundingBox"):
        coordinates = [float(value) for value in bounding_box.split()]
        if len(coordinates) == 4:
            first_x, first_y, second_x, second_y = coordinates
            return (
                min(first_x, second_x),
                min(first_y, second_y),
                max(first_x, second_x),
                max(first_y, second_y),
            )
    if position := element.get("p"):
        coordinates = [float(value) for value in position.split()]
        if len(coordinates) >= 2:
            return (coordinates[0], coordinates[1], coordinates[0], coordinates[1])
    return None


def _rectangle_distance(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Return the shortest distance between two axis-aligned rectangles."""
    first_min_x, first_min_y, first_max_x, first_max_y = first
    second_min_x, second_min_y, second_max_x, second_max_y = second
    delta_x = max(first_min_x - second_max_x, second_min_x - first_max_x, 0.0)
    delta_y = max(first_min_y - second_max_y, second_min_y - first_max_y, 0.0)
    return math.hypot(delta_x, delta_y)


# ---------------- Main Widget ----------------
class CdxmlUploadWidget(ipw.VBox):
    """Widget for uploading CDXML files and converting them into ASE.Atoms structures."""

    structure = tr.Instance(ase.Atoms, allow_none=True)
    _maximum_label_distance_in_bonds = 2.5

    def __init__(self, title: str = "CDXML", description: str = "Upload CDXML"):
        self.title = title

        # --- File upload widget ---
        self.file_upload = ipw.FileUpload(
            description=description,
            multiple=False,
            layout={"width": "initial"},
            accept=".cdxml",
        )
        self.file_upload.observe(self._on_file_upload, names="value")

        # --- Additional widgets ---
        self.structure_selector = ipw.Dropdown(
            description="Structure",
            options=(),
            disabled=True,
            layout={"display": "none", "width": "initial"},
            style={"description_width": "initial"},
        )
        self.structure_selector.observe(self._on_structure_selection, names="value")
        self.png_preview = ipw.Image(format="png")
        self.preview_canvas = ipw.Box([self.png_preview])
        self._picker_class = f"cdxml-pick-{uuid.uuid4().hex}"
        self.preview_canvas.add_class(self._picker_class)
        self.picker_css = ipw.HTML()
        self._preview_layout = None
        self._pair_pick_next = 0
        self.preview_message = ipw.HTML()
        self.preview_zoom = ipw.Button(
            description="Enlarge preview",
            icon="search-plus",
            disabled=True,
            layout={"width": "initial"},
        )
        self.preview_zoom.on_click(self._toggle_preview_size)
        self.bracket_shifts = {
            side: ipw.FloatText(
                value=0.0,
                description=f"{side.capitalize()} (Å)",
                step=0.1,
                layout={"width": "170px"},
                style={"description_width": "initial"},
            )
            for side in ("left", "right")
        }
        self.bracket_shift_step = ipw.BoundedFloatText(
            value=0.1,
            min=0.001,
            max=10.0,
            step=0.1,
            description="Step (Å)",
            layout={"width": "165px"},
            style={"description_width": "initial"},
        )
        self.bracket_shift_buttons = {}
        self.bracket_shift_resets = {}
        self._updating_bracket_shift = False
        rows = [ipw.HTML("<b>Bracket drawing shifts</b>"), self.bracket_shift_step]
        for side, offset in self.bracket_shifts.items():
            offset.observe(self._on_bracket_shift_change, names="value")
            buttons = []
            for direction, icon in ((-1, "arrow-left"), (1, "arrow-right")):
                button = ipw.Button(
                    icon=icon,
                    tooltip=f"Move the {side} bracket",
                    layout={"width": "40px"},
                )
                button.on_click(
                    lambda _, selected=side, sign=direction: (
                        self._shift_bracket_drawing(selected, sign)
                    )
                )
                self.bracket_shift_buttons[(side, direction)] = button
                buttons.append(button)
            reset = ipw.Button(
                icon="undo",
                tooltip=f"Reset the {side} bracket drawing shift",
                layout={"width": "40px"},
            )
            reset.on_click(
                lambda _, selected=side: setattr(
                    self.bracket_shifts[selected], "value", 0.0
                )
            )
            self.bracket_shift_resets[side] = reset
            rows.append(ipw.HBox([offset, *buttons, reset]))
        self.bracket_shift_step.observe(self._on_bracket_step_change, names="value")
        self.bracket_controls = ipw.VBox(rows, layout={"display": "none"})
        self.preview_box = ipw.VBox(
            [
                ipw.HTML("<b>Selected chemical sketch</b>"),
                self.preview_canvas,
                self.picker_css,
                self.preview_zoom,
                self.bracket_controls,
                self.preview_message,
            ],
            layout={"display": "none"},
        )
        self._preview_expanded = False
        self.nunits = ipw.Text(description="N units", value="Infinite", disabled=True)
        self.nunits.observe(lambda _: setattr(self, "structure", None), names="value")
        self.equivalent_atoms = [
            ipw.BoundedIntText(
                value=index,
                min=0,
                max=10000,
                description=label,
                layout={"width": "180px"},
            )
            for index, label in enumerate(("Atom A", "Equivalent B"))
        ]
        for field in self.equivalent_atoms:
            field.observe(self._on_pair_field_change, names="value")
        self.apply_atom_pair = ipw.Button(
            description="Use atom pair", icon="link", layout={"width": "initial"}
        )
        self.apply_atom_pair.on_click(self._apply_periodic_atom_pair)
        self.reset_atom_pair = ipw.Button(
            description="Use automatic matching",
            icon="undo",
            layout={"width": "initial"},
        )
        self.reset_atom_pair.on_click(self._reset_periodic_atom_pair)
        self.show_atom_indices = ipw.Checkbox(
            description="Show indices and select atoms by clicking",
            value=False,
            indent=False,
        )
        self.show_atom_indices.observe(
            lambda _: (
                self._update_png_preview(reset_zoom=False)
                if self._cdxml_content is not None
                else None
            ),
            names="value",
        )
        self.equivalence_controls = ipw.VBox(
            [
                ipw.HTML(
                    "Choose two equivalent atoms in adjacent cells if periodicity is ambiguous. Indices start at 0."
                ),
                self.show_atom_indices,
                ipw.HBox(self.equivalent_atoms),
                ipw.HBox([self.apply_atom_pair, self.reset_atom_pair]),
            ],
            layout={"display": "none"},
        )
        self.use_clever_hydrogenation = ipw.Checkbox(
            description="Infer implicit hydrogens from bond orders",
            value=True,
            indent=False,
        )
        self.symmetrize_geometry = ipw.Checkbox(
            description="Clean up 2D carbon geometry",
            value=False,
            indent=False,
        )
        self.symmetrize_geometry.observe(self._on_geometry_option_change, names="value")
        self.resolve_steric_collisions = ipw.Checkbox(
            description="Resolve steric contacts in 3D",
            value=False,
            disabled=True,
            indent=False,
        )
        self.resolve_steric_collisions.observe(
            self._on_steric_option_change, names="value"
        )
        self.steric_mode = ipw.Dropdown(
            description="3D method",
            options=[
                ("Steric separation", "steric"),
                ("Minimal 3D torsion", "minimal"),
            ],
            value="steric",
            layout={"width": "auto"},
        )
        self.minimum_contact_distances = {}
        for pair, distance in MINIMUM_CONTACT_DISTANCES.items():
            cutoff = 1.09 * sum(
                covalent_radii[6 if symbol == "C" else 1] for symbol in pair
            )
            field = ipw.BoundedFloatText(
                description=f"{'–'.join(pair)} (Å)",
                value=distance,
                min=math.ceil((cutoff + 0.001) * 100) / 100,
                max=10.0,
                step=0.01,
                layout={"width": "190px"},
            )
            field.observe(self._on_steric_option_change, names="value")
            self.minimum_contact_distances[pair] = field
        self.steric_margin_percent = ipw.BoundedFloatText(
            description="Extra margin (%)",
            value=0.0,
            min=0.0,
            max=100.0,
            step=1.0,
            style={"description_width": "initial"},
            layout={"width": "240px"},
        )
        self.steric_margin_percent.observe(self._on_steric_option_change, names="value")
        self.steric_mode.observe(self._on_steric_option_change, names="value")
        self.minimum_contacts_box = ipw.VBox(
            [
                ipw.HTML(
                    "Minimum nonbonded distances; the extra margin scales all of them."
                ),
                ipw.HBox(
                    list(self.minimum_contact_distances.values()),
                    layout={"flex_flow": "row wrap"},
                ),
                self.steric_margin_percent,
            ],
            layout={"display": "none"},
        )
        self.steric_controls = ipw.VBox(
            [self.steric_mode, self.minimum_contacts_box],
            layout={"display": "none"},
        )
        self.use_clever_hydrogenation.observe(
            self._on_hydrogenation_option_change, names="value"
        )
        self.steric_help = ipw.HTML(
            "Finite C/H/halogen molecules: rotate connected aromatic blocks and "
            "methyl hydrogens. The sketch stays 2D; Create model produces a 3D "
            "starting conformer, without energy optimization.",
            layout={"display": "none"},
        )
        self.create_button = ipw.Button(
            description="Create model", button_style="success"
        )
        self.create_button.on_click(self._on_button_click)

        supported_formats = ipw.HTML(
            """
            <a href="https://pubs.acs.org/doi/10.1021/ja0697875" target="_blank">
            Supported structure formats: ".cdxml"
            </a>
            """
        )
        self.output_message = ipw.HTML(value="")

        # --- Layout ---
        super().__init__(
            children=[
                self.file_upload,
                self.structure_selector,
                self.preview_box,
                self.nunits,
                self.equivalence_controls,
                self.use_clever_hydrogenation,
                self.symmetrize_geometry,
                self.resolve_steric_collisions,
                self.steric_controls,
                self.steric_help,
                supported_formats,
                self.create_button,
                self.output_message,
            ]
        )

        # --- Internal state ---
        self.structure: ase.Atoms | None = None
        self.crossing_points: np.ndarray | None = None
        self.cdxml_atoms: np.ndarray | None = None
        self.atoms: ase.Atoms | None = None
        self.whole_atoms: ase.Atoms | None = None
        self._cdxml_content = None
        self._conversion_signature = None
        self._preview_cdxml = b""
        self._geometry_message = ""
        self._updating_structure_selector = False
        self._updating_pair = False
        self._pair_dirty = False
        self._equivalent_atom_ids = None
        self._source_graph = None
        self._periodic_repeat = None

    def _on_pair_field_change(self, _=None):
        if not self._updating_pair:
            self.structure = None
            self._pair_dirty = True
            self.create_button.disabled = True

    def _apply_periodic_atom_pair(self, _=None):
        self.structure = None
        self._pair_dirty = False
        if self._source_graph is None:
            self.output_message.value = "Error: Load a valid chemical drawing first."
            return
        indices = [field.value for field in self.equivalent_atoms]
        if any(index >= len(self._source_graph.atoms) for index in indices):
            self.output_message.value = "Error: The selected atom index is unavailable."
            return
        self._equivalent_atom_ids = tuple(
            self._source_graph.atoms[index].id for index in indices
        )
        self._conversion_signature = None
        self._convert_uploaded_cdxml()
        self._update_png_preview(reset_zoom=False)

    def _reset_periodic_atom_pair(self, _=None):
        self.structure = None
        self._equivalent_atom_ids = None
        self._pair_dirty = False
        self._conversion_signature = None
        if self._cdxml_content is not None:
            self._convert_uploaded_cdxml()
            self._update_png_preview(reset_zoom=False)

    # ---------------- Event handlers ----------------
    @classmethod
    def _structure_caption_matches(
        cls, cdxml_content: str | bytes
    ) -> list[tuple[str | None, int | None]]:
        """Find top-level structures and associate nearby captions one-to-one."""
        root = cdxml_root(cdxml_content)
        fragments = _top_level_cdxml_fragments(root)
        if not fragments:
            if any(node.get("p") for node in root.iter("n")):
                return [(None, None)]
            return []

        parents = {child: parent for parent in root.iter() for child in parent}
        captions = []

        def page_of(element):
            while element is not None and element.tag != "page":
                element = parents.get(element)
            return element

        for caption_index, text_element in enumerate(root.iter("t")):
            parent = parents.get(text_element)
            is_chemical_label = False
            while parent is not None:
                if parent.tag in {"fragment", "n"}:
                    is_chemical_label = True
                    break
                parent = parents.get(parent)
            if is_chemical_label:
                continue
            text = "".join(text_element.itertext()).strip()
            bounding_box = _element_bounding_box(text_element)
            if text and bounding_box is not None:
                captions.append(
                    (text, bounding_box, caption_index, page_of(text_element))
                )

        root_bond_length = None
        try:
            root_bond_length = float(root.get("BondLength", ""))
        except ValueError:
            pass

        fragment_boxes = []
        fragment_scales: list[float | None] = []
        for fragment in fragments:
            atom_positions = {
                node.get("id"): np.asarray(
                    [float(value) for value in node.get("p", "").split()[:2]]
                )
                for node in fragment.iter("n")
                if node.get("id") and len(node.get("p", "").split()) >= 2
            }
            points = np.asarray(list(atom_positions.values()))
            fragment_boxes.append(
                (
                    float(np.min(points[:, 0])),
                    float(np.min(points[:, 1])),
                    float(np.max(points[:, 0])),
                    float(np.max(points[:, 1])),
                )
            )
            bond_lengths = [
                np.linalg.norm(
                    atom_positions[bond.get("B")] - atom_positions[bond.get("E")]
                )
                for bond in fragment.iter("b")
                if bond.get("B") in atom_positions and bond.get("E") in atom_positions
            ]
            scale = float(np.median(bond_lengths)) if bond_lengths else None
            fragment_scales.append(scale if scale and scale > 0.0 else None)

        available_scales = [scale for scale in fragment_scales if scale is not None]
        fallback_scale = root_bond_length
        if not fallback_scale or fallback_scale <= 0.0:
            fallback_scale = (
                float(np.median(available_scales)) if available_scales else 1.0
            )
        resolved_fragment_scales = [
            scale if scale is not None else fallback_scale for scale in fragment_scales
        ]

        assigned_labels: list[str | None] = [None] * len(fragments)
        assigned_captions: list[int | None] = [None] * len(fragments)
        if captions:
            normalized_distances = np.asarray(
                [
                    [
                        _rectangle_distance(fragment_box, caption_box) / scale
                        if page_of(fragment) is caption_page
                        else np.inf
                        for _, caption_box, _, caption_page in captions
                    ]
                    for fragment, fragment_box, scale in zip(
                        fragments, fragment_boxes, resolved_fragment_scales
                    )
                ]
            )
            unmatched_cost = cls._maximum_label_distance_in_bonds
            costs = np.full(
                (len(fragments), len(captions) + len(fragments)),
                unmatched_cost,
            )
            costs[:, : len(captions)] = normalized_distances
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if (
                    column < len(captions)
                    and normalized_distances[row, column] <= unmatched_cost
                ):
                    assigned_labels[row] = captions[column][0]
                    assigned_captions[row] = captions[column][2]

        return list(zip(assigned_labels, assigned_captions))

    @classmethod
    def _cdxml_structure_options(
        cls, cdxml_content: str | bytes
    ) -> list[tuple[str, int]]:
        """Display one nearby caption, or a number, for each structure."""
        assigned_labels = [
            label for label, _ in cls._structure_caption_matches(cdxml_content)
        ]

        label_counts = {
            label: assigned_labels.count(label)
            for label in assigned_labels
            if label is not None
        }
        options = []
        for index, label in enumerate(assigned_labels, start=1):
            if label is None:
                display_label = f"Structure {index}"
            elif label_counts[label] > 1:
                display_label = f"{label} — Structure {index}"
            else:
                display_label = label
            options.append((display_label, index - 1))
        return options

    @classmethod
    def _select_cdxml_structure(
        cls, cdxml_content: str | bytes, structure_index: int
    ) -> str | bytes:
        """Return CDXML containing only the selected top-level structure."""
        root = cdxml_root(cdxml_content)
        fragments = _top_level_cdxml_fragments(root)
        if not fragments:
            if structure_index == 0 and any(node.get("p") for node in root.iter("n")):
                return cdxml_content
            raise ValueError("No chemical structure was found in the CDXML file.")
        if not 0 <= structure_index < len(fragments):
            raise ValueError("The selected CDXML structure is unavailable.")
        if len(fragments) == 1:
            return cdxml_content

        # Keep the caption assigned in the complete document. Reassigning after
        # removing other structures could steal a distant label for an unlabeled one.
        caption_index = cls._structure_caption_matches(cdxml_content)[structure_index][
            1
        ]
        parents = {child: parent for parent in root.iter() for child in parent}
        for index, text_element in enumerate(list(root.iter("t"))):
            parent = parents.get(text_element)
            chemical = False
            while parent is not None:
                if parent.tag in {"fragment", "n"}:
                    chemical = True
                    break
                parent = parents.get(parent)
            if not chemical and index != caption_index:
                parents[text_element].remove(text_element)

        selected_fragment = fragments[structure_index]
        selected_ids = {
            element.get("id")
            for element in selected_fragment.iter()
            if element.get("id")
        }
        selected_bond_ids = {
            bond.get("id") for bond in selected_fragment.iter("b") if bond.get("id")
        }

        parents = {child: parent for parent in root.iter() for child in parent}
        for fragment in fragments:
            if fragment is not selected_fragment:
                parents[fragment].remove(fragment)

        selected_graphic_ids = set()
        parents = {child: parent for parent in root.iter() for child in parent}
        for group in list(root.iter("bracketedgroup")):
            object_ids = set(group.get("BracketedObjectIDs", "").split())
            crossing_bond_ids = {
                crossing.get("BondID") for crossing in group.iter("crossingbond")
            }
            if object_ids & selected_ids or crossing_bond_ids & selected_bond_ids:
                selected_graphic_ids.update(
                    attachment.get("GraphicID")
                    for attachment in group.findall("./bracketattachment")
                    if attachment.get("GraphicID")
                )
            else:
                parents[group].remove(group)

        parents = {child: parent for parent in root.iter() for child in parent}
        for graphic in list(root.iter("graphic")):
            is_unused_bracket = (
                graphic.get("BracketType") is not None
                and graphic.get("id") not in selected_graphic_ids
            )
            is_unused_electron = graphic.get("SymbolType") == "Electron" and not any(
                representation.get("object") in selected_ids
                for representation in graphic.iter("represent")
            )
            if is_unused_bracket or is_unused_electron:
                parents[graphic].remove(graphic)

        return ET.tostring(root, encoding="utf-8")

    def _configure_structure_selector(self) -> None:
        """Populate the structure selector for the uploaded document."""
        if self._cdxml_content is None:
            options = []
        else:
            options = self._cdxml_structure_options(self._cdxml_content)
        self._updating_structure_selector = True
        try:
            self.structure_selector.options = options
            self.structure_selector.value = options[0][1] if options else None
            self.structure_selector.disabled = len(options) <= 1
            self.structure_selector.layout.display = (
                "flex" if len(options) > 1 else "none"
            )
        finally:
            self._updating_structure_selector = False

    def _selected_cdxml_content(self) -> str | bytes:
        """Return the uploaded CDXML restricted to the current selection."""
        if self._cdxml_content is None:
            raise ValueError("No CDXML file has been uploaded.")
        if not self.structure_selector.options:
            return self._cdxml_content
        structure_index = self.structure_selector.value
        if structure_index is None:
            structure_index = 0
        return self._select_cdxml_structure(self._cdxml_content, structure_index)

    def _update_png_preview(self, *, reset_zoom: bool = True) -> None:
        """Show the original selected drawing before creating an atomistic model."""
        set_png_widget(self.png_preview, b"")
        self.preview_message.value = ""
        self.bracket_controls.layout.display = "none"
        self.preview_zoom.disabled = True
        if reset_zoom:
            self._preview_expanded = False
        self.preview_zoom.description = (
            "Reduce preview" if self._preview_expanded else "Enlarge preview"
        )
        self.preview_zoom.icon = (
            "search-minus" if self._preview_expanded else "search-plus"
        )
        self.preview_box.layout.display = "flex"
        try:
            content = self._selected_cdxml_content()
            if (
                self.symmetrize_geometry.value
                and self._conversion_signature is not None
            ):
                content = self._cleaned_drawing(content)
                if any(
                    graphic.get("BracketType") == "Square"
                    for graphic in ET.fromstring(content).iter("graphic")
                ):
                    self.bracket_controls.layout.display = "flex"
            self._preview_cdxml = (
                content.encode("utf-8") if isinstance(content, str) else content
            )
            png, self._preview_layout = render_cdxml_png(
                self._preview_cdxml,
                atom_indices={
                    atom.id: index
                    for index, atom in enumerate(self._source_graph.atoms)
                }
                if self._source_graph is not None and self.show_atom_indices.value
                else None,
                return_layout=True,
            )
            set_png_widget(
                self.png_preview,
                png,
                max_side=900 if self._preview_expanded else 300,
            )
            self._update_atom_picker()
            self.preview_zoom.disabled = False
        except Exception as exc:  # noqa: BLE001
            self.preview_canvas.children = [self.png_preview]
            self.picker_css.value = ""
            self.preview_message.value = (
                f"Sketch preview unavailable: {html.escape(str(exc))}"
            )

    def _cleaned_drawing(self, content: str | bytes) -> bytes:
        """Project the exact prepared geometry back into the drawing's scale."""
        if self.atoms is None:
            raise ValueError("No prepared geometry is available.")
        root = cdxml_root(content)
        nodes = [node for node in root.iter("n") if node.get("p")]
        original = np.array(
            [[*map(float, node.get("p").split()), 0.0] for node in nodes]
        )
        updated = np.asarray(
            self.transform_points(self.atoms.positions, original, self.atoms.positions)
        )
        for node, position in zip(nodes, updated):
            node.set("p", f"{position[0]:.9g} {position[1]:.9g}")
        bracket_graphics = []
        corners = []
        for graphic in root.iter("graphic"):
            if graphic.get("BracketType") != "Square" or not graphic.get("BoundingBox"):
                continue
            coordinates = list(map(float, graphic.get("BoundingBox").split()))
            if len(coordinates) == 4:
                bracket_graphics.append(graphic)
                corners.extend(
                    [
                        [coordinates[0], coordinates[1], 0.0],
                        [coordinates[2], coordinates[3], 0.0],
                    ]
                )
        if corners:
            transformed = np.asarray(
                self.transform_points(original, updated, np.asarray(corners))
            ).reshape((-1, 2, 3))
            if len(bracket_graphics) == 2 and self.crossing_points is not None:
                boundaries = np.asarray(
                    self.transform_points(
                        self.atoms.positions, updated, self.crossing_points
                    )
                )
                centres = transformed.mean(axis=1)
                first, second = linear_sum_assignment(
                    np.linalg.norm(centres[:, None, :] - boundaries[None, :, :], axis=2)
                )
                for graphic_index, boundary_index in zip(first, second):
                    stem = transformed[graphic_index, 1] - transformed[graphic_index, 0]
                    stem_length = np.linalg.norm(stem)
                    if stem_length <= 1.0e-12:
                        continue
                    tangent = stem / stem_length
                    displacement = boundaries[boundary_index] - centres[graphic_index]
                    # A crossing bond fixes the cut line, not the bracket's
                    # midpoint along that line. Preserve the original graphical
                    # extent instead of centring the bracket on a chosen bond.
                    perpendicular = (
                        displacement - np.dot(displacement, tangent) * tangent
                    )
                    transformed[graphic_index] += perpendicular
            atom_indices = {node.get("id"): index for index, node in enumerate(nodes)}
            bond_lengths = []
            carbon_bond_lengths = []
            for bond in root.iter("b"):
                first_index = atom_indices.get(bond.get("B"))
                second_index = atom_indices.get(bond.get("E"))
                if first_index is None or second_index is None:
                    continue
                length = np.linalg.norm(updated[second_index] - updated[first_index])
                if length <= 1.0e-12:
                    continue
                bond_lengths.append(length)
                if (
                    self.atoms[first_index].symbol
                    == self.atoms[second_index].symbol
                    == "C"
                ):
                    carbon_bond_lengths.append(length)
            padding = (
                float(np.median(carbon_bond_lengths or bond_lengths))
                if bond_lengths
                else 0.0
            )
            physical_size = np.linalg.norm(
                self.atoms.positions - self.atoms.positions.mean(axis=0), axis=1
            ).mean()
            drawing_size = np.linalg.norm(updated - updated.mean(axis=0), axis=1).mean()
            drawing_scale = (
                drawing_size / physical_size if physical_size > 1.0e-12 else 0.0
            )
            left_to_right = sorted(
                range(len(transformed)),
                key=lambda index: tuple(transformed[index].mean(axis=0)[:2]),
            )
            sides = {
                index: "left" if rank == 0 else "right"
                for rank, index in enumerate(left_to_right)
            }
            for index, (graphic, pair) in enumerate(zip(bracket_graphics, transformed)):
                stem = pair[1] - pair[0]
                stem_length = np.linalg.norm(stem)
                if stem_length > 1.0e-12:
                    tangent = stem / stem_length
                    projections = updated @ tangent
                    pair[0] += (
                        projections.min() - padding - np.dot(pair[0], tangent)
                    ) * tangent
                    pair[1] += (
                        projections.max() + padding - np.dot(pair[1], tangent)
                    ) * tangent
                    normal = normalize(np.array([tangent[1], -tangent[0], 0.0]))
                    if normal[0] < 0 or (abs(normal[0]) < 1.0e-12 and normal[1] < 0):
                        normal *= -1
                    pair += (
                        normal * self.bracket_shifts[sides[index]].value * drawing_scale
                    )
                graphic.set(
                    "BoundingBox",
                    " ".join(f"{value:.9g}" for value in pair[:, :2].ravel()),
                )
        return ET.tostring(root, encoding="utf-8")

    def _reset_bracket_drawing_shift(self) -> None:
        self._updating_bracket_shift = True
        try:
            for offset in self.bracket_shifts.values():
                offset.value = 0.0
        finally:
            self._updating_bracket_shift = False

    def _on_bracket_shift_change(self, _=None) -> None:
        if not self._updating_bracket_shift and self._cdxml_content is not None:
            self._update_png_preview(reset_zoom=False)

    def _on_bracket_step_change(self, _=None) -> None:
        for offset in self.bracket_shifts.values():
            offset.step = self.bracket_shift_step.value

    def _shift_bracket_drawing(self, side: str, direction: int) -> None:
        offset = self.bracket_shifts[side]
        offset.value = round(
            offset.value + direction * self.bracket_shift_step.value, 12
        )

    def _on_geometry_option_change(self, _=None) -> None:
        if self._cdxml_content is None:
            return
        self.structure = None
        self._convert_uploaded_cdxml()
        self._update_png_preview(reset_zoom=False)

    def _update_steric_option(self) -> None:
        self.resolve_steric_collisions.disabled = (
            self._conversion_signature is None
            or not self.nunits.disabled
            or not self.use_clever_hydrogenation.value
        )
        self.steric_help.layout.display = (
            "none" if self.resolve_steric_collisions.disabled else ""
        )
        active = (
            not self.resolve_steric_collisions.disabled
            and self.resolve_steric_collisions.value
        )
        self.steric_controls.layout.display = "" if active else "none"
        self.minimum_contacts_box.layout.display = (
            "" if active and self.steric_mode.value == "minimal" else "none"
        )
        self.steric_help.value = (
            "Minimal 3D torsion keeps nonbonded atoms beyond the selected distances. "
            "It can leave van der Waals overlaps; relax this starting conformer. "
            "The sketch stays 2D and rings remain rigid."
            if active and self.steric_mode.value == "minimal"
            else "Finite C/H/halogen molecules: rotate connected aromatic blocks and "
            "methyl hydrogens. Steric separation uses 75% of summed van der Waals "
            "radii. The sketch stays 2D; no energy optimization."
        )

    def _on_steric_option_change(self, _=None) -> None:
        self.structure = None
        self._update_steric_option()

    def _on_hydrogenation_option_change(self, _=None) -> None:
        self.structure = None
        self._update_steric_option()

    def _toggle_preview_size(self, _=None) -> None:
        self._preview_expanded = not self._preview_expanded
        set_png_widget(
            self.png_preview,
            bytes(self.png_preview.value),
            max_side=900 if self._preview_expanded else 300,
        )
        self._update_atom_picker()
        self.preview_zoom.description = (
            "Reduce preview" if self._preview_expanded else "Enlarge preview"
        )
        self.preview_zoom.icon = (
            "search-minus" if self._preview_expanded else "search-plus"
        )

    def _pick_atom(self, index, source_token=None):
        if source_token is not None and source_token != (
            hashlib.sha256(self._cdxml_content).hexdigest(),
            self.structure_selector.value,
        ):
            return
        if self.equivalence_controls.layout.display == "none":
            return
        self.equivalent_atoms[self._pair_pick_next].value = index
        self._pair_pick_next = (self._pair_pick_next + 1) % 2
        self.structure = None
        self._pair_dirty = True
        self.create_button.disabled = True

    def _update_atom_picker(self):
        """Overlay standard widget buttons at the rendered atom positions."""
        if not self._preview_layout or not self.png_preview.value:
            return
        width, height = int(self.png_preview.width), int(self.png_preview.height)
        self.preview_canvas.layout.width = f"{width}px"
        self.preview_canvas.layout.height = f"{height}px"
        children = [self.png_preview]
        for old in self.preview_canvas.children[1:]:
            old.close()
        styles = [f".{self._picker_class} {{ position: relative; flex: none; }}"]
        if (
            self.show_atom_indices.value
            and self._source_graph is not None
            and self.equivalence_controls.layout.display != "none"
        ):
            low = np.asarray(self._preview_layout["low"])
            span = np.asarray(self._preview_layout["high"]) - low
            source_token = (
                hashlib.sha256(self._cdxml_content).hexdigest(),
                self.structure_selector.value,
            )
            for identifier, point in self._preview_layout["positions"].items():
                if identifier not in self._source_graph.index:
                    continue
                index = self._source_graph.index[identifier]
                x, y = (np.asarray(point) - low) / span * [width, height]
                button = ipw.Button(
                    description="",
                    tooltip=f"Select atom {index} ({self._source_graph.atoms[index].symbol})",
                    layout={
                        "width": "16px",
                        "height": "16px",
                        "min_width": "16px",
                        "padding": "0",
                    },
                )
                marker = f"{self._picker_class}-atom-{index}"
                button.add_class(marker)
                button.on_click(
                    lambda _, selected=index, token=source_token: self._pick_atom(
                        selected, token
                    )
                )
                styles.extend(
                    (
                        (
                            f".{marker} {{ position: absolute; left: {x - 8:.3f}px; top: {y - 8:.3f}px; "
                            "background: transparent; border: 1px solid transparent; border-radius: 50%; z-index: 8; }"
                        ),
                        f".{marker}:hover {{ border-color: #135d9e; background: rgba(19,93,158,0.16); }}",
                    )
                )
                children.append(button)
        self.preview_canvas.children = children
        self.picker_css.value = "<style>" + "\n".join(styles) + "</style>"

    def _on_structure_selection(self, change=None) -> None:
        """Reconvert the document when the selected structure changes."""
        if self._updating_structure_selector or self._cdxml_content is None:
            return
        if change is not None and change.get("new") is None:
            return
        self._reset_bracket_drawing_shift()
        self.structure = None
        self._equivalent_atom_ids = None
        self._pair_dirty = False
        self._pair_pick_next = 0
        self.nunits.value = "Infinite"
        self.nunits.disabled = True
        self._convert_uploaded_cdxml()
        self._update_png_preview()

    def _convert_uploaded_cdxml(self) -> bool:
        """Convert the stored CDXML with the currently selected geometry options."""
        if self._cdxml_content is None:
            self.output_message.value = "Error: No CDXML file has been uploaded."
            return False
        try:
            selected_content = self._selected_cdxml_content()
            signature = (
                selected_content,
                self.symmetrize_geometry.value,
                self._equivalent_atom_ids,
            )
            if signature == self._conversion_signature:
                return True
            self._conversion_signature = None
            self.create_button.disabled = True
            self.atoms = self.whole_atoms = self.cdxml_atoms = self.crossing_points = (
                None
            )
            self._source_graph = read_cdxml_graph(selected_content)
            # Even an ambiguous repeat has a valid finite drawing to inspect.
            self._geometry_message, self.atoms, self.whole_atoms = (
                self.cdxml_to_ase_from_string(selected_content)
            )
            self._updating_pair = True
            try:
                for field in self.equivalent_atoms:
                    field.max = max(0, len(self._source_graph.atoms) - 1)
            finally:
                self._updating_pair = False
            self._periodic_repeat = identify_repeat(
                selected_content, self._source_graph, self._equivalent_atom_ids
            )
            self.equivalence_controls.layout.display = (
                "" if self._periodic_repeat is not None else "none"
            )
            if self._periodic_repeat is None:
                self.show_atom_indices.value = False
            self._geometry_message, self.atoms, self.whole_atoms = (
                self.cdxml_to_ase_from_string(
                    selected_content,
                    symmetrize=self.symmetrize_geometry.value,
                    periodic_repeat=self._periodic_repeat,
                )
            )
            if (
                self._periodic_repeat is not None
                and self._periodic_repeat.resonance_aligned
            ):
                self._geometry_message += " Conjugated boundary bond orders aligned for the periodic repeat; H per source site preserved."
            self.cdxml_atoms = self.atoms.positions.copy()
            if self._periodic_repeat is not None:
                central = set(self._periodic_repeat.atom_ids)
                left = []
                images = {
                    image: shift for image, _, shift in self._periodic_repeat.images
                }
                for bond in self._source_graph.bonds:
                    inner, outer = (
                        (bond.begin, bond.end)
                        if bond.begin in central
                        else (bond.end, bond.begin)
                    )
                    if inner in central and images.get(outer) == -1:
                        left.append(
                            (
                                self.atoms.positions[self._source_graph.index[inner]]
                                + self.atoms.positions[self._source_graph.index[outer]]
                            )
                            / 2
                        )
                first = np.mean(left, axis=0)
                vector = self._periodic_repeat.vector_for_positions(
                    self._source_graph, self.atoms.positions
                )
                self.crossing_points = np.asarray([first, first + vector])
            else:
                self.crossing_points = None
            self.nunits.disabled = self._periodic_repeat is None
            self._conversion_signature = signature
            self.create_button.disabled = self._pair_dirty
            self._update_steric_option()
            self.output_message.value = (
                f"Ready to create the structure. {self._geometry_message}"
            )
            return True
        except PeriodicAmbiguity as exc:
            self._periodic_repeat = None
            self.equivalence_controls.layout.display = ""
            self.nunits.disabled = False
            self.show_atom_indices.value = True
            indices = (
                sorted(
                    {
                        self._source_graph.index[identifier]
                        for identifier in exc.atom_ids
                        if identifier in self._source_graph.index
                    }
                )
                if self._source_graph is not None
                else []
            )
            detail = (
                f" Check atom indices {', '.join(map(str, indices[:12]))}."
                if indices
                else ""
            )
            self.output_message.value = (
                f"Periodic ambiguity: {html.escape(str(exc))}{detail}"
            )
        except ValueError as exc:
            self.output_message.value = f"Error: {html.escape(str(exc))}"
        # A file-upload callback should report unexpected malformed input in the
        # widget instead of interrupting the notebook event loop.
        except Exception as exc:  # noqa: BLE001
            self.output_message.value = f"Unexpected error: {html.escape(str(exc))}"
        self._update_steric_option()
        self.create_button.disabled = True
        return False

    def _on_file_upload(self, change=None) -> None:
        """Handle file upload and convert CDXML to ASE Atoms."""
        upload_value = self.file_upload.value
        if change is not None:
            upload_value = change.get("new", upload_value)
        if not upload_value:
            return

        self.nunits.value = "Infinite"
        self.nunits.disabled = True

        def get_unified_representation(value):
            """Return name/content pairs for ipywidgets 7 and 8 payloads."""
            if isinstance(value, dict):
                return [
                    (filename, bytes(item["content"]))
                    for filename, item in value.items()
                ]

            files = []
            for item in value:
                if isinstance(item, dict):
                    files.append((item["name"], bytes(item["content"])))
                else:
                    files.append((item["name"], item.content.tobytes()))
            return files

        self._cdxml_filename, self._cdxml_content = get_unified_representation(
            upload_value
        )[0]
        self._equivalent_atom_ids = None
        self._pair_dirty = False
        self._source_graph = None
        self._conversion_signature = None
        self._reset_bracket_drawing_shift()
        self.structure = None
        try:
            self._configure_structure_selector()
        except Exception:  # noqa: BLE001
            # Conversion below reports malformed CDXML consistently in the widget.
            self._updating_structure_selector = True
            try:
                self.structure_selector.options = ()
                self.structure_selector.disabled = True
                self.structure_selector.layout.display = "none"
            finally:
                self._updating_structure_selector = False
        self._convert_uploaded_cdxml()
        self._update_png_preview()

    def _on_button_click(self, _=None) -> None:
        """Create the ASE model when 'Create model' button is clicked."""
        if self._pair_dirty:
            self.structure = None
            self.output_message.value = (
                "Apply the selected atom pair before creating the model."
            )
            return
        if not self._convert_uploaded_cdxml():
            return
        if self.atoms is None or len(self.atoms) == 0:
            self.output_message.value = "Error: No atoms available to process."
            return

        atoms = self.atoms.copy()
        source_atoms = (
            self.whole_atoms if self.use_clever_hydrogenation.value else self.atoms
        )
        if source_atoms is None:
            self.output_message.value = "Error: Converted atoms are unavailable."
            return
        steric_message = ""

        if self.nunits.disabled:
            # A finite molecule has no periodic cut. Keep every explicit and
            # bond-order-derived H, even when its initial drawing is crowded.
            atoms = source_atoms.copy()
            bonds = atoms.info.pop("_cdxml_bonds", [])
            stereochemistry = atoms.info.pop("_cdxml_stereochemistry", False)
            if bonds:
                resolve = (
                    self.resolve_steric_collisions.value
                    and self.use_clever_hydrogenation.value
                )
                if resolve and stereochemistry:
                    steric_message = (
                        "3D relief skipped: the drawing encodes stereochemistry; "
                        "all atoms retained. Inspect the model."
                    )
                else:
                    try:
                        result = relieve_steric_contacts(
                            atoms,
                            bonds,
                            resolve=resolve,
                            mode=self.steric_mode.value if resolve else "steric",
                            minimum_distances={
                                pair: field.value
                                for pair, field in self.minimum_contact_distances.items()
                            },
                            margin_percent=self.steric_margin_percent.value,
                        )
                    except ValueError as exc:
                        self.structure = None
                        self.output_message.value = f"Error: {html.escape(str(exc))}"
                        return
                    atoms = result.atoms
                    if resolve or result.initial_contacts:
                        steric_message = result.message
        elif self.crossing_points is not None:
            if self.cdxml_atoms is None:
                self.output_message.value = (
                    "Error: CDXML atom positions are unavailable."
                )
                return
            try:
                atoms = build_repeat_model(
                    source_atoms,
                    self._source_graph,
                    self._periodic_repeat,
                    self.nunits.value,
                )
            except ValueError as exc:
                self.structure = None
                self.output_message.value = f"Error: {html.escape(str(exc))}"
                return
        else:
            self.output_message.value = "Error: No periodic boundaries found."
            return

        if self.nunits.disabled:
            extra_cell = 15.0
            atoms.cell = np.ptp(atoms.positions, axis=0) + extra_cell
            atoms.center()
            atoms.pbc = False

        if not self.nunits.disabled and self.nunits.value == "Infinite":
            atoms.pbc = [True, False, False]

        messages = ["Structure created.", self._geometry_message]
        if steric_message:
            messages.append(html.escape(steric_message))
        if not self.use_clever_hydrogenation.value:
            messages.append(
                "Implicit hydrogens omitted; only explicitly drawn H atoms retained."
            )
        self.output_message.value = " ".join(messages)

        atoms.info.pop("_cdxml_bonds", None)
        atoms.info.pop("_cdxml_stereochemistry", None)
        from .cdxml_storage import model_fingerprint

        generation = atoms.info.setdefault("cdxml_generation", {})
        atom_sources = atoms.info.pop("_cdxml_atom_sources", None)
        if atom_sources is not None:
            generation["atom_sources"] = atom_sources
        generation.update(
            {
                "cleanup": bool(atoms.info.pop("_cdxml_cleanup_applied", False)),
                "cleanup_requested": self.symmetrize_geometry.value,
                "implicit_hydrogens_included": self.use_clever_hydrogenation.value,
                "steric_mode": self.steric_mode.value
                if self.resolve_steric_collisions.value
                and not self.resolve_steric_collisions.disabled
                else None,
                "formal_charge": generation.get(
                    "formal_charge",
                    sum(atom.charge for atom in self._source_graph.atoms),
                ),
                "model_hash": model_fingerprint(atoms),
                "stereochemistry_reconstructed": False,
                "equivalent_atom_pair": list(self._equivalent_atom_ids)
                if self._equivalent_atom_ids
                else None,
            }
        )
        formula = atoms.get_chemical_formula()
        charge = generation["formal_charge"]
        self.output_message.value += f" Formula: {formula}; formal charge: {charge:+d}."
        if charge:
            self.output_message.value += (
                " Set the calculation charge explicitly; spin is not inferred."
            )
        self.structure = atoms

    @staticmethod
    def add_safe_hydrogen_atoms(atoms: Atoms) -> tuple[str, Atoms]:
        """Add one H to C atoms with fewer than three neighbors."""
        neighbor_list = NeighborList(
            [covalent_radii[atom.number] for atom in atoms],
            bothways=True,
            self_interaction=False,
        )
        neighbor_list.update(atoms)

        need_hydrogen = [
            atom.index
            for atom in atoms
            if atom.symbol == "C"
            and len(neighbor_list.get_neighbors(atom.index)[0]) < 3
        ]

        for index in need_hydrogen:
            vec = np.zeros(3)
            indices, offsets = neighbor_list.get_neighbors(atoms[index].index)
            for i, offset in zip(indices, offsets):
                vec += -atoms[index].position + (
                    atoms.positions[i] + np.dot(offset, atoms.get_cell())
                )
            vec_norm = np.linalg.norm(vec)
            if vec_norm > 1e-12:
                position = -vec / vec_norm * 1.1 + atoms[index].position
            else:
                position = atoms[index].position + np.array([0.0, 0.0, 1.1])
            atoms.append(ase.Atom("H", position))

        return (
            f"Added missing Hydrogen atoms (safe hydrogenation): {need_hydrogen}.",
            atoms,
        )

    # ---------------- Core conversion logic ----------------
    @staticmethod
    def symmetrize_carbon_network(
        positions: np.ndarray,
        symbols: list[str],
        bonds: list[tuple[int, int]],
        target_cc_length: float = 1.43,
        minimum_cc_length: float = 1.35,
        maximum_cc_length: float = 1.60,
        periodic_equivalences: tuple[tuple[int, int, int], ...] = (),
        period_vector: np.ndarray | None = None,
    ) -> np.ndarray:
        """Relax a carbon drawing while preserving its planar graph embedding.

        The local sp2 preference is deliberately soft. Ring closure and fused
        edges therefore distribute strain globally instead of forcing each
        5-, 6-, or 7-membered face to become an independent regular polygon.
        """
        positions = np.asarray(positions, dtype=float)
        carbon_indices = [
            index for index, symbol in enumerate(symbols) if symbol == "C"
        ]
        local_index = {
            atom_index: carbon_index
            for carbon_index, atom_index in enumerate(carbon_indices)
        }
        carbon_edges = [
            (local_index[first], local_index[second])
            for first, second in bonds
            if first in local_index and second in local_index
        ]
        if len(carbon_edges) < 3:
            return positions.copy()

        all_edges = [(first, second) for first, second in bonds]
        if _has_bond_crossings(positions[:, :2], all_edges):
            raise ValueError("the input bonds already contain a geometric crossing")
        all_faces = [
            (face, _signed_area(positions[face, :2]))
            for face in _bounded_faces(positions[:, :2], all_edges)
            if abs(_signed_area(positions[face, :2])) > 1.0e-8
        ]

        original = positions[carbon_indices, :2].copy()
        if _has_bond_crossings(original, carbon_edges):
            raise ValueError(
                "the input carbon bonds already contain a geometric crossing"
            )

        adjacency = [[] for _ in range(len(original))]
        for first, second in carbon_edges:
            adjacency[first].append(second)
            adjacency[second].append(first)

        angle_terms = []
        for center, neighbors in enumerate(adjacency):
            if len(neighbors) not in (2, 3):
                continue
            for first_index in range(len(neighbors)):
                for second_index in range(first_index + 1, len(neighbors)):
                    angle_terms.append(
                        (center, neighbors[first_index], neighbors[second_index])
                    )

        faces = _bounded_faces(original, carbon_edges)
        face_areas = [
            (face, _signed_area(original[face]))
            for face in faces
            if abs(_signed_area(original[face])) > 1.0e-8
        ]

        # Parameterize equivalent image atoms exactly as p_image=p_site+shift*T.
        # Cleanup can change T, but cannot choose different periodic sites.
        parent = list(range(len(original)))
        image_shift = [0] * len(original)

        def find(index):
            if parent[index] == index:
                return index, 0
            root, shift = find(parent[index])
            image_shift[index] += shift
            parent[index] = root
            return root, image_shift[index]

        periodic = bool(periodic_equivalences)
        initial_period = np.asarray(period_vector)[:2] if periodic else np.zeros(2)
        for image, central, shift in periodic_equivalences:
            if image not in local_index or central not in local_index:
                if symbols[image] != "H":
                    raise ValueError(
                        "periodic cleanup of noncarbon sites is not supported"
                    )
                continue
            first, second = local_index[image], local_index[central]
            first_root, first_shift = find(first)
            second_root, second_shift = find(second)
            if first_root == second_root:
                if first_shift - second_shift != shift:
                    raise ValueError("inconsistent periodic image equivalences")
            else:
                parent[first_root] = second_root
                image_shift[first_root] = second_shift + shift - first_shift
        roots_and_shifts = [find(index) for index in range(len(original))]
        roots = sorted({root for root, _ in roots_and_shifts})
        root_index = {root: index for index, root in enumerate(roots)}
        mapping = np.asarray([root_index[root] for root, _ in roots_and_shifts])
        shifts = np.asarray([shift for _, shift in roots_and_shifts])
        initial_sites = np.asarray(
            [
                np.mean(
                    [
                        original[index] - shifts[index] * initial_period
                        for index in range(len(original))
                        if mapping[index] == site
                    ],
                    axis=0,
                )
                for site in range(len(roots))
            ]
        )
        initial_variables = initial_sites.ravel()
        if periodic:
            initial_variables = np.r_[initial_variables, initial_period]

        def expand(variables):
            sites = variables[: 2 * len(roots)].reshape((-1, 2))
            vector = variables[-2:] if periodic else initial_period
            return sites[mapping] + shifts[:, None] * vector

        def residual(flat_positions):
            current = expand(flat_positions)
            values: list[float] = [
                float(
                    (
                        np.linalg.norm(current[second] - current[first])
                        - target_cc_length
                    )
                    / 0.04
                )
                for first, second in carbon_edges
            ]
            for center, first, second in angle_terms:
                first_vector = current[first] - current[center]
                second_vector = current[second] - current[center]
                denominator = np.linalg.norm(first_vector) * np.linalg.norm(
                    second_vector
                )
                cosine = np.dot(first_vector, second_vector) / max(denominator, 1.0e-12)
                values.append(float(math.sqrt(0.10) * (cosine + 0.5) / 0.15))

            # Keep the user's global layout while allowing local cleanup.
            values.extend((math.sqrt(0.10) * (current - original) / 0.20).ravel())
            if periodic:
                values.extend(
                    (
                        math.sqrt(0.10) * (flat_positions[-2:] - initial_period) / 0.25
                    ).ravel()
                )

            # This barrier protects every detected face from collapse or inversion.
            for face, original_area in face_areas:
                area_ratio = _signed_area(current[face]) / original_area
                values.append(math.sqrt(20.0) * max(0.0, 0.35 - area_ratio))
            return np.asarray(values)

        maximum_displacement = 0.75
        result = least_squares(
            residual,
            initial_variables,
            bounds=(
                initial_variables - maximum_displacement,
                initial_variables + maximum_displacement,
            ),
            max_nfev=200,
            ftol=1.0e-9,
            xtol=1.0e-9,
            gtol=1.0e-9,
        )
        if not result.success:
            raise ValueError(f"geometry optimizer did not converge: {result.message}")

        optimized = expand(result.x)
        lengths = np.array(
            [
                np.linalg.norm(optimized[second] - optimized[first])
                for first, second in carbon_edges
            ]
        )
        if (
            np.min(lengths) < minimum_cc_length - 1.0e-6
            or np.max(lengths) > maximum_cc_length + 1.0e-6
        ):
            raise ValueError(
                "optimized C-C lengths fall outside "
                f"{minimum_cc_length:.2f}-{maximum_cc_length:.2f} Å"
            )
        if _has_bond_crossings(optimized, carbon_edges):
            raise ValueError("geometry cleanup would create crossing carbon bonds")

        for face, original_area in face_areas:
            optimized_area = _signed_area(optimized[face])
            if optimized_area * original_area <= 0.0 or abs(
                optimized_area
            ) < 0.25 * abs(original_area):
                raise ValueError(
                    "geometry cleanup would invert or collapse a carbon face"
                )

        updated = positions.copy()
        carbon_displacements = optimized - original
        updated[carbon_indices, :2] = optimized

        # Extend the carbon displacement harmonically through attached heteroatom
        # groups. An explicit N-H must follow its N, including in fused rings.
        adjacency_all = {index: [] for index in range(len(positions))}
        for first, second in bonds:
            adjacency_all[first].append(second)
            adjacency_all[second].append(first)
        attached = set(carbon_indices)
        pending = list(carbon_indices)
        while pending:
            for neighbor in adjacency_all[pending.pop()]:
                if neighbor not in attached:
                    attached.add(neighbor)
                    pending.append(neighbor)
        movable = sorted(attached.difference(carbon_indices))
        movable_index = {index: local for local, index in enumerate(movable)}
        if movable:
            laplacian = np.zeros((len(movable), len(movable)))
            right_hand_side = np.zeros((len(movable), 2))
            for atom_index, row in movable_index.items():
                laplacian[row, row] = len(adjacency_all[atom_index])
                for neighbor in adjacency_all[atom_index]:
                    if neighbor in movable_index:
                        laplacian[row, movable_index[neighbor]] -= 1
                    else:
                        right_hand_side[row] += carbon_displacements[
                            local_index[neighbor]
                        ]
            updated[movable, :2] += np.linalg.solve(laplacian, right_hand_side)
        if _has_bond_crossings(updated[:, :2], all_edges):
            raise ValueError("geometry cleanup would create crossing bonds")
        for face, original_area in all_faces:
            area = _signed_area(updated[face, :2])
            if area * original_area <= 0 or abs(area) < 0.25 * abs(original_area):
                raise ValueError("geometry cleanup would invert or collapse a face")

        return updated

    @staticmethod
    def cdxml_to_ase_from_string(
        cdxml_content: str | bytes,
        target_cc_length: float = 1.43,
        symmetrize: bool = False,
        periodic_repeat: RepeatUnit | None = None,
    ) -> tuple[str, Atoms, Atoms]:
        """
        Convert CDXML content (string) into ASE Atoms objects:
        one bare molecule (no hydrogens) and one with hydrogens.
        """
        graph = read_cdxml_graph(cdxml_content)
        impl_H = graph.added_hydrogens()
        atoms = {
            atom.id: {"el": atom.symbol, "pos": np.asarray(atom.position)}
            for atom in graph.atoms
        }
        bonds = [
            {"a1": bond.begin, "a2": bond.end, "order": bond.order, "type": bond.type}
            for bond in graph.bonds
        ]

        # --- Scale from explicit C-C bonds, never from unrelated close atoms ---
        cc_lengths = [
            np.linalg.norm(atoms[bond["a1"]]["pos"] - atoms[bond["a2"]]["pos"])
            for bond in bonds
            if atoms[bond["a1"]]["el"] == atoms[bond["a2"]]["el"] == "C"
            and bond["type"] != "dative"
        ]
        if cc_lengths:
            scale = target_cc_length / np.median(cc_lengths)
        else:
            from ase.data import atomic_numbers

            ratios = [
                (
                    covalent_radii[atomic_numbers[atoms[bond["a1"]]["el"]]]
                    + covalent_radii[atomic_numbers[atoms[bond["a2"]]["el"]]]
                )
                / np.linalg.norm(atoms[bond["a1"]]["pos"] - atoms[bond["a2"]]["pos"])
                for bond in bonds
            ]
            scale = float(np.median(ratios)) if ratios else 1.0
        for atom in atoms.values():
            atom["pos"] *= scale

        geometry_message = "Original CDXML geometry retained."
        cleanup_applied = False
        if symmetrize and graph.stereochemistry:
            geometry_message = (
                "⚠️ Carbon geometry cleanup skipped: stereochemical drawing. "
                "Original scaled geometry retained."
            )
        elif symmetrize and len(cc_lengths) < 3:
            geometry_message = (
                "Carbon geometry cleanup not applicable: fewer than three C-C bonds."
            )
        elif symmetrize:
            atom_ids = list(atoms)
            atom_index = {atom_id: index for index, atom_id in enumerate(atom_ids)}
            positions = np.array([atoms[atom_id]["pos"] for atom_id in atom_ids])
            symbols = [atoms[atom_id]["el"] for atom_id in atom_ids]
            edge_indices = [
                (atom_index[bond["a1"]], atom_index[bond["a2"]]) for bond in bonds
            ]
            try:
                positions = CdxmlUploadWidget.symmetrize_carbon_network(
                    positions,
                    symbols,
                    edge_indices,
                    target_cc_length=target_cc_length,
                    periodic_equivalences=tuple(
                        (graph.index[image], graph.index[central], shift)
                        for image, central, shift in periodic_repeat.images
                    )
                    if periodic_repeat is not None
                    else (),
                    period_vector=np.asarray(periodic_repeat.vector) * scale
                    if periodic_repeat is not None
                    else None,
                )
            except ValueError as exc:
                geometry_message = (
                    "⚠️ Carbon geometry cleanup skipped: "
                    f"{exc}. Original scaled geometry retained."
                )
            else:
                cleanup_applied = True
                for atom_id, position in zip(atom_ids, positions):
                    atoms[atom_id]["pos"] = position
                geometry_message = (
                    "Carbon geometry cleaned globally; planar embedding preserved "
                    "and C-C bonds constrained to 1.35-1.60 Å."
                )
        if periodic_repeat is not None:
            # Neighbors used to place H must be the identified periodic copies,
            # including when cleanup is disabled and the original sketch is noisy.
            ids = graph.index
            current_positions = np.asarray(
                [atoms[atom.id]["pos"] for atom in graph.atoms]
            )
            translation = periodic_repeat.vector_for_positions(graph, current_positions)
            before_alignment = {
                identifier: atom["pos"].copy() for identifier, atom in atoms.items()
            }
            for image, central, shift in periodic_repeat.images:
                if graph.atoms[ids[image]].number != 1:
                    atoms[image]["pos"] = atoms[central]["pos"] + shift * translation
            for spec in graph.atoms:
                if spec.number == 1:
                    neighbors = [
                        bond.end if bond.begin == spec.id else bond.begin
                        for bond in graph.bonds
                        if spec.id in {bond.begin, bond.end}
                    ]
                    if len(neighbors) == 1:
                        parent = neighbors[0]
                        atoms[spec.id]["pos"] += (
                            atoms[parent]["pos"] - before_alignment[parent]
                        )
            for image, central, shift in periodic_repeat.images:
                if graph.atoms[ids[image]].number == 1:
                    atoms[image]["pos"] = atoms[central]["pos"] + shift * translation

        # --- Connectivity ---
        conn = {k: [] for k in atoms}
        for b in bonds:
            conn[b["a1"]].append((b["a2"], b["order"]))
            conn[b["a2"]].append((b["a1"], b["order"]))

        # --- Bare molecule ---
        bare_pos = [a["pos"] for a in atoms.values()]
        bare_sym = [a["el"] for a in atoms.values()]
        bare_mol = Atoms(symbols=bare_sym, positions=bare_pos)

        # --- Add hydrogens ---
        pos, sym = list(bare_pos), list(bare_sym)
        atom_sources = [{"source_id": atom.id} for atom in graph.atoms]
        atom_index = {aid: index for index, aid in enumerate(atoms)}
        chemical_bonds = [
            (atom_index[b["a1"]], atom_index[b["a2"]], b["order"]) for b in bonds
        ]
        for aid, nH in impl_H.items():
            if nH == 0:
                continue

            el = atoms[aid]["el"]
            c = atoms[aid]["pos"]
            neighbors = [normalize(atoms[n]["pos"] - c) for n, _ in conn[aid]]
            orders = [o for _, o in conn[aid]]

            def add_H(
                vecs: list[np.ndarray],
                length: float = 1.09,
                center: np.ndarray = c,
                parent: int = atom_index[aid],
            ):
                for v in vecs:
                    chemical_bonds.append((parent, len(pos), 1.0))
                    pos.append(center + length * v)
                    sym.append("H")
                    atom_sources.append({"hydrogen_parent_id": aid})

            # Isolated methane/ammonium and unbound radical fragments require
            # actual H positions too, rather than only a correct H count.
            if not neighbors:
                if nH == 4:
                    directions = np.asarray(
                        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                        dtype=float,
                    ) / math.sqrt(3)
                elif nH == 3:
                    height = 1 / 3 if el == "N" else 0.0
                    radius = math.sqrt(1 - height**2)
                    directions = [
                        np.array(
                            [radius * math.cos(phi), radius * math.sin(phi), height]
                        )
                        for phi in (0, 2 * math.pi / 3, 4 * math.pi / 3)
                    ]
                elif el == "C" and nH == 2:
                    directions = [
                        np.array([0.5, sign * math.sqrt(3) / 2, 0]) for sign in (-1, 1)
                    ]
                elif nH == 1:
                    directions = [np.array([1.0, 0.0, 0.0])]
                else:
                    directions = None
                if directions is not None:
                    length = {
                        "C": 1.09,
                        "N": 1.00,
                        "O": 0.98,
                        "B": 1.19,
                        "Si": 1.48,
                        "P": 1.42,
                        "S": 1.34,
                    }.get(
                        el,
                        covalent_radii[graph.atoms[atom_index[aid]].number]
                        + covalent_radii[1],
                    )
                    add_H(directions, length)
                    continue

            # NH3 attached by a single bond (e.g. methylammonium) shares the
            # tetrahedral construction used for terminal carbon methyl groups.
            if el == "N" and nH == 3 and len(neighbors) == 1:
                axis = neighbors[0]
                perpendicular = normalize(np.cross(axis, [0, 0, 1]))
                if np.linalg.norm(perpendicular) < 1e-8:
                    perpendicular = np.array([1.0, 0.0, 0.0])
                other = np.cross(axis, perpendicular)
                add_H(
                    [
                        -axis / 3
                        + math.sqrt(8 / 9)
                        * (math.cos(phi) * perpendicular + math.sin(phi) * other)
                        for phi in (0, 2 * math.pi / 3, 4 * math.pi / 3)
                    ],
                    1.00,
                )
                continue

            # --- Oxygen or Nitrogen (improved geometry) ---
            if el in ("O", "N"):
                v_sum = np.sum(neighbors, axis=0) if neighbors else np.zeros(3)
                base_dir = (
                    normalize(-v_sum)
                    if np.linalg.norm(v_sum) > 1e-6
                    else np.array([0, 0, 1])
                )

                # --- Single hydrogen (OH, NH) ---
                if nH == 1:
                    if el == "O" and len(neighbors) == 1:
                        # Rotate 75 degrees from the opposite direction: C-O-H=105 degrees.
                        v = neighbors[0]
                        # Keep hydrogen roughly in the same molecular plane
                        rot_axis = np.array([0, 0, 1])
                        if np.allclose(np.abs(np.dot(v, rot_axis)), 1.0):
                            rot_axis = np.array([1, 0, 0])
                        dirs = [rotate_vector(-v, rot_axis, math.radians(75))]
                        add_H(dirs, 0.98)
                    else:
                        # Default for NH etc.
                        add_H([base_dir], 1.00 if el == "N" else 0.98)
                    continue

                # --- Two hydrogens (H2O, NH2) ---
                if nH == 2:
                    theta = math.radians(104.5 if el == "O" else 107.0)
                    rot_axis = np.array([0, 0, 1])
                    if np.allclose(np.abs(np.dot(base_dir, rot_axis)), 1.0):
                        rot_axis = np.array([1, 0, 0])
                    dirs = [
                        rotate_vector(base_dir, rot_axis, a)
                        for a in (-theta / 2, theta / 2)
                    ]
                    add_H(dirs, 0.98 if el == "O" else 1.00)
                    continue

            # --- Other heteroatoms (unchanged behaviour) ---
            if el != "C":
                avg = (
                    normalize(-np.sum(neighbors, axis=0))
                    if neighbors
                    else np.array([0, 0, 1])
                )
                if np.linalg.norm(avg) < 1e-8:
                    avg = np.array([0.0, 0.0, 1.0])
                if nH == 1:
                    directions = [avg]
                else:
                    # An explicit multi-H heteroatom label must retain every H.
                    tangent = normalize(np.cross(avg, [0, 0, 1]))
                    if np.linalg.norm(tangent) < 1e-8:
                        tangent = np.array([1.0, 0.0, 0.0])
                    other = np.cross(avg, tangent)
                    directions = [
                        normalize(avg + math.cos(phi) * tangent + math.sin(phi) * other)
                        for phi in np.linspace(0, 2 * math.pi, nH, endpoint=False)
                    ]
                from ase.data import atomic_numbers

                length = {"B": 1.19, "Si": 1.48, "P": 1.42, "S": 1.34}.get(
                    el, covalent_radii[atomic_numbers[el]] + covalent_radii[1]
                )
                add_H(directions, length)
                continue

            # --- Carbon atoms (original logic, untouched) ---

            # CH3
            if nH == 3 and len(neighbors) == 1:
                v = neighbors[0]
                theta = math.radians(109.47)
                dirs = [
                    np.array(
                        [
                            math.sin(theta) * math.cos(p),
                            math.sin(theta) * math.sin(p),
                            math.cos(theta),
                        ]
                    )
                    for p in (0, 2 * math.pi / 3, 4 * math.pi / 3)
                ]
                R = rotation_matrix_from_vectors(np.array([0, 0, 1]), -v)
                add_H([-R @ d for d in dirs], 1.10)
                continue

            # CH2
            if nH == 2:
                CH_len = 1.09 if any(o >= 1.5 for o in orders) else 1.10
                if len(neighbors) == 1:
                    v = neighbors[0]
                    if any(o >= 1.5 for o in orders):
                        normal = np.array([0, 0, 1])
                        bis = -v
                        add_H(
                            [
                                rotate_vector(bis, normal, math.radians(a))
                                for a in (60, -60)
                            ],
                            CH_len,
                        )
                    else:
                        theta = math.radians(109.47)
                        dirs = [
                            np.array(
                                [
                                    math.sin(theta) * math.cos(p),
                                    math.sin(theta) * math.sin(p),
                                    math.cos(theta),
                                ]
                            )
                            for p in (0, 2 * math.pi / 3)
                        ]
                        R = rotation_matrix_from_vectors(np.array([0, 0, 1]), -v)
                        add_H([-R @ d for d in dirs], CH_len)
                    continue

                if len(neighbors) == 2:
                    v1, v2 = neighbors
                    bis = normalize(-(v1 + v2))
                    plane_normal = normalize(np.cross(v1, v2))
                    if np.linalg.norm(plane_normal) < 1e-8:
                        plane_normal = np.array([0.0, 0.0, 1.0])
                    if np.linalg.norm(bis) < 1e-8:
                        bis = normalize(np.cross(v1, plane_normal))
                    if any(o >= 1.5 for o in orders):
                        add_H(
                            [
                                rotate_vector(bis, plane_normal, math.radians(a))
                                for a in (60, -60)
                            ],
                            CH_len,
                        )
                    else:
                        angle = math.radians(54.75)
                        perp = plane_normal
                        add_H(
                            [
                                math.cos(angle) * bis + math.sin(angle) * perp,
                                math.cos(angle) * bis - math.sin(angle) * perp,
                            ],
                            CH_len,
                        )
                    continue

            # CH
            if nH == 1:
                if len(neighbors) == 3 and all(order == 1 for order in orders):
                    # A saturated methine drawn with three planar neighbors
                    # needs its H out of that plane, never toward another C.
                    normal = normalize(np.cross(neighbors[0], neighbors[1]))
                    if np.linalg.norm(normal) < 1e-8:
                        normal = np.array([0.0, 0.0, 1.0])
                    add_H([normal], 1.09)
                    continue
                avg = normalize(-np.sum(neighbors, axis=0))
                if np.linalg.norm(avg) < 1e-8:
                    avg = normalize(np.cross(neighbors[0], [0, 0, 1]))
                add_H([avg], 1.09)

        mol = Atoms(symbols=sym, positions=pos)
        from rdkit import Chem

        for index, spec in enumerate(graph.atoms):
            if spec.isotope:
                mass = Chem.GetPeriodicTable().GetMassForIsotope(
                    spec.number, spec.isotope
                )
                if mass <= 0:
                    raise ValueError(
                        f"Unknown isotope {spec.isotope} for atom {spec.id} ({spec.symbol})."
                    )
                bare_mol[index].mass = mol[index].mass = mass
        source = (
            cdxml_content.encode() if isinstance(cdxml_content, str) else cdxml_content
        )
        mol.info["cdxml_source"] = {
            "schema": 1,
            "sha256": hashlib.sha256(source).hexdigest(),
            "formal_charge": sum(atom.charge for atom in graph.atoms),
            "atoms": {
                atom.id: {
                    "element": atom.symbol,
                    "formal_charge": atom.charge,
                    "isotope": atom.isotope,
                    "radical_electrons": atom.radical_electrons,
                    "declared_hydrogens": atom.hydrogens,
                    "added_hydrogens": impl_H[atom.id],
                }
                for atom in graph.atoms
            },
            "bonds": [
                {"begin": bond.begin, "end": bond.end, "type": bond.type}
                for bond in graph.bonds
            ],
        }
        # Transient conversion metadata, consumed before storing/exporting the
        # finite model. Close geometric contacts never enter this bond list.
        mol.info["_cdxml_bonds"] = chemical_bonds
        mol.info["_cdxml_stereochemistry"] = graph.stereochemistry
        mol.info["_cdxml_cleanup_applied"] = cleanup_applied
        mol.info["_cdxml_atom_sources"] = atom_sources
        import copy

        bare_mol.info["cdxml_source"] = copy.deepcopy(mol.info["cdxml_source"])
        bare_mol.info["_cdxml_bonds"] = [
            (graph.index[bond.begin], graph.index[bond.end], bond.order)
            for bond in graph.bonds
        ]
        bare_mol.info["_cdxml_stereochemistry"] = graph.stereochemistry
        bare_mol.info["_cdxml_cleanup_applied"] = cleanup_applied
        bare_mol.info["_cdxml_atom_sources"] = atom_sources[: len(graph.atoms)]
        coordination = graph.coordination_interpretations()
        if coordination:
            interpretation = ", ".join(
                f"{item['donor']}→{item['metal']}" for item in coordination
            )
            geometry_message += (
                f" Warning: single N/O-metal bonds interpreted as coordination ({interpretation}); "
                "original drawing and declared H retained. Review this chemical interpretation."
            )
            mol.info["cdxml_source"]["coordination_interpretations"] = coordination
            bare_mol.info["cdxml_source"]["coordination_interpretations"] = coordination
        if graph.stereochemistry:
            geometry_message += " Warning: encoded stereochemistry is retained in the source drawing but is not reconstructed in this starting model."
        if graph.annotation_warnings:
            geometry_message += (
                " Warning: " + "; ".join(graph.annotation_warnings) + "."
            )
            mol.info["cdxml_source"]["annotation_warnings"] = list(
                graph.annotation_warnings
            )
            bare_mol.info["cdxml_source"]["annotation_warnings"] = list(
                graph.annotation_warnings
            )
        msg = geometry_message
        return msg, bare_mol, mol

    # ---------------- Geometry utilities ----------------
    @staticmethod
    def transform_points(
        set1: np.ndarray, set2: np.ndarray, points: np.ndarray
    ) -> list[list[float]]:
        """Transform points based on scaling and rotation aligning set1→set2."""
        centroid1, centroid2 = np.mean(set1, axis=0), np.mean(set2, axis=0)
        centered1, centered2 = set1 - centroid1, set2 - centroid2
        if np.linalg.norm(centered1, axis=1).mean() < 1e-12:
            return (np.asarray(points) - centroid1 + centroid2).tolist()
        scale = (
            np.linalg.norm(centered2, axis=1).mean()
            / np.linalg.norm(centered1, axis=1).mean()
        )
        cross_cov = np.dot(centered1.T, centered2)
        u, _, vt = np.linalg.svd(cross_cov)
        rotation_m = np.dot(vt.T, u.T)
        return (scale * np.dot(points - centroid1, rotation_m.T) + centroid2).tolist()

    @staticmethod
    def max_extension_points(points: np.ndarray) -> np.ndarray:
        """Return two points defining the maximum extension of a set of 3D points."""
        x_range, y_range = np.ptp(points[:, 0]), np.ptp(points[:, 1])
        if x_range >= y_range:
            minx, maxx = np.min(points[:, 0]), np.max(points[:, 0])
            return np.array([[minx - 7.5, 0, 0], [maxx + 7.5, 0, 0]])
        miny, maxy = np.min(points[:, 1]), np.max(points[:, 1])
        return np.array([[0, miny - 7.5, 0], [0, maxy + 7.5, 0]])

    # ---------------- CDXML analysis ----------------
    def extract_crossing_and_atom_positions(
        self,
        cdxml_content: str | bytes,
        atom_positions_override: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray, bool]:
        """Extract robust periodic boundaries and atom positions from CDXML.

        ChemDraw may serialize the right bracket before the left bracket. Crossing
        matching must therefore be independent of both XML order and vector sign.
        """
        graph = read_cdxml_graph(cdxml_content)
        positions = np.asarray([atom.position for atom in graph.atoms])
        if atom_positions_override is not None:
            override = np.asarray(atom_positions_override, dtype=float)
            if override.ndim != 2 or len(override) != len(positions):
                raise ValueError(
                    "atom position override does not match the CDXML atom count"
                )
            if override.shape[1] == 2:
                override = np.column_stack([override, np.zeros(len(override))])
            if override.shape != positions.shape or not np.isfinite(override).all():
                raise ValueError(
                    "atom position override must have shape (N, 2) or (N, 3) and finite coordinates"
                )
            positions = override.copy()
        repeat = identify_repeat(cdxml_content, graph)
        if repeat is None:
            return self.max_extension_points(positions), positions, True
        central = set(repeat.atom_ids)
        left = set(repeat.left_outside)
        midpoints = []
        for bond in graph.bonds:
            if (bond.begin in central and bond.end in left) or (
                bond.end in central and bond.begin in left
            ):
                midpoints.append(
                    (
                        positions[graph.index[bond.begin]]
                        + positions[graph.index[bond.end]]
                    )
                    / 2
                )
        origin = np.mean(midpoints, axis=0)
        return (
            np.asarray(
                [origin, origin + repeat.vector_for_positions(graph, positions)]
            ),
            positions,
            False,
        )

    @staticmethod
    def align_and_trim_atoms(
        atoms: Atoms,
        crossing_points: np.ndarray,
        units: str | None = None,
        original_atoms: Atoms | None = None,
    ) -> Atoms:
        """
        Align, trim, and optionally replicate atoms along the periodic direction.
        Restores hydrogens lost during trimming using original geometry.
        """
        assert crossing_points.shape == (2, 3), (
            "crossing_points must be a 2x3 NumPy array."
        )

        vector = crossing_points[1] - crossing_points[0]
        norm_vector = np.linalg.norm(vector)
        angle = np.arctan2(vector[1], vector[0])

        def rotate_atoms(atoms_obj: Atoms) -> Atoms:
            atoms_copy = atoms_obj.copy()
            atoms_copy.rotate(-np.degrees(angle), "z", center=(0, 0, 0))
            return atoms_copy

        atoms_rot = rotate_atoms(atoms)
        orig_rot = rotate_atoms(original_atoms if original_atoms is not None else atoms)

        rotation_matrix = np.array(
            [
                [np.cos(-angle), -np.sin(-angle), 0],
                [np.sin(-angle), np.cos(-angle), 0],
                [0, 0, 1],
            ]
        )
        rotated_cp = np.dot(crossing_points, rotation_matrix.T)
        x_min, x_max = min(rotated_cp[:, 0]), max(rotated_cp[:, 0]) + 0.1

        pos = atoms_rot.get_positions()
        mask_main, mask_tail, mask_head = (
            (pos[:, 0] > x_min) & (pos[:, 0] <= x_max),
            pos[:, 0] <= x_min,
            pos[:, 0] > x_max,
        )
        bounded_atoms, tail_atoms, head_atoms = (
            atoms_rot[mask_main].copy(),
            atoms_rot[mask_tail].copy(),
            atoms_rot[mask_head].copy(),
        )
        kept_indices = np.where(mask_main)[0]

        nl = NeighborList(
            [covalent_radii[num] * 1.2 for num in orig_rot.numbers],
            self_interaction=False,
            bothways=True,
        )
        nl.update(orig_rot)
        symbols = orig_rot.get_chemical_symbols()

        restored_positions, restored_symbols = [], []
        for i in kept_indices:
            if symbols[i] == "H":
                continue
            indices, _ = nl.get_neighbors(i)
            for j in indices:
                if symbols[j] == "H":
                    h_pos = orig_rot.positions[j]
                    if not (x_min < h_pos[0] <= x_max):
                        restored_positions.append(h_pos)
                        restored_symbols.append("H")

        if restored_positions:
            bounded_atoms += ase.Atoms(restored_symbols, positions=restored_positions)

        if units is None:
            n_units = None
        else:
            try:
                n_units = int(units)
            except ValueError:
                n_units = None

        if n_units is None or n_units < 1:
            atoms_final = bounded_atoms
        else:
            replicated_atoms = bounded_atoms.copy()
            for ni in range(1, n_units):
                shifted = bounded_atoms.get_positions() + np.array(
                    [ni * norm_vector, 0, 0]
                )
                replicated_atoms += ase.Atoms(
                    bounded_atoms.get_chemical_symbols(), positions=shifted
                )
            shifted_head = head_atoms.get_positions() + np.array(
                [(n_units - 1) * norm_vector, 0, 0]
            )
            replicated_atoms += ase.Atoms(
                head_atoms.get_chemical_symbols(), positions=shifted_head
            )
            replicated_atoms += tail_atoms
            atoms_final = replicated_atoms

        # Hydrogens restored across a periodic cut can overlap a carbon on the
        # opposite boundary. Such atoms are trimming artefacts, not chemistry.
        symbols = np.asarray(atoms_final.get_chemical_symbols())
        hydrogen_indices = np.where(symbols == "H")[0]
        heavy_indices = np.where(symbols != "H")[0]
        keep = np.ones(len(atoms_final), dtype=bool)
        for hydrogen_index in hydrogen_indices:
            if not len(heavy_indices):
                keep[hydrogen_index] = False
                continue
            nearest_heavy = np.min(
                np.linalg.norm(
                    atoms_final.positions[heavy_indices]
                    - atoms_final.positions[hydrogen_index],
                    axis=1,
                )
            )
            if nearest_heavy < 0.70 or nearest_heavy > 1.35:
                keep[hydrogen_index] = False
        atoms_final = atoms_final[keep]

        if n_units is None or n_units < 1:
            l1, atoms_final.pbc = norm_vector, [True, False, False]
        else:
            l1 = np.ptp(atoms_final.get_positions()[:, 0]) + 15.0
        l2 = 15.0 + np.ptp(atoms_final.get_positions()[:, 1])
        l3 = 15.0
        atoms_final.set_cell([l1, l2, l3])
        atoms_final.center()

        return atoms_final
