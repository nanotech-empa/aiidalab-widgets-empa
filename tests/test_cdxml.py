from pathlib import Path
import xml.etree.ElementTree as ET

import ase
import numpy as np
import pytest

import aiidalab_widgets_empa as awe
from aiidalab_widgets_empa.cdxml import (
    _bounded_faces,
    _has_bond_crossings,
    _signed_area,
)


# @pytest.mark.usefixtures("aiida_profile_clean")
def test_structure_upload_widget():
    """Test the `StructureUploadWidget`."""
    widget = awe.CdxmlUploadWidget()
    assert widget.structure is None

    filename = Path(__file__).parent / "7AGNR.cdxml"

    with open(filename, "rb") as f:
        content = f.read()

    # Simulate the structure upload.
    widget._on_file_upload(
        change={
            "new": {
                "7AGNR.cdxml": {
                    "content": content,
                }
            }
        }
    )
    widget.create_button.click()
    assert isinstance(widget.structure, ase.Atoms)
    assert widget.structure.get_chemical_formula() == "C14H4"
    # Simulate the structure upload.
    widget._on_file_upload(
        change={
            "new": {
                "7AGNR.cdxml": {
                    "content": content,
                }
            }
        }
    )
    # sets 2 units, finite size
    widget.nunits.value = "2"
    widget.create_button.click()
    assert isinstance(widget.structure, ase.Atoms)
    assert widget.structure.get_chemical_formula() == "C56H22"

    # case of benzene molecule
    filename = Path(__file__).parent / "benzene.cdxml"

    with open(filename, "rb") as f:
        content = f.read()
    # Simulate the structure upload.
    widget._on_file_upload(
        change={
            "new": {
                "benzene.cdxml": {
                    "content": content,
                }
            }
        }
    )
    widget.create_button.click()
    assert isinstance(widget.structure, ase.Atoms)
    assert widget.structure.get_chemical_formula() == "C6H6"


def _explicit_graph(content):
    root = ET.fromstring(content)
    atom_ids = [node.get("id") for node in root.iter("n") if "p" in node.attrib]
    atom_index = {atom_id: index for index, atom_id in enumerate(atom_ids)}
    edges = [
        (atom_index[bond.get("B")], atom_index[bond.get("E")])
        for bond in root.iter("b")
        if bond.get("B") in atom_index and bond.get("E") in atom_index
    ]
    return root, edges


def _sp2_angle_error(positions, edges):
    adjacency = [[] for _ in positions]
    for first, second in edges:
        adjacency[first].append(second)
        adjacency[second].append(first)
    errors = []
    for center, neighbors in enumerate(adjacency):
        if len(neighbors) not in (2, 3):
            continue
        for first_index in range(len(neighbors)):
            for second_index in range(first_index + 1, len(neighbors)):
                first = positions[neighbors[first_index]] - positions[center]
                second = positions[neighbors[second_index]] - positions[center]
                cosine = np.dot(first, second) / (
                    np.linalg.norm(first) * np.linalg.norm(second)
                )
                errors.append((cosine + 0.5) ** 2)
    return float(np.sqrt(np.mean(errors)))


def test_planarity_check_rejects_crossing_and_overlapping_bonds():
    crossing_positions = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    overlapping_positions = np.array([[0.0, 0.0], [2.0, 0.0], [0.5, 0.0], [1.5, 0.0]])

    assert _has_bond_crossings(crossing_positions, [(0, 1), (2, 3)])
    assert _has_bond_crossings(overlapping_positions, [(0, 1), (2, 3)])


def test_periodic_crossings_are_independent_of_bracket_order():
    content = (Path(__file__).parent / "mixed_rings_periodic.cdxml").read_bytes()
    widget = awe.CdxmlUploadWidget()

    crossing_points, _, is_not_periodic = widget.extract_crossing_and_atom_positions(
        content
    )

    assert not is_not_periodic
    assert crossing_points.shape == (2, 3)
    assert np.linalg.norm(crossing_points[1] - crossing_points[0]) == pytest.approx(
        42.75, abs=0.05
    )


def test_cdxml_scaling_uses_explicit_carbon_bonds():
    content = (Path(__file__).parent / "mixed_rings_periodic.cdxml").read_bytes()
    root, edges = _explicit_graph(content)
    scaled_root = ET.fromstring(content)
    for node in scaled_root.iter("n"):
        if "p" not in node.attrib:
            continue
        coordinates = [5.0 * float(value) for value in node.attrib["p"].split()]
        node.attrib["p"] = " ".join(str(value) for value in coordinates)

    _, original, _ = awe.CdxmlUploadWidget.cdxml_to_ase_from_string(content)
    _, rescaled, _ = awe.CdxmlUploadWidget.cdxml_to_ase_from_string(
        ET.tostring(scaled_root)
    )
    original_lengths = sorted(
        np.linalg.norm(original.positions[second] - original.positions[first])
        for first, second in edges
    )
    rescaled_lengths = sorted(
        np.linalg.norm(rescaled.positions[second] - rescaled.positions[first])
        for first, second in edges
    )

    assert np.allclose(original_lengths, rescaled_lengths)


def test_geometry_cleanup_preserves_planar_fused_ring_graph():
    content = (Path(__file__).parent / "mixed_rings_periodic.cdxml").read_bytes()
    _, edges = _explicit_graph(content)

    _, original, _ = awe.CdxmlUploadWidget.cdxml_to_ase_from_string(content)
    message, cleaned, _ = awe.CdxmlUploadWidget.cdxml_to_ase_from_string(
        content, symmetrize=True
    )

    original_positions = original.positions[:, :2]
    cleaned_positions = cleaned.positions[:, :2]
    lengths = np.array(
        [
            np.linalg.norm(cleaned_positions[second] - cleaned_positions[first])
            for first, second in edges
        ]
    )

    assert "cleanup skipped" not in message
    assert lengths.min() >= 1.35
    assert lengths.max() <= 1.60
    assert not _has_bond_crossings(cleaned_positions, edges)
    assert _sp2_angle_error(cleaned_positions, edges) < _sp2_angle_error(
        original_positions, edges
    )

    faces = _bounded_faces(original_positions, edges)
    for face in faces:
        original_area = _signed_area(original_positions[face])
        cleaned_area = _signed_area(cleaned_positions[face])
        assert original_area * cleaned_area > 0

    pentagon_angle_spreads = []
    for face in faces:
        if len(face) != 5:
            continue
        points = cleaned_positions[face]
        angles = []
        for index in range(5):
            first = points[index - 1] - points[index]
            second = points[(index + 1) % 5] - points[index]
            cosine = np.dot(first, second) / (
                np.linalg.norm(first) * np.linalg.norm(second)
            )
            angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        pentagon_angle_spreads.append(np.std(angles))

    # Fused pentagons must be allowed to remain flattened/non-regular.
    assert pentagon_angle_spreads
    assert max(pentagon_angle_spreads) > 5.0


def test_mixed_ring_widget_flow_with_cleanup():
    content = (Path(__file__).parent / "mixed_rings_periodic.cdxml").read_bytes()
    widget = awe.CdxmlUploadWidget()
    widget.symmetrize_geometry.value = True

    widget._on_file_upload(
        change={
            "new": {
                "mixed_rings_periodic.cdxml": {
                    "content": content,
                }
            }
        }
    )
    assert not widget.nunits.disabled
    assert widget.crossing_points is not None
    assert "cleanup skipped" not in widget.output_message.value

    widget.create_button.click()

    assert isinstance(widget.structure, ase.Atoms)
    assert widget.structure.pbc.tolist() == [True, False, False]
    assert "Carbon geometry cleaned globally" in widget.output_message.value
