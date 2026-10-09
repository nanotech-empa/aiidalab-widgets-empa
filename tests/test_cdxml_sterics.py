"""Chemical H ownership and rigid-block torsions in crowded finite drawings."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from aiidalab_widgets_empa import CdxmlUploadWidget
from aiidalab_widgets_empa.cdxml_sterics import relieve_steric_contacts


def crowded_biaryl(scale=1.0):
    """Two ortho methyl substituents touch in a planar biphenyl drawing."""
    angles = np.arange(6) * np.pi / 3
    hexagon = 1.43 * np.column_stack((np.cos(angles), np.sin(angles)))
    positions = np.concatenate((hexagon, hexagon + [4.29, 0]))
    # Outward bisectors at the two mutually facing ortho ring carbons.
    positions = np.vstack(
        (positions, positions[1] + 1.43 * np.array([0.5, np.sqrt(3) / 2]))
    )
    positions = np.vstack(
        (positions, positions[8] + 1.43 * np.array([-0.5, np.sqrt(3) / 2]))
    )
    root = ET.Element("CDXML")
    fragment = ET.SubElement(ET.SubElement(root, "page"), "fragment", id="molecule")
    for index, (x, y) in enumerate(positions * scale):
        ET.SubElement(fragment, "n", id=str(index), p=f"{x:.12g} {y:.12g}")
    edges = [
        (offset + index, offset + (index + 1) % 6, 2 if index % 2 == 0 else 1)
        for offset in (0, 6)
        for index in range(6)
    ] + [(0, 9, 1), (1, 12, 1), (8, 13, 1)]
    for index, (first, second, order) in enumerate(edges):
        ET.SubElement(
            fragment,
            "b",
            id=f"b{index}",
            B=str(first),
            E=str(second),
            Order=str(order),
            BS="N",
        )
    return ET.tostring(root)


def upload(widget, content):
    widget._on_file_upload({"new": ({"name": "crowded.cdxml", "content": content},)})


def bond_lengths(atoms, bonds):
    return np.array(
        [np.linalg.norm(atoms.positions[i] - atoms.positions[j]) for i, j, _ in bonds]
    )


def local_cosines(atoms, bonds):
    adjacency = [set() for _ in atoms]
    for first, second, _ in bonds:
        adjacency[first].add(second)
        adjacency[second].add(first)
    cosines = []
    for center, neighbors in enumerate(adjacency):
        neighbors = sorted(neighbors)
        for index, first in enumerate(neighbors):
            for second in neighbors[index + 1 :]:
                a, b = atoms.positions[[first, second]] - atoms.positions[center]
                cosines.append(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
    return np.asarray(cosines)


def test_finite_planar_import_keeps_graph_hydrogens_and_both_methyls():
    widget = CdxmlUploadWidget()
    upload(widget, crowded_biaryl())
    prepared = widget.whole_atoms.copy()
    bonds = prepared.info["_cdxml_bonds"]
    widget.create_button.click()
    assert widget.structure.get_chemical_formula() == "C14H14"
    assert widget.structure.pbc.tolist() == [False, False, False]
    assert "All hydrogens retained" in widget.output_message.value
    assert not widget.structure.info  # no stale index-based graph after editing
    assert np.allclose(
        bond_lengths(widget.structure, bonds), bond_lengths(prepared, bonds)
    )
    for parent in (12, 13):
        children = [j for i, j, _ in bonds if i == parent and prepared[j].symbol == "H"]
        assert len(children) == 3
        assert all(widget.structure[j].symbol == "H" for j in children)
        assert np.allclose(
            [widget.structure.get_distance(parent, j) for j in children], 1.10
        )


def test_torsions_clear_contacts_preserve_bonds_angles_and_planar_rings():
    _, _, original = CdxmlUploadWidget.cdxml_to_ase_from_string(crowded_biaryl())
    bonds = original.info["_cdxml_bonds"]
    before = original.positions.copy()
    result = relieve_steric_contacts(original, bonds)
    assert result.initial_contacts > 0
    assert result.remaining_contacts == 0
    assert result.atoms.get_chemical_formula() == "C14H14"
    assert np.array_equal(original.positions, before)
    assert np.allclose(
        bond_lengths(original, bonds), bond_lengths(result.atoms, bonds), atol=1e-9
    )
    assert np.allclose(
        local_cosines(original, bonds), local_cosines(result.atoms, bonds), atol=1e-9
    )
    for block in (range(6), range(6, 12)):
        ring = result.atoms.positions[list(block)]
        assert np.linalg.svd(ring - ring.mean(axis=0), compute_uv=False)[-1] < 1e-9
    carbon_positions = result.atoms.positions[:14]
    bonded = {frozenset((i, j)) for i, j, _ in bonds}
    assert (
        min(
            np.linalg.norm(carbon_positions[i] - carbon_positions[j])
            for i in range(14)
            for j in range(i + 1, 14)
            if frozenset((i, j)) not in bonded
        )
        > 1.6568
    )


def test_3d_toggle_keeps_the_2d_preview_and_restores_planar_model():
    widget = CdxmlUploadWidget()
    upload(widget, crowded_biaryl())
    assert not widget.resolve_steric_collisions.disabled
    original_png = bytes(widget.png_preview.value)
    prepared = widget.whole_atoms.positions.copy()
    widget.resolve_steric_collisions.value = True
    widget.create_button.click()
    assert np.ptp(widget.structure.positions[:14, 2]) > 0.1
    assert "not energy optimized" in widget.output_message.value
    assert bytes(widget.png_preview.value) == original_png
    assert np.array_equal(widget.whole_atoms.positions, prepared)
    widget.resolve_steric_collisions.value = False
    assert widget.structure is None
    widget.create_button.click()
    assert np.ptp(widget.structure.positions[:14, 2]) < 1e-9
    assert widget.structure.get_chemical_formula() == "C14H14"


def test_drawing_units_do_not_change_the_steric_result():
    _, _, first = CdxmlUploadWidget.cdxml_to_ase_from_string(crowded_biaryl())
    _, _, scaled = CdxmlUploadWidget.cdxml_to_ase_from_string(crowded_biaryl(scale=10))
    result = relieve_steric_contacts(first, first.info["_cdxml_bonds"])
    rescaled = relieve_steric_contacts(scaled, scaled.info["_cdxml_bonds"])
    assert result.remaining_contacts == rescaled.remaining_contacts == 0
    a, b = result.atoms.positions, rescaled.atoms.positions
    assert np.allclose(
        np.linalg.norm(a[:, None] - a[None, :], axis=2),
        np.linalg.norm(b[:, None] - b[None, :], axis=2),
        atol=1e-4,
    )


def test_periodic_import_disables_torsions_and_keeps_existing_formula():
    widget = CdxmlUploadWidget()
    upload(widget, crowded_biaryl())
    widget.resolve_steric_collisions.value = True
    upload(widget, (Path(__file__).parent / "7AGNR.cdxml").read_bytes())
    assert widget.resolve_steric_collisions.disabled
    widget.create_button.click()
    assert widget.structure.get_chemical_formula() == "C14H4"
    assert np.ptp(widget.structure.positions[:, 2]) < 1e-9
    assert not widget.structure.info


def test_rigid_or_unsupported_inputs_report_contacts_without_deleting_atoms():
    _, _, atoms = CdxmlUploadWidget.cdxml_to_ase_from_string(
        (Path(__file__).parent / "benzene.cdxml").read_bytes()
    )
    # Clash between two graph-distinct H inside a rigid ring: torsions cannot fix it.
    atoms.positions[7] = atoms.positions[9]
    result = relieve_steric_contacts(atoms, atoms.info["_cdxml_bonds"])
    assert result.remaining_contacts > 0
    assert "All atoms retained" in result.message
    assert np.array_equal(result.atoms.positions, atoms.positions)
    unsupported = Atoms("FeC", positions=[[0, 0, 0], [1.4, 0, 0]])
    assert (
        "supports C/H/halogen"
        in relieve_steric_contacts(unsupported, [(0, 1, 1)]).message
    )
    unsupported.pbc = True
    assert (
        "finite molecules only"
        in relieve_steric_contacts(unsupported, [(0, 1, 1)]).message
    )


@pytest.mark.parametrize("attribute,value", [("Display", "WedgeBegin"), ("BS", "E")])
def test_encoded_stereochemistry_skips_generic_torsion_search(attribute, value):
    root = ET.fromstring(crowded_biaryl())
    next(root.iter("b")).set(attribute, value)
    widget = CdxmlUploadWidget()
    upload(widget, ET.tostring(root))
    widget.resolve_steric_collisions.value = True
    widget.create_button.click()
    assert "encodes stereochemistry" in widget.output_message.value
    assert widget.structure.get_chemical_formula() == "C14H14"
    assert np.ptp(widget.structure.positions[:14, 2]) < 1e-9
