"""Periodic membership and H ownership must survive geometry and UI changes."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from ase.build import graphene_nanoribbon

from aiidalab_widgets_empa.cdxml import CdxmlUploadWidget as Widget
from aiidalab_widgets_empa.cdxml_graph import read_cdxml_graph
from aiidalab_widgets_empa.cdxml_periodic import (
    PeriodicAmbiguity,
    build_repeat_model,
    identify_repeat,
)

DATA = Path(__file__).parent


def upload(widget, content):
    widget._on_file_upload({"new": ({"name": "repeat.cdxml", "content": content},)})


def prepare(content, cleanup=False):
    graph = read_cdxml_graph(content)
    repeat = identify_repeat(content, graph)
    _, bare, whole = Widget.cdxml_to_ase_from_string(
        content, symmetrize=cleanup, periodic_repeat=repeat
    )
    return graph, repeat, bare, whole


def test_armchair_formula_matches_independent_lattice_reference():
    content = (DATA / "7AGNR.cdxml").read_bytes()
    graph, repeat, _, whole = prepare(content)
    actual = build_repeat_model(whole, graph, repeat)
    expected = graphene_nanoribbon(3.5, 1, type="armchair", saturated=True)
    assert actual.get_chemical_formula() == expected.get_chemical_formula() == "C14H4"
    assert actual.pbc.tolist() == [True, False, False]


def test_missing_double_bond_is_not_hidden_by_deleting_required_H():
    root = ET.fromstring((DATA / "7AGNR.cdxml").read_bytes())
    root.find('.//b[@id="101"]').attrib.pop("Order")
    graph, repeat, _, whole = prepare(ET.tostring(root))
    actual = build_repeat_model(whole, graph, repeat)
    assert actual.get_chemical_formula() == "C14H6"
    # These are saturated methines in the graph, even though the 2D drawing
    # places the carbon skeleton in a plane. Their H must not point at another C.
    for identifier in ("69", "97"):
        parent = graph.index[identifier]
        child = next(
            j
            for i, j, _ in whole.info["_cdxml_bonds"]
            if i == parent and whole[j].symbol == "H"
        )
        assert abs(whole.positions[child, 2] - whole.positions[parent, 2]) > 1.0


@pytest.mark.parametrize("cleanup", [False, True])
def test_bracket_locations_do_not_define_membership_or_cell(cleanup):
    source = (DATA / "mixed_rings_periodic.cdxml").read_bytes()
    root = ET.fromstring(source)
    for index, graphic in enumerate(root.iter("graphic")):
        if graphic.get("BracketType") == "Square":
            graphic.set(
                "BoundingBox", f"{2000 + index * 100} -700 {2100 + index * 100} 900"
            )
    altered = ET.tostring(root)
    graph, repeat, bare, whole = prepare(source, cleanup)
    second_graph, second_repeat, second_bare, second_whole = prepare(altered, cleanup)
    assert repeat.atom_ids == second_repeat.atom_ids
    assert repeat.images == second_repeat.images
    assert np.allclose(repeat.vector, second_repeat.vector)
    assert np.allclose(bare.positions, second_bare.positions)
    first = build_repeat_model(whole, graph, repeat)
    second = build_repeat_model(second_whole, second_graph, second_repeat)
    assert first.get_chemical_formula() == second.get_chemical_formula()
    assert np.allclose(first.cell, second.cell)


@pytest.mark.parametrize("scale,angle", [(0.1, 0), (5, 0), (2, 67), (1, 133)])
def test_drawing_scale_and_rotation_preserve_periodic_model(scale, angle):
    source = (DATA / "7AGNR.cdxml").read_bytes()
    graph, repeat, _, whole = prepare(source)
    reference = build_repeat_model(whole, graph, repeat)
    root = ET.fromstring(source)
    theta = np.radians(angle)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    for node in root.iter("n"):
        xy = np.asarray(list(map(float, node.get("p").split())))
        point = scale * (rotation @ xy) + [400, -700]
        node.set("p", " ".join(map(str, point)))
    graph, repeat, _, whole = prepare(ET.tostring(root))
    actual = build_repeat_model(whole, graph, repeat)
    assert actual.get_chemical_formula() == reference.get_chemical_formula()
    assert np.allclose(actual.cell, reference.cell, atol=1e-8)
    assert np.allclose(
        actual.get_all_distances(mic=True),
        reference.get_all_distances(mic=True),
        atol=1e-8,
    )


def test_cleanup_enforces_the_same_periodic_equivalences_exactly():
    graph, repeat, bare, whole = prepare(
        (DATA / "mixed_rings_periodic.cdxml").read_bytes(), True
    )
    vector = repeat.vector_for_positions(graph, bare.positions)
    for image, central, shift in repeat.images:
        assert np.allclose(
            bare.positions[graph.index[image]],
            bare.positions[graph.index[central]] + shift * vector,
            atol=1e-8,
        )
    lengths = [
        bare.get_distance(graph.index[b.begin], graph.index[b.end]) for b in graph.bonds
    ]
    assert min(lengths) >= 1.35 - 1e-6
    assert max(lengths) <= 1.60 + 1e-6


def test_finite_repetition_retains_every_source_end_H():
    content = (DATA / "7AGNR.cdxml").read_bytes()
    graph, repeat, _, whole = prepare(content)
    single = build_repeat_model(whole, graph, repeat, "1")
    double = build_repeat_model(whole, graph, repeat, "2")
    periodic = build_repeat_model(whole, graph, repeat)
    assert single.get_chemical_formula() == whole.get_chemical_formula()
    assert len(double) == len(single) + len(periodic)
    assert double.get_chemical_formula() == "C56H24"
    assert not double.pbc.any()


@pytest.mark.parametrize("units", ["abc", "0", "-1", "1.5", "2.0", ""])
def test_bad_unit_input_is_not_silently_treated_as_infinite(units):
    graph, repeat, _, whole = prepare((DATA / "7AGNR.cdxml").read_bytes())
    with pytest.raises(ValueError, match="positive integer"):
        build_repeat_model(whole, graph, repeat, units)


def test_manual_atom_pair_survives_cleanup_and_click_selection():
    widget = Widget()
    upload(widget, (DATA / "7AGNR.cdxml").read_bytes())
    image, central, _ = widget._periodic_repeat.images[0]
    indices = [widget._source_graph.index[central], widget._source_graph.index[image]]
    widget.show_atom_indices.value = True
    assert len(widget.preview_canvas.children) == len(widget._source_graph.atoms) + 1
    widget.preview_canvas.children[indices[0] + 1].click()
    widget.preview_canvas.children[indices[1] + 1].click()
    assert [field.value for field in widget.equivalent_atoms] == indices
    assert widget._pair_dirty and widget.create_button.disabled
    widget.apply_atom_pair.click()
    assert not widget.create_button.disabled
    pair = widget._equivalent_atom_ids
    widget.symmetrize_geometry.value = True
    assert widget._equivalent_atom_ids == pair
    widget.create_button.click()
    assert widget.structure.get_chemical_formula() == "C14H4"
    old_css = widget.picker_css.value
    widget.preview_zoom.click()
    assert widget.picker_css.value != old_css
    widget.reset_atom_pair.click()
    assert widget._equivalent_atom_ids is None


def test_graphical_brackets_require_an_explicit_atom_pair_without_metadata():
    root = ET.fromstring((DATA / "7AGNR.cdxml").read_bytes())
    graph, repeat, _, _ = prepare(ET.tostring(root))
    for page in root.iter("page"):
        for group in list(page.findall("bracketedgroup")):
            page.remove(group)
    content = ET.tostring(root)
    with pytest.raises(PeriodicAmbiguity, match="Select two equivalent"):
        identify_repeat(content, graph)
    image, central, _ = repeat.images[0]
    manual = identify_repeat(content, graph, (central, image))
    assert len(manual.atom_ids) == 14


def test_finite_component_brackets_are_not_polymer_periodicity():
    root = ET.fromstring((DATA / "7AGNR.cdxml").read_bytes())
    for element in root.iter():
        if element.tag == "bracketedgroup":
            element.set("BracketUsage", "Component")
    graph = read_cdxml_graph(ET.tostring(root))
    assert identify_repeat(ET.tostring(root), graph) is None


def test_source_based_copolymer_is_not_silently_treated_as_a_single_repeat():
    root = ET.fromstring((DATA / "7AGNR.cdxml").read_bytes())
    next(root.iter("bracketedgroup")).set("BracketUsage", "CopolymerRandom")
    graph = read_cdxml_graph(ET.tostring(root))
    with pytest.raises(PeriodicAmbiguity, match="single structural repeat"):
        identify_repeat(ET.tostring(root), graph)


def test_model_is_invalidated_when_unit_or_atom_selection_changes():
    widget = Widget()
    upload(widget, (DATA / "7AGNR.cdxml").read_bytes())
    widget.create_button.click()
    assert widget.structure is not None
    widget.nunits.value = "2"
    assert widget.structure is None
    widget.create_button.click()
    widget.equivalent_atoms[0].value = 3
    assert widget.structure is None
    widget._on_button_click()
    assert widget.structure is None
    assert "Apply" in widget.output_message.value
