"""Chemistry must agree with the drawing, not only its total molecular formula."""

import xml.etree.ElementTree as ET
from io import StringIO

import ase.io
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdMolDescriptors

from aiidalab_widgets_empa import cdxml_rendering
from aiidalab_widgets_empa.cdxml import CdxmlUploadWidget
from aiidalab_widgets_empa.cdxml_graph import read_cdxml_graph


def drawing(nodes, edges=()):
    root = ET.Element("CDXML")
    fragment = ET.SubElement(ET.SubElement(root, "page"), "fragment")
    for index, (number, xy, attributes) in enumerate(nodes):
        spec = {"id": str(index), "p": " ".join(map(str, xy))}
        if number is not None:
            spec["Element"] = str(number)
        spec.update({key: str(value) for key, value in attributes.items()})
        ET.SubElement(fragment, "n", spec)
    for index, (begin, end, order) in enumerate(edges):
        ET.SubElement(
            fragment, "b", id=f"b{index}", B=str(begin), E=str(end), Order=str(order)
        )
    return ET.tostring(root)


def convert(content):
    return CdxmlUploadWidget.cdxml_to_ase_from_string(content)[2]


@pytest.mark.parametrize(
    "element,formula", [(14, "C2H8Si"), (32, "C2H8Ge"), (50, "C2H6Sn")]
)
def test_element_numbers_never_fall_back_to_carbon(element, formula):
    content = drawing(
        [(6, (-30, 0), {}), (element, (0, 0), {}), (6, (15, 26), {})],
        [(0, 1, 1), (1, 2, 1)],
    )
    atoms = convert(content)
    assert atoms[1].number == element
    assert atoms.get_chemical_formula() == formula


@pytest.mark.parametrize("element", [12, 20, 26, 27, 28, 29, 30, 78])
def test_metals_do_not_acquire_implicit_hydrides(element):
    atoms = convert(drawing([(element, (0, 0), {})]))
    assert len(atoms) == 1
    assert atoms[0].number == element


def test_dative_direction_and_hydrogen_ownership_are_preserved():
    content = drawing(
        [
            (6, (-30, 0), {}),
            (7, (0, 0), {"NumHydrogens": 2}),
            (26, (30, 0), {"Charge": 2, "NumHydrogens": 0}),
        ],
        [(0, 1, 1), (1, 2, "dative")],
    )
    graph = read_cdxml_graph(content)
    assert graph.bonds[1].begin == "1" and graph.bonds[1].end == "2"
    assert graph.bonds[1].type == "dative"
    atoms = convert(content)
    assert atoms.get_chemical_formula() == "CH5FeN"
    assert [
        i
        for parent, i, _ in atoms.info["_cdxml_bonds"]
        if parent == 1 and atoms[i].symbol == "H"
    ]
    assert (
        sum(
            parent == 1 and atoms[i].symbol == "H"
            for parent, i, _ in atoms.info["_cdxml_bonds"]
        )
        == 2
    )
    assert not any(
        parent == 2 and atoms[i].symbol == "H"
        for parent, i, _ in atoms.info["_cdxml_bonds"]
    )
    assert atoms.info["cdxml_source"]["formal_charge"] == 2


@pytest.mark.parametrize("symbol", ["Fe", "Co", "Zn"])
def test_synthetic_metal_porphine(symbol):
    smiles = (
        f"C1=Cc2cc3ccc4cc5[n]6->[{symbol}+2]7(<-[n]2c1cc1ccc(cc6C=C5)[n-]->71)<-[n-]34"
    )
    molecule = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(molecule)
    Chem.Kekulize(molecule, clearAromaticFlags=True)
    nodes = []
    for atom in molecule.GetAtoms():
        point = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        attributes = {"Charge": atom.GetFormalCharge()}
        if atom.GetAtomicNum() != 6:
            attributes["NumHydrogens"] = atom.GetTotalNumHs()
        nodes.append((atom.GetAtomicNum(), (point.x * 30, point.y * 30), attributes))
    edges = [
        (
            b.GetBeginAtomIdx(),
            b.GetEndAtomIdx(),
            "dative"
            if b.GetBondType() == Chem.BondType.DATIVE
            else str(int(b.GetBondTypeAsDouble())),
        )
        for b in molecule.GetBonds()
    ]
    atoms = convert(drawing(nodes, edges))
    assert atoms.get_chemical_formula() == rdMolDescriptors.CalcMolFormula(molecule)
    assert atoms.info["cdxml_source"]["formal_charge"] == 0
    assert (
        sum(bond["type"] == "dative" for bond in atoms.info["cdxml_source"]["bonds"])
        == 4
    )
    assert sum(atom.symbol == symbol for atom in atoms) == 1


@pytest.mark.parametrize("order", ["1.5", "A"])
def test_aromatic_benzene_matches_kekule_formula(order):
    nodes = [
        (6, (30 * np.cos(phi), 30 * np.sin(phi)), {})
        for phi in np.arange(6) * np.pi / 3
    ]
    atoms = convert(drawing(nodes, [(i, (i + 1) % 6, order) for i in range(6)]))
    assert atoms.get_chemical_formula() == "C6H6"


@pytest.mark.parametrize(
    "radical,formula",
    [
        ("None", "C2H6"),
        ("0", "C2H6"),
        ("Doublet", "C2H5"),
        ("2", "C2H5"),
        ("Singlet", "C2H4"),
        ("1", "C2H4"),
        ("Triplet", "C2H4"),
        ("3", "C2H4"),
    ],
)
def test_radical_electron_counts(radical, formula):
    atoms = convert(
        drawing([(6, (0, 0), {"Radical": radical}), (6, (30, 0), {})], [(0, 1, 1)])
    )
    assert atoms.get_chemical_formula() == formula


def test_graphical_radical_dot_changes_CH2_to_CH():
    root = ET.fromstring(drawing([(6, (0, 0), {}), (6, (30, 0), {})], [(0, 1, 2)]))
    graphic = ET.SubElement(root, "graphic", SymbolType="Electron")
    ET.SubElement(graphic, "represent", object="0")
    atoms = convert(ET.tostring(root))
    assert atoms.get_chemical_formula() == "C2H3"
    assert atoms.info["cdxml_source"]["atoms"]["0"]["added_hydrogens"] == 1


@pytest.mark.parametrize(
    "number,attributes,formula",
    [(6, {}, "CH4"), (7, {}, "H3N"), (7, {"Charge": 1, "NumHydrogens": 4}, "H4N")],
)
def test_isolated_atom_hydrogen_counts_and_nonzero_bond_lengths(
    number, attributes, formula
):
    atoms = convert(drawing([(number, (0, 0), attributes)]))
    assert atoms.get_chemical_formula() == formula
    assert all(atoms.get_distance(0, index) > 0.9 for index in range(1, len(atoms)))


def test_charge_and_explicit_zero_hydrogens_prevent_spurious_OH():
    atoms = convert(
        drawing(
            [(6, (0, 0), {}), (8, (30, 0), {"Charge": -1, "NumHydrogens": 0})],
            [(0, 1, 1)],
        )
    )
    assert atoms.get_chemical_formula() == "CH3O"
    assert atoms.info["cdxml_source"]["formal_charge"] == -1
    assert atoms.info["cdxml_source"]["atoms"]["1"]["added_hydrogens"] == 0


def test_explicit_H_nodes_and_label_H_are_not_double_counted():
    atoms = convert(
        drawing([(7, (0, 0), {"NumHydrogens": 2}), (1, (30, 0), {})], [(0, 1, 1)])
    )
    assert atoms.get_chemical_formula() == "H3N"
    assert len(atoms) == 4


def test_scale_without_carbon_carbon_is_independent_of_drawing_units():
    def molecule(size):
        return convert(drawing([(6, (0, 0), {}), (8, (size, 0), {})], [(0, 1, 1)]))

    assert np.allclose(molecule(3).positions, molecule(30).positions)
    assert 1.2 < molecule(30).get_distance(0, 1) < 1.6


def test_collinear_CH2_does_not_generate_H_on_the_carbon():
    atoms = convert(
        drawing(
            [(6, (-30, 0), {}), (6, (0, 0), {}), (6, (30, 0), {})],
            [(0, 1, 1), (1, 2, 1)],
        )
    )
    assert atoms.get_chemical_formula() == "C3H8"
    for first, second, _ in atoms.info["_cdxml_bonds"]:
        assert atoms.get_distance(first, second) > 1.0


def test_isotope_mass_and_source_survive_extended_xyz():
    atoms = convert(
        drawing([(6, (0, 0), {"Isotope": 13}), (6, (30, 0), {})], [(0, 1, 1)])
    )
    assert atoms[0].mass == pytest.approx(13.00335484)
    atoms.info.pop("_cdxml_bonds")
    atoms.info.pop("_cdxml_stereochemistry")
    output = StringIO()
    ase.io.write(output, atoms, format="extxyz")
    output.seek(0)
    loaded = ase.io.read(output, format="extxyz")
    assert loaded[0].mass == pytest.approx(atoms[0].mass)
    assert loaded.info["cdxml_source"] == atoms.info["cdxml_source"]


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda root: root.findall(".//n")[1].set("id", "0"), "duplicate atom"),
        (lambda root: root.findall(".//n")[1].set("p", "0 0"), "zero drawing length"),
        (lambda root: root.findall(".//n")[1].set("p", "nan 0"), "nonfinite"),
        (lambda root: root.findall(".//n")[1].set("Element", "120"), "atomic number"),
        (
            lambda root: root.findall(".//n")[1].set("NodeType", "Nickname"),
            "Expand abbreviations",
        ),
        (lambda root: root.find(".//b").set("B", "absent"), "unavailable atom"),
        (lambda root: root.find(".//b").set("B", "1"), "itself"),
        (lambda root: root.find(".//b").set("Order", "nonsense"), "unsupported order"),
    ],
)
def test_bad_chemistry_is_reported_instead_of_silently_substituted(mutate, message):
    root = ET.fromstring(drawing([(6, (0, 0), {}), (6, (30, 0), {})], [(0, 1, 1)]))
    mutate(root)
    with pytest.raises(ValueError, match=message):
        convert(ET.tostring(root))


def test_overvalence_and_duplicate_connections_are_rejected():
    content = drawing(
        [(6, (0, 0), {}), (6, (30, 0), {}), (6, (0, 30), {})], [(0, 1, 3), (0, 2, 3)]
    )
    with pytest.raises(ValueError, match="cannot be validated"):
        convert(content)
    content = drawing([(6, (0, 0), {}), (6, (30, 0), {})], [(0, 1, 1), (0, 1, 1)])
    with pytest.raises(ValueError, match="duplicates the connection"):
        convert(content)


def test_preview_draws_dative_arrow_in_source_direction(monkeypatch):
    arrows = []
    original = cdxml_rendering.FancyArrowPatch

    def record(first, second, **kwargs):
        arrows.append((first, second))
        return original(first, second, **kwargs)

    monkeypatch.setattr(cdxml_rendering, "FancyArrowPatch", record)
    content = drawing(
        [(7, (0, 0), {}), (30, (30, 0), {"Charge": 2})], [(0, 1, "dative")]
    )
    png = cdxml_rendering.render_cdxml_png(content)
    assert png.startswith(b"\x89PNG")
    assert len(arrows) == 1
    assert np.array_equal(arrows[0][0], [0, 0])
    assert np.array_equal(arrows[0][1], [30, 0])


@pytest.mark.parametrize(
    "radical,count", [("None", 0), ("2", 1), ("Doublet", 1), ("1", 2), ("Triplet", 2)]
)
def test_preview_radical_symbols_match_the_chemical_electron_count(
    monkeypatch, radical, count
):
    dots = []
    original = cdxml_rendering.Circle

    def record(*args, **kwargs):
        dots.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(cdxml_rendering, "Circle", record)
    cdxml_rendering.render_cdxml_png(drawing([(6, (0, 0), {"Radical": radical})]))
    assert len(dots) == count
