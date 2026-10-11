"""Read CDXML chemical graphs without inferring bonds from drawing distances."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from ase.data import chemical_symbols
from rdkit import Chem

_RADICAL_ELECTRONS = {
    "": 0,
    "none": 0,
    "0": 0,
    "singlet": 2,
    "1": 2,
    "doublet": 1,
    "2": 1,
    "triplet": 2,
    "3": 2,
}
# These elements may have implicit hydrogens. Metals never acquire implicit
# hydrides simply because a generic valence table would leave unused valence.
_NONMETALS = {
    1,
    2,
    5,
    6,
    7,
    8,
    9,
    10,
    14,
    15,
    16,
    17,
    18,
    32,
    33,
    34,
    35,
    36,
    51,
    52,
    53,
    54,
    85,
    86,
    117,
    118,
}
_BOND_TYPES = {
    "1": Chem.BondType.SINGLE,
    "2": Chem.BondType.DOUBLE,
    "3": Chem.BondType.TRIPLE,
    "1.5": Chem.BondType.AROMATIC,
    "a": Chem.BondType.AROMATIC,
    "aromatic": Chem.BondType.AROMATIC,
    "dative": Chem.BondType.DATIVE,
}


def cdxml_root(content: str | bytes) -> ET.Element:
    """Parse XML and normalize namespaces while retaining all attributes."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid CDXML: {exc}") from exc
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    if root.tag != "CDXML":
        raise ValueError("The uploaded document is not CDXML.")
    return root


def _integer(value: str, name: str, atom_id: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Atom {atom_id} has invalid {name}: {value!r}.") from exc


@dataclass(frozen=True)
class CdxmlAtom:
    """An actual element node; labels and external connection points are not atoms."""

    id: str
    number: int
    position: tuple[float, float, float]
    charge: int = 0
    isotope: int = 0
    hydrogens: int | None = None
    radical_electrons: int = 0

    @property
    def symbol(self) -> str:
        return chemical_symbols[self.number]


@dataclass(frozen=True)
class CdxmlBond:
    """Keep bond type and dative direction independently of drawing style."""

    id: str
    begin: str
    end: str
    type: str

    @property
    def order(self) -> float:
        # Coordination still connects atoms geometrically, but is not a
        # covalent single bond when determining a donor's implicit hydrogens.
        if self.type == "dative":
            return 1.0
        return float(_BOND_TYPES[self.type]) if self.type != "1.5" else 1.5


@dataclass(frozen=True)
class CdxmlGraph:
    atoms: tuple[CdxmlAtom, ...]
    bonds: tuple[CdxmlBond, ...]
    stereochemistry: bool

    @property
    def index(self) -> dict[str, int]:
        return {atom.id: index for index, atom in enumerate(self.atoms)}

    def rdkit_molecule(self):
        """Build a chemistry model without rewriting the source bond types."""
        molecule = Chem.RWMol()
        for spec in self.atoms:
            atom = Chem.Atom(spec.number)
            atom.SetFormalCharge(spec.charge)
            atom.SetIsotope(spec.isotope)
            atom.SetNumRadicalElectrons(spec.radical_electrons)
            if spec.hydrogens is not None:
                atom.SetNumExplicitHs(spec.hydrogens)
                atom.SetNoImplicit(True)
            elif spec.number not in _NONMETALS or spec.number == 1:
                atom.SetNoImplicit(True)
            molecule.AddAtom(atom)
        indices = self.index
        for spec in self.bonds:
            bond_type = _BOND_TYPES[spec.type]
            first, second = indices[spec.begin], indices[spec.end]
            molecule.AddBond(first, second, bond_type)
            if bond_type == Chem.BondType.AROMATIC:
                molecule.GetAtomWithIdx(first).SetIsAromatic(True)
                molecule.GetAtomWithIdx(second).SetIsAromatic(True)
        molecule = molecule.GetMol()
        # In particular do not silently replace metal single bonds with datives,
        # or rewrite a charged/overvalent drawing into a different Lewis structure.
        operations = Chem.SanitizeFlags.SANITIZE_ALL
        operations ^= Chem.SanitizeFlags.SANITIZE_CLEANUP
        operations &= ~getattr(
            Chem.SanitizeFlags, "SANITIZE_CLEANUP_ORGANOMETALLICS", 0
        )
        try:
            Chem.SanitizeMol(molecule, sanitizeOps=operations)
        except Exception as exc:
            raise ValueError(
                "The CDXML chemical graph cannot be validated: "
                f"{exc}. Check charges, hydrogen labels and bond types; "
                "metal coordination may require a dative bond."
            ) from exc
        return molecule

    def added_hydrogens(self) -> dict[str, int]:
        """Return H from labels and valence, excluding H already drawn as nodes."""
        molecule = self.rdkit_molecule()
        return {
            spec.id: atom.GetNumImplicitHs() + atom.GetNumExplicitHs()
            for spec, atom in zip(self.atoms, molecule.GetAtoms())
        }


def read_cdxml_graph(content: str | bytes) -> CdxmlGraph:
    """Read one selected structure, rejecting uninterpretable chemistry."""
    root = cdxml_root(content)
    nodes = list(root.iter("n"))
    if not nodes:
        raise ValueError("The CDXML contains no atoms.")
    atoms = []
    ids = set()
    for node in nodes:
        identifier = node.get("id")
        if not identifier or identifier in ids:
            raise ValueError(f"Missing or duplicate atom ID: {identifier!r}.")
        ids.add(identifier)
        node_type = node.get("NodeType", "Element")
        if node_type != "Element":
            raise ValueError(
                f"Atom {identifier} uses unsupported NodeType {node_type!r}. "
                "Expand abbreviations into explicit element atoms in ChemDraw."
            )
        number = _integer(node.get("Element", "6"), "element", identifier)
        if not 1 <= number < len(chemical_symbols):
            raise ValueError(
                f"Atom {identifier} has unsupported atomic number {number}."
            )
        try:
            x, y = map(float, node.get("p", "").split())
        except ValueError as exc:
            raise ValueError(
                f"Atom {identifier} has invalid drawing coordinates."
            ) from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Atom {identifier} has nonfinite drawing coordinates.")
        charge = _integer(node.get("Charge", "0"), "charge", identifier)
        isotope = _integer(node.get("Isotope", "0"), "isotope", identifier)
        hydrogens = (
            _integer(node.get("NumHydrogens"), "hydrogen count", identifier)
            if "NumHydrogens" in node.attrib
            else None
        )
        if isotope < 0 or hydrogens is not None and hydrogens < 0:
            raise ValueError(f"Atom {identifier} has a negative isotope or H count.")
        radical = node.get("Radical", "").lower()
        if radical not in _RADICAL_ELECTRONS:
            raise ValueError(f"Atom {identifier} has unsupported radical {radical!r}.")
        electrons = _RADICAL_ELECTRONS[radical]
        if not electrons and any(
            graphic.get("SymbolType") == "Electron"
            and any(
                rep.get("object") == identifier for rep in graphic.iter("represent")
            )
            for graphic in root.iter("graphic")
        ):
            electrons = 1
        atoms.append(
            CdxmlAtom(
                identifier, number, (x, y, 0.0), charge, isotope, hydrogens, electrons
            )
        )

    bonds = []
    bond_ids, connected = set(), set()
    by_id = {atom.id: atom for atom in atoms}
    for index, bond in enumerate(root.iter("b")):
        identifier = bond.get("id", f"bond-{index + 1}")
        if identifier in bond_ids:
            raise ValueError(f"Duplicate bond ID: {identifier!r}.")
        bond_ids.add(identifier)
        first, second = bond.get("B"), bond.get("E")
        if first not in ids or second not in ids:
            raise ValueError(f"Bond {identifier} refers to an unavailable atom.")
        if first == second:
            raise ValueError(f"Bond {identifier} connects atom {first} to itself.")
        pair = frozenset((first, second))
        if pair in connected:
            raise ValueError(
                f"Bond {identifier} duplicates the connection {first}–{second}."
            )
        connected.add(pair)
        if by_id[first].position == by_id[second].position:
            raise ValueError(f"Bond {identifier} has zero drawing length.")
        kind = bond.get("Order", "1").strip().lower()
        if kind not in _BOND_TYPES:
            raise ValueError(f"Bond {identifier} has unsupported order {kind!r}.")
        if kind in {"a", "aromatic"}:
            kind = "1.5"
        bonds.append(CdxmlBond(identifier, first, second, kind))
    stereochemistry = any(
        any(
            word in bond.get("Display", "").lower()
            for word in ("wedge", "hash", "wavy")
        )
        or bond.get("BS", "U") not in {"", "N", "U"}
        for bond in root.iter("b")
    ) or any(node.get("AS", "U") not in {"", "N", "U"} for node in nodes)
    return CdxmlGraph(tuple(atoms), tuple(bonds), stereochemistry)
