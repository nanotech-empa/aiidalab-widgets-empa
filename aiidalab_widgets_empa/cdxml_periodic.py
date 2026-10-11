"""Identify one-dimensional repeat topology independently of bracket placement."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from rdkit import Chem
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp

from .cdxml_graph import CdxmlGraph, cdxml_root


class PeriodicAmbiguity(ValueError):
    """The drawing does not identify one chemically consistent repeat."""

    def __init__(self, message, atom_ids=()):
        self.atom_ids = tuple(atom_ids)
        super().__init__(message)


@dataclass(frozen=True)
class RepeatEdge:
    begin: str
    end: str
    shift: int
    type: str


@dataclass(frozen=True)
class RepeatUnit:
    atom_ids: tuple[str, ...]
    vector: tuple[float, float, float]
    images: tuple[tuple[str, str, int], ...]
    edges: tuple[RepeatEdge, ...]
    left_outside: tuple[str, ...]
    right_outside: tuple[str, ...]
    resonance_aligned: bool = False

    def vector_for_positions(self, graph: CdxmlGraph, positions) -> np.ndarray:
        """Fit all identified equivalences, without selecting another atom pair."""
        indices = graph.index
        images = [
            item for item in self.images if graph.atoms[indices[item[0]]].number != 1
        ] or self.images
        return np.mean(
            [
                (positions[indices[image]] - positions[indices[central]]) / shift
                for image, central, shift in images
            ],
            axis=0,
        )


def _atom_signature(atom):
    return atom.number, atom.charge, atom.isotope, atom.radical_electrons


def _manual_group(root, graph, equivalent_atoms, length):
    """For plain graphical brackets, infer crossings from an explicit atom pair."""
    import xml.etree.ElementTree as ET

    ids = graph.index
    first, second = equivalent_atoms
    vector = (
        np.asarray(graph.atoms[ids[second]].position) - graph.atoms[ids[first]].position
    )
    norm = np.linalg.norm(vector)
    if norm < length * 0.5:
        raise PeriodicAmbiguity(
            "Choose equivalent sites in two different adjacent cells.", equivalent_atoms
        )
    axis = vector / norm
    origin = np.dot(graph.atoms[ids[first]].position, axis) - 0.25 * length
    central = {
        atom.id
        for atom in graph.atoms
        if origin <= np.dot(atom.position, axis) < origin + norm
    }
    group = ET.Element("bracketedgroup", BracketedObjectIDs=" ".join(sorted(central)))
    left, right = (ET.SubElement(group, "bracketattachment") for _ in range(2))
    for bond in graph.bonds:
        if (bond.begin in central) == (bond.end in central):
            continue
        inner, outer = (
            (bond.begin, bond.end) if bond.begin in central else (bond.end, bond.begin)
        )
        projection = np.dot(graph.atoms[ids[outer]].position, axis)
        attachment = left if projection < origin else right
        ET.SubElement(attachment, "crossingbond", BondID=bond.id, InnerAtomID=inner)
    return group


def _outside_components(graph, central, cuts):
    adjacency = {atom.id: set() for atom in graph.atoms if atom.id not in central}
    for bond in graph.bonds:
        if bond.begin in adjacency and bond.end in adjacency:
            adjacency[bond.begin].add(bond.end)
            adjacency[bond.end].add(bond.begin)
    owners = [{edge[1] for edge in cut} for cut in cuts]
    assigned = [set(), set()]
    unseen = set(adjacency)
    while unseen:
        pending = [next(iter(unseen))]
        component = set()
        while pending:
            atom = pending.pop()
            if atom in component:
                continue
            component.add(atom)
            pending.extend(adjacency[atom] - component)
        unseen -= component
        sides = [
            index for index, identifiers in enumerate(owners) if identifiers & component
        ]
        if len(sides) != 1:
            raise PeriodicAmbiguity(
                "An outside fragment cannot be assigned to one end of the repeat. "
                "Review the selected structure and its crossing bonds.",
                component,
            )
        assigned[sides[0]].update(component)
    return tuple(sorted(assigned[0])), tuple(sorted(assigned[1]))


def _align_resonance(graph, molecule, central, edges, flexible):
    """Align conjugated Kekulé orders while fixing every atom's source valence."""
    reference = Chem.Mol(molecule)
    Chem.Kekulize(reference, clearAromaticFlags=True)
    indices = graph.index
    target = np.asarray(
        [
            sum(
                bond.GetBondTypeAsDouble()
                for bond in reference.GetAtomWithIdx(indices[identifier]).GetBonds()
                if bond.GetBondType() != Chem.BondType.DATIVE
            )
            for identifier in central
        ]
    )
    local = {identifier: index for index, identifier in enumerate(central)}
    variable = [index for index, allowed in enumerate(flexible) if allowed]
    baseline = np.zeros(len(central))
    incidence = np.zeros((len(central), len(variable)))
    for index, edge in enumerate(edges):
        order = (
            1 if index in variable else float(edge.type) if edge.type != "dative" else 0
        )
        baseline[local[edge.begin]] += order
        baseline[local[edge.end]] += order
    for column, index in enumerate(variable):
        edge = edges[index]
        incidence[local[edge.begin], column] += 1
        incidence[local[edge.end], column] += 1
    if not variable:
        raise PeriodicAmbiguity(
            "Crossing bond orders disagree outside a validated conjugated system."
        )
    costs = np.asarray(
        [-1.0 if edges[index].type == "2" else 1.0 for index in variable]
    )
    result = milp(
        costs,
        integrality=np.ones(len(variable)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(incidence, target - baseline, target - baseline),
        options={"time_limit": 5},
    )
    if not result.success:
        raise PeriodicAmbiguity(
            "The periodic bond orders cannot preserve the source valence and H of every atom. "
            "Review equivalent sites, missing bonds or radical symbols before creating a model.",
            central,
        )
    updated = list(edges)
    for index, double in zip(variable, result.x):
        edge = edges[index]
        updated[index] = RepeatEdge(
            edge.begin, edge.end, edge.shift, "2" if double > 0.5 else "1"
        )
    return tuple(updated)


def identify_repeat(
    content, graph: CdxmlGraph, equivalent_atoms=None
) -> RepeatUnit | None:
    """Use all crossing attachments; a manual pair resolves geometric ambiguity."""
    root = cdxml_root(content)
    graph = replace(graph, bonds=graph.chemistry_bonds())
    finite_usages = {"Component", "MixtureOrdered", "MixtureUnordered", "Monomer"}
    for element in root.iter():
        if element.tag == "bracketedgroup" or element.get("BracketType") == "Square":
            usage = element.get("BracketUsage", "Unspecified")
            if usage not in finite_usages | {"SRU", "Unspecified"}:
                raise PeriodicAmbiguity(
                    f"BracketUsage {usage!r} does not identify a supported single structural repeat. Use an SRU with crossing bonds."
                )
    finite_graphics = {
        attachment.get("GraphicID")
        for group in root.iter("bracketedgroup")
        if group.get("BracketUsage") in finite_usages
        for attachment in group.iter("bracketattachment")
    }
    groups = [
        group
        for group in root.iter("bracketedgroup")
        if group.get("BracketUsage", "Unspecified") not in finite_usages
    ]
    has_brackets = any(
        graphic.get("BracketType") == "Square"
        and graphic.get("id") not in finite_graphics
        and graphic.get("BracketUsage", "Unspecified") not in finite_usages
        for graphic in root.iter("graphic")
    )
    if not groups and not has_brackets:
        return None
    ids = graph.index
    positions = np.asarray([atom.position for atom in graph.atoms])
    bond_lengths = [
        np.linalg.norm(positions[ids[bond.begin]] - positions[ids[bond.end]])
        for bond in graph.bonds
    ]
    length = float(np.median(bond_lengths)) if bond_lengths else 1.0
    if equivalent_atoms is not None:
        if len(set(equivalent_atoms)) != 2 or any(
            identifier not in ids for identifier in equivalent_atoms
        ):
            raise PeriodicAmbiguity(
                "Choose two distinct atom indices from this structure."
            )
        if _atom_signature(graph.atoms[ids[equivalent_atoms[0]]]) != _atom_signature(
            graph.atoms[ids[equivalent_atoms[1]]]
        ):
            raise PeriodicAmbiguity(
                "The selected atoms have different elements, charges, isotopes or radical states.",
                equivalent_atoms,
            )
    if not groups:
        if equivalent_atoms is None:
            raise PeriodicAmbiguity(
                "Graphical brackets have no usable repeat metadata. "
                "Select two equivalent atoms in adjacent cells to identify the repeat."
            )
        group = _manual_group(root, graph, equivalent_atoms, length)
    elif len(groups) != 1:
        raise PeriodicAmbiguity(
            "Select a structure with one repeat group; nested/multiple polymers are not supported."
        )
    else:
        group = groups[0]
    central = tuple(group.get("BracketedObjectIDs", "").split())
    if (
        not central
        or len(set(central)) != len(central)
        or any(identifier not in ids for identifier in central)
    ):
        raise PeriodicAmbiguity(
            "The repeat group has missing, duplicate or unavailable atom IDs."
        )
    # Source order, rather than graphical position or XML group order, fixes membership.
    central = tuple(atom.id for atom in graph.atoms if atom.id in set(central))
    central_set = set(central)
    by_bond = {bond.id: bond for bond in graph.bonds}
    molecule = graph.rdkit_molecule()
    kekule = Chem.Mol(molecule)
    Chem.Kekulize(kekule, clearAromaticFlags=True)
    cuts = []
    crossing_ids = set()
    for attachment in group.iter("bracketattachment"):
        crossing_edges = []
        for crossing in attachment.iter("crossingbond"):
            bond = by_bond.get(crossing.get("BondID"))
            inner = crossing.get("InnerAtomID")
            if (
                bond is None
                or inner not in central_set
                or inner not in {bond.begin, bond.end}
            ):
                raise PeriodicAmbiguity(
                    "Invalid crossing-bond or inner-atom reference."
                )
            outer = bond.end if bond.begin == inner else bond.begin
            if outer in central_set:
                raise PeriodicAmbiguity(
                    "A crossing bond has both endpoints inside the repeat."
                )
            source_bond = kekule.GetBondBetweenAtoms(ids[bond.begin], ids[bond.end])
            kind = (
                str(int(source_bond.GetBondTypeAsDouble()))
                if bond.type == "1.5"
                else bond.type
            )
            crossing_edges.append((inner, outer, kind, bond.id))
            crossing_ids.add(bond.id)
        if crossing_edges:
            cuts.append(crossing_edges)
    if len(cuts) != 2 or len(cuts[0]) != len(cuts[1]):
        raise PeriodicAmbiguity(
            "A 1D repeat needs two sides with matching crossing bonds. Review the atom pair or repeat selection."
        )
    boundary = {
        bond.id
        for bond in graph.bonds
        if (bond.begin in central_set) != (bond.end in central_set)
    }
    if boundary != crossing_ids:
        raise PeriodicAmbiguity(
            "The crossing metadata does not describe every bond leaving the repeat."
        )

    def midpoint(edge):
        return (positions[ids[edge[0]]] + positions[ids[edge[1]]]) / 2

    cuts.sort(
        key=lambda edges: tuple(np.mean([midpoint(edge) for edge in edges], axis=0))
    )
    left, right = cuts
    initial_vector = np.mean([midpoint(edge) for edge in right], axis=0) - np.mean(
        [midpoint(edge) for edge in left], axis=0
    )
    if equivalent_atoms is not None:
        hint = positions[ids[equivalent_atoms[1]]] - positions[ids[equivalent_atoms[0]]]
        if np.dot(hint, initial_vector) < 0:
            hint = -hint
        if np.linalg.norm(hint - initial_vector) > length * 0.6:
            raise PeriodicAmbiguity(
                "The selected pair does not match adjacent cells of this repeat group.",
                equivalent_atoms,
            )
        initial_vector = hint
    costs = np.full((len(left), len(right)), np.inf)
    incompatible = []
    for row, first in enumerate(left):
        for column, second in enumerate(right):
            pairs = ((first[1], second[0]), (second[1], first[0]))
            if any(
                _atom_signature(graph.atoms[ids[outer]])
                != _atom_signature(graph.atoms[ids[inner]])
                for outer, inner in pairs
            ):
                continue
            # Cropping may remove neighbors from peripheral image atoms, but
            # cannot explain MORE bonds than the supposed interior equivalent.
            if any(
                molecule.GetAtomWithIdx(ids[outer]).GetDegree()
                > molecule.GetAtomWithIdx(ids[inner]).GetDegree()
                for outer, inner in pairs
            ):
                incompatible.extend(pairs)
                continue
            first_vector = positions[ids[second[0]]] - positions[ids[first[1]]]
            second_vector = positions[ids[second[1]]] - positions[ids[first[0]]]
            costs[row, column] = (
                np.linalg.norm(first_vector - initial_vector) ** 2
                + np.linalg.norm(second_vector - initial_vector) ** 2
            )
    try:
        rows, columns = linear_sum_assignment(costs)
    except ValueError as exc:
        raise PeriodicAmbiguity(
            "Crossing sites are chemically incompatible; review their bonds and select equivalent atoms.",
            [identifier for pair in incompatible for identifier in pair],
        ) from exc
    vectors = []
    images = []
    edges = []
    flexible = []
    disagreement = False
    for row, column in zip(rows, columns):
        first, second = left[row], right[column]
        images.extend(((first[1], second[0], -1), (second[1], first[0], 1)))
        vectors.extend(
            (
                positions[ids[second[0]]] - positions[ids[first[1]]],
                positions[ids[second[1]]] - positions[ids[first[0]]],
            )
        )
        if first[2] == "dative" and by_bond[first[3]].begin == first[1]:
            edges.append(RepeatEdge(second[0], first[0], 1, "dative"))
        else:
            edges.append(RepeatEdge(first[0], second[0], -1, first[2]))
        source_bonds = [
            molecule.GetBondBetweenAtoms(ids[edge[0]], ids[edge[1]])
            for edge in (first, second)
        ]
        flexible.append(
            first[2] in {"1", "2"}
            and any(
                bond.GetIsAromatic() or bond.GetIsConjugated() for bond in source_bonds
            )
        )
        disagreement |= first[2] != second[2]
    vector = np.mean(vectors, axis=0)
    if (
        np.linalg.norm(vector) < length * 0.5
        or max(np.linalg.norm(value - vector) for value in vectors) > length * 0.4
    ):
        raise PeriodicAmbiguity(
            "The crossing correspondences do not give one consistent translation. Select equivalent atoms in adjacent cells.",
            [identifier for image, inner, _ in images for identifier in (image, inner)],
        )
    for bond in graph.bonds:
        if bond.begin in central_set and bond.end in central_set:
            original = molecule.GetBondBetweenAtoms(ids[bond.begin], ids[bond.end])
            kind = (
                str(
                    int(
                        kekule.GetBondBetweenAtoms(
                            ids[bond.begin], ids[bond.end]
                        ).GetBondTypeAsDouble()
                    )
                )
                if bond.type == "1.5"
                else bond.type
            )
            edges.append(RepeatEdge(bond.begin, bond.end, 0, kind))
            flexible.append(
                kind in {"1", "2"}
                and (original.GetIsAromatic() or original.GetIsConjugated())
            )
    if disagreement:
        edges = _align_resonance(graph, molecule, central, edges, flexible)
    left_outside, right_outside = _outside_components(graph, central_set, cuts)
    repeat = RepeatUnit(
        central,
        tuple(vector),
        tuple(images),
        tuple(edges),
        left_outside,
        right_outside,
        disagreement,
    )
    validate_repeat_chemistry(graph, repeat)
    return repeat


def validate_repeat_chemistry(graph: CdxmlGraph, repeat: RepeatUnit):
    """Validate the bulk graph and H per site on a closed four-cell cover."""
    source = graph.rdkit_molecule()
    source_index = graph.index
    local = {identifier: index for index, identifier in enumerate(repeat.atom_ids)}
    size = len(local)
    cover = Chem.RWMol()
    for cell in range(4):
        for identifier in repeat.atom_ids:
            atom = Chem.Atom(source.GetAtomWithIdx(source_index[identifier]))
            atom.SetIsAromatic(False)
            cover.AddAtom(atom)
    for cell in range(4):
        for edge in repeat.edges:
            first = cell * size + local[edge.begin]
            second = ((cell + edge.shift) % 4) * size + local[edge.end]
            kind = {
                "1": Chem.BondType.SINGLE,
                "2": Chem.BondType.DOUBLE,
                "3": Chem.BondType.TRIPLE,
                "dative": Chem.BondType.DATIVE,
            }[edge.type]
            if cover.GetBondBetweenAtoms(first, second) is not None:
                raise PeriodicAmbiguity(
                    "Two crossing connections produce the same periodic bond; review the equivalent atoms."
                )
            cover.AddBond(first, second, kind)
    result = cover.GetMol()
    try:
        operations = (
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_CLEANUP
        )
        operations &= ~getattr(
            Chem.SanitizeFlags, "SANITIZE_CLEANUP_ORGANOMETALLICS", 0
        )
        Chem.SanitizeMol(result, sanitizeOps=operations)
    except Exception as exc:
        raise PeriodicAmbiguity(
            "The identified periodic graph has inconsistent valence or bond orders. Review the crossing connections."
        ) from exc
    changed = []
    for cell in range(4):
        for index, identifier in enumerate(repeat.atom_ids):
            reference = source.GetAtomWithIdx(source_index[identifier])
            atom = result.GetAtomWithIdx(cell * size + index)
            if (
                atom.GetNumImplicitHs() + atom.GetNumExplicitHs()
                != reference.GetNumImplicitHs() + reference.GetNumExplicitHs()
                or atom.GetNumRadicalElectrons() != reference.GetNumRadicalElectrons()
            ):
                changed.append(identifier)
    if changed:
        raise PeriodicAmbiguity(
            "Periodic closure would change H or radicals on source atoms. Review the selected equivalent sites and bond orders.",
            changed,
        )
    return result


def build_repeat_model(whole, graph: CdxmlGraph, repeat: RepeatUnit, units="Infinite"):
    """Keep H by graph ownership and replicate without clipping by coordinates."""
    import copy

    from ase import Atoms

    if units is None or str(units).strip().lower() == "infinite":
        count = None
    else:
        try:
            count = int(units)
        except (TypeError, ValueError) as exc:
            raise ValueError("N units must be Infinite or a positive integer.") from exc
        if count < 1 or str(count) != str(units).strip():
            raise ValueError("N units must be Infinite or a positive integer.")
    ids = graph.index
    central = set(repeat.atom_ids)
    parents = {}
    for first, second, _ in whole.info.get("_cdxml_bonds", []):
        if whole[second].symbol == "H" and whole[first].symbol != "H":
            parents[second] = first
        elif whole[first].symbol == "H" and whole[second].symbol != "H":
            parents[first] = second
    index_to_id = {index: atom.id for index, atom in enumerate(graph.atoms)}

    def owned(atom_ids):
        selected = {ids[identifier] for identifier in atom_ids}
        for hydrogen, parent in parents.items():
            if index_to_id[parent] in atom_ids:
                selected.add(hydrogen)
            else:
                selected.discard(hydrogen)
        return sorted(selected)

    body = owned(central)
    left = owned(set(repeat.left_outside))
    right = owned(set(repeat.right_outside))
    total = len(body) * (count or 1) + (
        len(left) + len(right) if count is not None else 0
    )
    if total > 20000:
        raise ValueError("The requested model exceeds 20000 atoms; reduce N units.")
    vector = repeat.vector_for_positions(graph, whole.positions[: len(graph.atoms)])
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        raise ValueError("The periodic translation has zero length.")
    blocks = [(body, cell * vector) for cell in range(count or 1)]
    if count is not None:
        blocks.extend(((left, np.zeros(3)), (right, (count - 1) * vector)))
    symbols = []
    positions = []
    masses = []
    for selected, offset in blocks:
        symbols.extend(whole[index].symbol for index in selected)
        positions.extend(whole.positions[selected] + offset)
        masses.extend(whole.get_masses()[selected])
    model = Atoms(symbols, positions=positions, masses=masses)
    model.info = copy.deepcopy(whole.info)
    model.info.pop("_cdxml_bonds", None)
    model.info.pop("_cdxml_stereochemistry", None)
    atom_sources = model.info.pop("_cdxml_atom_sources", None)
    model.rotate(-np.degrees(np.arctan2(vector[1], vector[0])), "z", center=(0, 0, 0))
    transverse = np.ptp(model.positions, axis=0)
    model.cell = [
        length if count is None else transverse[0] + 15,
        transverse[1] + 15,
        transverse[2] + 15,
    ]
    model.pbc = [count is None, False, False]
    model.center()
    charge_by_id = {atom.id: atom.charge for atom in graph.atoms}
    charge = sum(charge_by_id[identifier] for identifier in central) * (count or 1)
    if count is not None:
        charge += sum(
            charge_by_id[identifier]
            for identifier in (*repeat.left_outside, *repeat.right_outside)
        )
    model.info["cdxml_generation"] = {
        "periodic": count is None,
        "units": "Infinite" if count is None else count,
        "formal_charge": charge,
        "repeat_source_ids": list(repeat.atom_ids),
        "resonance_aligned": repeat.resonance_aligned,
        "repeat_edges": [
            {
                "begin": edge.begin,
                "end": edge.end,
                "shift": edge.shift,
                "type": edge.type,
            }
            for edge in repeat.edges
        ],
        "equivalent_images": [list(item) for item in repeat.images],
        "translation": vector.tolist(),
    }
    if atom_sources is not None:
        model.info["cdxml_generation"]["atom_sources"] = [
            {**atom_sources[index], "translation_cells": cell}
            for selected, cell in [(body, cell) for cell in range(count or 1)]
            + ([(left, 0), (right, count - 1)] if count is not None else [])
            for index in selected
        ]
    return model
