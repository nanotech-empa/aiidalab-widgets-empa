"""Conservative torsion-only relief of crowded finite CDXML drawings.

This produces a starting conformer, not an energy minimum. The explicit
chemical graph is authoritative: contacts never add bonds or remove atoms.
Only single C-C bridges between cyclic blocks and terminal methyl groups are
rotated. Ring coordinates, bond lengths and local valence angles stay fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.data import vdw_radii
from scipy.optimize import least_squares, minimize


@dataclass
class StericResult:
    """Result and diagnostics for a local, deterministic conformer search."""

    atoms: Atoms
    initial_contacts: int
    remaining_contacts: int
    torsions: list[tuple[int, int, float]]
    note: str = ""

    @property
    def message(self) -> str:
        if self.note:
            return self.note
        if not self.initial_contacts:
            return "No steric contacts need 3D relief."
        angles = ", ".join(f"{abs(angle):.1f}°" for _, _, angle in self.torsions)
        status = (
            f"{self.remaining_contacts} steric contacts remain; inspect the model."
            if self.remaining_contacts
            else "Steric contact limits satisfied."
        )
        return (
            f"3D torsion relief: {self.initial_contacts} → "
            f"{self.remaining_contacts} contacts. {status} "
            f"Rotations: {angles or 'none'}. Starting conformer; not energy optimized."
        )


def _side(adjacency, start, blocked):
    reached, pending = set(), [start]
    while pending:
        atom = pending.pop()
        if atom in reached:
            continue
        reached.add(atom)
        pending.extend(
            neighbor
            for neighbor in adjacency[atom]
            if frozenset((atom, neighbor)) != blocked and neighbor not in reached
        )
    return reached


def _torsions(atoms, bonds, adjacency):
    bridges, ring_atoms = [], set()
    for first, second, order in bonds:
        side = _side(adjacency, second, frozenset((first, second)))
        if first not in side:
            bridges.append((first, second, order, side))
        else:
            ring_atoms.update((first, second))

    symbols = atoms.get_chemical_symbols()
    sp2_atoms = {
        atom
        for atom in ring_atoms
        if symbols[atom] == "C"
        and len(adjacency[atom]) == 3
        and any(
            atom in (first, second) and 1.5 <= order <= 2
            for first, second, order in bonds
        )
    }
    rotations = []
    for first, second, order, side in bridges:
        if order != 1 or symbols[first] != "C" or symbols[second] != "C":
            continue
        methyl = next(
            (
                end
                for end, other in ((first, second), (second, first))
                if other in sp2_atoms
                and len(adjacency[end]) == 4
                and sum(symbols[i] == "H" for i in adjacency[end]) == 3
            ),
            None,
        )
        if methyl is not None:
            moving = {i for i in adjacency[methyl] if symbols[i] == "H"}
            limit = np.pi / 3  # three indistinguishable H: period 120 degrees
        elif first in sp2_atoms and second in sp2_atoms:
            other_side = _side(adjacency, first, frozenset((first, second)))
            moving = min(
                (side, other_side),
                key=lambda block: (sum(symbols[i] != "H" for i in block), min(block)),
            )
            limit = np.pi
        else:
            continue
        mask = np.zeros(len(atoms), dtype=bool)
        mask[list(moving)] = True
        rotations.append((first, second, mask, limit))
    return rotations


def _positions(original, rotations, angles):
    positions = original.copy()
    for (first, second, mask, _), angle in zip(rotations, angles):
        origin = positions[first].copy()
        axis = positions[second] - origin
        axis /= np.linalg.norm(axis)
        relative = positions[mask] - origin
        positions[mask] = (
            origin
            + relative * np.cos(angle)
            + np.cross(axis, relative) * np.sin(angle)
            + np.outer(relative @ axis, axis) * (1 - np.cos(angle))
        )
    return positions


def relieve_steric_contacts(
    atoms: Atoms,
    bonds: list[tuple[int, int, float]],
    contact_scale: float = 0.75,
    resolve: bool = True,
) -> StericResult:
    """Find small bridge torsions while retaining every atom and explicit bond.

    Exclude bonded and angle-related (1-3) pairs. The contact floor is 75% of
    the sum of ASE van der Waals radii, a documented overlap heuristic rather
    than a force field. Optimize all movable contacts together, then minimize
    squared rotations subject to those floors, with a smaller penalty for
    methyl-H rotations to prefer moving H over tilting whole cyclic blocks.
    Four deterministic starts help
    escape the stationary planar geometry; no global minimum is claimed.

    Unsupported chemistry and contacts within a rigid block are reported, never
    corrected by bending rings, inventing bonds, or deleting hydrogens.
    Set resolve=False to inspect the same contact criteria without moving atoms.
    """
    result = StericResult(atoms.copy(), 0, 0, [])
    if np.any(atoms.pbc):
        result.note = "3D torsion relief supports finite molecules only."
        return result
    if not 0 < contact_scale <= 1:
        raise ValueError("contact_scale must lie in (0, 1].")
    if not set(atoms.get_chemical_symbols()) <= {"C", "H", "F", "Cl", "Br", "I"}:
        result.note = "3D torsion relief currently supports C/H/halogen molecules only."
        return result
    adjacency = [set() for _ in atoms]
    for first, second, _ in bonds:
        adjacency[first].add(second)
        adjacency[second].add(first)
        if np.linalg.norm(atoms.positions[first] - atoms.positions[second]) < 1e-8:
            result.note = "3D relief skipped: an explicit bond has zero length."
            return result
    pairs = [
        (first, second)
        for first in range(len(atoms))
        for second in range(first + 1, len(atoms))
        if second not in adjacency[first]
        and not adjacency[first].intersection(adjacency[second])
    ]
    if not pairs:
        return result
    pairs = np.asarray(pairs)
    radii = vdw_radii[atoms.numbers]
    floors = contact_scale * (radii[pairs[:, 0]] + radii[pairs[:, 1]])

    def distances(positions):
        return np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)

    tolerance = 1e-4  # angstrom, optimization acceptance only
    result.initial_contacts = int(
        np.count_nonzero(distances(atoms.positions) < floors - tolerance)
    )
    result.remaining_contacts = result.initial_contacts
    if not result.initial_contacts:
        return result
    if not resolve:
        result.note = (
            f"{result.initial_contacts} close nonbonded contacts in the planar "
            "starting geometry. All hydrogens retained; enable 3D steric relief "
            "or inspect the model."
        )
        return result
    rotations = _torsions(atoms, bonds, adjacency)
    if not rotations or len(rotations) > 12:
        result.note = (
            f"{result.initial_contacts} steric contacts remain; no supported small "
            "torsion search is available. All atoms retained; inspect the model."
        )
        return result
    movable = np.zeros(len(pairs), dtype=bool)
    for first, second, mask, _ in rotations:
        movable |= (
            (mask[pairs[:, 0]] != mask[pairs[:, 1]])
            & ~np.isin(pairs[:, 0], (first, second))
            & ~np.isin(pairs[:, 1], (first, second))
        )
    if not np.any(movable):
        result.note = f"{result.initial_contacts} steric contacts lie within rigid blocks; all atoms retained."
        return result
    lower = -np.asarray([rotation[3] for rotation in rotations])
    upper = -lower
    weights = np.asarray(
        [0.01 if limit == np.pi / 3 else 1.0 for *_, limit in rotations]
    )
    original = atoms.positions.copy()

    def gaps(angles):
        return (
            distances(_positions(original, rotations, angles))[movable]
            - floors[movable]
        )

    def residual(angles):
        return np.concatenate(
            (np.minimum(gaps(angles), 0), 0.005 * np.sqrt(weights) * angles)
        )

    best_angles = np.zeros(len(rotations))
    best_key = (float("inf"), float("inf"), float("inf"))
    for signs in (
        np.ones(len(rotations)),
        (-1.0) ** np.arange(len(rotations)),
        -np.ones(len(rotations)),
        -((-1.0) ** np.arange(len(rotations))),
    ):
        seed = signs * np.minimum(upper * 0.5, np.pi / 4)
        fitted = least_squares(residual, seed, bounds=(lower, upper), max_nfev=150)
        minimized = minimize(
            lambda angles: float(angles @ (weights * angles)),
            fitted.x,
            jac=lambda angles: 2 * weights * angles,
            bounds=list(zip(lower, upper)),
            constraints={"type": "ineq", "fun": gaps},
            method="SLSQP",
            options={"maxiter": 200, "ftol": 1e-9},
        )
        for candidate in (fitted.x, minimized.x):
            overlaps = np.maximum(-gaps(candidate) - tolerance, 0)
            key = (
                bool(np.any(overlaps)),
                float(overlaps @ overlaps),
                float(candidate @ (weights * candidate)),
            )
            if key < best_key:
                best_key, best_angles = key, candidate.copy()
    result.atoms.positions = _positions(original, rotations, best_angles)
    result.remaining_contacts = int(
        np.count_nonzero(distances(result.atoms.positions) < floors - tolerance)
    )
    result.torsions = [
        (first, second, float(np.degrees(angle)))
        for (first, second, _, _), angle in zip(rotations, best_angles)
        if abs(angle) > 1e-6
    ]
    return result
