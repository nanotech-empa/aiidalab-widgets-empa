from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Optional, Tuple, List

import ase
import ipywidgets as ipw
import numpy as np
import traitlets as tr
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList

"""Widget to convert CDXML to planar structures"""

"""Widget to convert CDXML files into planar ASE.Atoms structures with alignment,
trimming, hydrogen restoration, and optional replication along the periodic axis."""


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


# ---------------- Main Widget ----------------
class CdxmlUploadWidget(ipw.VBox):
    """Widget for uploading CDXML files and converting them into ASE.Atoms structures."""

    structure = tr.Instance(ase.Atoms, allow_none=True)

    def __init__(
        self, title: str = "CDXML to GNR", description: str = "Upload Structure"
    ):
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
        self.nunits = ipw.Text(description="N units", value="Infinite", disabled=True)
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
                self.nunits,
                supported_formats,
                self.create_button,
                self.output_message,
            ]
        )

        # --- Internal state ---
        self.structure: Optional[ase.Atoms] = None
        self.crossing_points: Optional[np.ndarray] = None
        self.cdxml_atoms: Optional[np.ndarray] = None
        self.atoms: Optional[ase.Atoms] = None
        self.whole_atoms: Optional[ase.Atoms] = None

    # ---------------- Event handlers ----------------
    def _on_file_upload(self, change=None) -> None:
        """Handle file upload and convert CDXML to ASE Atoms."""
        self.nunits.value = "Infinite"
        self.nunits.disabled = True

        def get_unified_representation(value):
            """Compatibility wrapper for ipywidgets 7.x and 8.x."""
            try:
                return [(fname, item["content"]) for fname, item in value.items()]
            except AttributeError:
                return [(f["name"], f.content.tobytes()) for f in value]

        _, cdxml_content = get_unified_representation(change["new"])[0]

        try:
            self.output_message.value, self.atoms, self.whole_atoms = (
                self.cdxml_to_ase_from_string(cdxml_content)
            )
            self.crossing_points, self.cdxml_atoms, self.nunits.disabled = (
                self.extract_crossing_and_atom_positions(cdxml_content)
            )
        except ValueError as exc:
            self.output_message.value = f"Error: {exc}"
        except Exception as exc:
            self.output_message.value = f"Unexpected error: {exc}"

    def _on_button_click(self, _=None) -> None:
        """Create the ASE model when 'Create model' button is clicked."""
        if not self.atoms:
            self.output_message.value = "Error: No atoms available to process."
            return

        atoms = self.atoms.copy()

        if self.crossing_points is not None:
            crossing_points = self.transform_points(
                self.cdxml_atoms, atoms.positions, self.crossing_points
            )
            atoms = self.align_and_trim_atoms(
                self.whole_atoms, np.array(crossing_points), units=self.nunits.value
            )
        else:
            self.output_message.value = "Error: No 'crossing points' found."
            return

        if self.nunits.disabled:
            extra_cell = 15.0
            atoms.cell = np.ptp(atoms.positions, axis=0) + extra_cell
            atoms.center()

        if self.nunits.value == "Infinite":
            atoms.pbc = True

        self.structure = atoms
        self.output_message.value = "✅ 3D structure created"

    # ---------------- Core conversion logic ----------------
    @staticmethod
    def cdxml_to_ase_from_string(
        cdxml_content: str, target_cc_length: float = 1.43
    ) -> Tuple[str, Atoms, Atoms]:
        """
        Convert CDXML content (string) into ASE Atoms objects:
        one bare molecule (no hydrogens) and one with hydrogens.
        """
        root = ET.fromstring(cdxml_content)

        # --- Element and bond maps ---
        element_map = {
            "1": "H",
            "5": "B",
            "6": "C",
            "7": "N",
            "8": "O",
            "9": "F",
            "15": "P",
            "16": "S",
            "17": "Cl",
            "26": "Fe",
            "27": "Co",
            "28": "Ni",
            "22": "Ti",
            "40": "Zr",
            "65": "Tb",
            "35": "Br",
            "53": "I",
        }

        default_valence = {
            "H": 1,
            "B": 3,
            "C": 4,
            "N": 3,
            "O": 2,
            "F": 1,
            "P": 3,
            "S": 2,
            "Cl": 1,
            "Br": 1,
            "I": 1,
            "Fe": 2,
            "Co": 2,
            "Ni": 2,
            "Ti": 4,
            "Zr": 4,
            "Tb": 3,
        }

        bond_order_map = {
            "1": 1.0,  # single bond
            "2": 2.0,  # double bond
            "3": 3.0,  # triple bond
            "A": 1.5,  # aromatic bond
        }

        atoms, bonds, radicals = {}, [], set()

        # --- Parse atoms ---
        for n in root.iter("n"):
            a_id = n.attrib["id"]
            el = element_map.get(n.attrib.get("Element", ""), "C")
            x, y = map(float, n.attrib["p"].split())
            atoms[a_id] = {"el": el, "pos": np.array([x, y, 0.0])}
            if "Radical" in n.attrib:
                radicals.add(a_id)

        for g in root.iter("graphic"):
            if g.attrib.get("SymbolType") == "Electron":
                for rep in g.iter("represent"):
                    target = rep.attrib.get("object")
                    if target:
                        radicals.add(target)

        # --- Parse bonds ---
        for b in root.iter("b"):
            a1, a2 = b.attrib["B"], b.attrib["E"]
            order = bond_order_map.get(b.attrib.get("Order", "1"), 1.0)
            bonds.append({"a1": a1, "a2": a2, "order": order})

        # --- Determine scaling from one C=C bond ---
        scale = 1.0
        for b in bonds:
            if b["order"] == 2.0:
                if atoms[b["a1"]]["el"] == atoms[b["a2"]]["el"] == "C":
                    d = np.linalg.norm(atoms[b["a1"]]["pos"] - atoms[b["a2"]]["pos"])
                    scale = target_cc_length / d
                    break
        for a in atoms.values():
            a["pos"] *= scale

        # --- Connectivity ---
        conn = {k: [] for k in atoms}
        for b in bonds:
            conn[b["a1"]].append((b["a2"], b["order"]))
            conn[b["a2"]].append((b["a1"], b["order"]))

        # --- Bare molecule ---
        bare_pos = [a["pos"] for a in atoms.values()]
        bare_sym = [a["el"] for a in atoms.values()]
        bare_mol = Atoms(symbols=bare_sym, positions=bare_pos)

        # --- Determine implicit hydrogens ---
        impl_H = {}
        for aid, a in atoms.items():
            el = a["el"]
            total = sum(o for _, o in conn[aid])
            val = default_valence.get(el, 4)
            if aid in radicals:
                val -= 1
            impl_H[aid] = max(0, round(val - total))

        # --- Add hydrogens ---
        pos, sym = list(bare_pos), list(bare_sym)
        for aid, nH in impl_H.items():
            if nH == 0:
                continue

            el = atoms[aid]["el"]
            c = atoms[aid]["pos"]
            neighbors = [normalize(atoms[n]["pos"] - c) for n, _ in conn[aid]]
            orders = [o for _, o in conn[aid]]

            def add_H(vecs: List[np.ndarray], length: float = 1.09):
                for v in vecs:
                    pos.append(c + length * v)
                    sym.append("H")

            if el != "C":
                avg = (
                    normalize(-np.sum(neighbors, axis=0))
                    if neighbors
                    else np.array([0, 0, 1])
                )
                add_H([avg], 1.01)
                continue

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
                        add_H([R @ d for d in dirs], CH_len)
                    continue

                if len(neighbors) == 2:
                    v1, v2 = neighbors
                    bis = normalize(-(v1 + v2))
                    plane_normal = normalize(np.cross(v1, v2))
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
                        perp = normalize(np.cross(v1, v2))
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
                avg = normalize(-np.sum(neighbors, axis=0))
                add_H([avg], 1.09)

        mol = Atoms(symbols=sym, positions=pos)
        msg = "✅ Ready to create the structure"
        return msg, bare_mol, mol

    # ---------------- Geometry utilities ----------------
    @staticmethod
    def transform_points(
        set1: np.ndarray, set2: np.ndarray, points: np.ndarray
    ) -> List[List[float]]:
        """Transform points based on scaling and rotation aligning set1→set2."""
        centroid1, centroid2 = np.mean(set1, axis=0), np.mean(set2, axis=0)
        centered1, centered2 = set1 - centroid1, set2 - centroid2
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
        self, cdxml_content: str
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Extract crossing points and atom positions from CDXML."""
        root = ET.fromstring(cdxml_content)

        atom_positions, atom_id_map = [], {}
        for node in root.findall(".//n"):
            atom_id = node.get("id")
            if atom_id and "p" in node.attrib:
                x, y = map(float, node.attrib["p"].split())
                atom_positions.append((x, y, 0.0))
                atom_id_map[atom_id] = len(atom_positions) - 1
        atom_positions = np.array(atom_positions)

        crossing_points = []
        for crossing in root.findall(".//crossingbond"):
            bond_id = crossing.get("BondID")
            if not bond_id:
                continue
            bond = root.find(f".//b[@id='{bond_id}']")
            if bond is not None:
                s, e = bond.get("B"), bond.get("E")
                if s in atom_id_map and e in atom_id_map:
                    s_pos, e_pos = (
                        atom_positions[atom_id_map[s]],
                        atom_positions[atom_id_map[e]],
                    )
                    midpoint = (
                        (s_pos[0] + e_pos[0]) / 2,
                        (s_pos[1] + e_pos[1]) / 2,
                        0.0,
                    )
                    crossing_points.append(midpoint)
        crossing_points = np.array(crossing_points)

        brackets = []
        for g in root.findall(".//graphic[@BracketType='Square']"):
            if "BoundingBox" in g.attrib:
                x_min, y_min, x_max, y_max = map(float, g.attrib["BoundingBox"].split())
                brackets.append(((x_min + x_max) / 2, (y_min + y_max) / 2, 0.0))

        if not brackets:
            return self.max_extension_points(atom_positions), atom_positions, True

        if len(brackets) == 2:
            brackets = np.array(brackets)
            vector = brackets[1] - brackets[0]
            unit_vec = vector[:2] / np.linalg.norm(vector[:2])
            if unit_vec[0] < 0 or unit_vec[1] < 0:
                unit_vec = -unit_vec
            for i in range(len(crossing_points)):
                for j in range(i + 1, len(crossing_points)):
                    v = crossing_points[j][:2] - crossing_points[i][:2]
                    if np.dot(v / np.linalg.norm(v), unit_vec) > 0.99:
                        return crossing_points[[i, j]], atom_positions, False

        return None, atom_positions, True

    # ---------------- Alignment & trimming ----------------
    @staticmethod
    def align_and_trim_atoms(
        atoms: Atoms,
        crossing_points: np.ndarray,
        units: Optional[str] = None,
        original_atoms: Optional[Atoms] = None,
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

        try:
            n_units = int(units)
        except (TypeError, ValueError):
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

        if n_units is None or n_units < 1:
            l1, atoms_final.pbc = norm_vector, True
        else:
            l1 = np.ptp(atoms_final.get_positions()[:, 0]) + 15.0
        l2 = 15.0 + np.ptp(atoms_final.get_positions()[:, 1])
        l3 = 15.0
        atoms_final.set_cell([l1, l2, l3])
        atoms_final.center()

        return atoms_final
