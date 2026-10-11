# aiidalab-widgets-empa
Reusable AiiDAlab widgets developed at Empa.

The CDXML importer retains the explicit chemical graph and infers implicit H
from bond orders. Finite molecules retain all those H even when a crowded 2D
drawing puts unrelated atoms close together. An optional **Resolve steric
contacts in 3D** checkbox rotates single C-C connections between cyclic sp2
blocks and terminal methyl hydrogens, keeping rings rigid, bond lengths and
local valence angles unchanged. The chemical-sketch preview stays two
dimensional; the created atomic model contains the resulting starting conformer.

This is a deterministic local torsion search, not a force-field or electronic
structure optimization and not a guarantee of the globally smallest twist.
Its overlap limit is 75% of the sum of ASE van der Waals radii, excluding bonded
and 1-3 pairs. Methyl rotations have a lower penalty than block rotations.
Unresolved contacts are reported and atoms are never deleted to clear them.
The option currently supports finite C/H/halogen molecules without encoded
stereochemistry, with at most twelve supported rotatable connections. Periodic
models use the graph-based planar repeat construction described below.

The **3D method** selector offers **Steric separation** (the default above) or
**Minimal 3D torsion**, which uses editable minimum nonbonded distances of
1.70 Å for C-C, 1.30 Å for C-H and 0.80 Å for H-H. **Extra margin (%)** scales
these distance floors; it starts at zero. Other element pairs use the viewer's
covalent cutoff plus 0.05 Å. Minimal mode checks all nonbonded pairs, including
1-3 pairs, and requires thresholds above the viewer's bond cutoffs. This mode
reduces twisting but can leave van der Waals overlaps: relax the starting model
before a physical calculation. Both methods use the same rigid-block search.

CDXML chemistry is validated with RDKit, now a required dependency. Element
numbers use the complete ASE element table; unsupported or ambiguous node/bond
types produce an error instead of becoming carbon or a single bond. Hydrogen
counts respect `NumHydrogens`, explicit H nodes, formal charge and radical
electrons. Aromatic `1.5` and directed dative bonds are distinct chemical bond
types. Metals do not acquire implicit hydrides; explicitly drawn or labelled
metal hydrogens are retained. Single N/O-metal lines are interpreted as directed
coordination, with a visible warning and a record of the interpreted bonds;
the original source types and declared H remain available. Review this assumption.
Abbreviations must currently be expanded into
explicit element atoms in ChemDraw. Encoded stereochemistry is reported but
is not reconstructed into a stereospecific 3D model.

Without C-C reference bonds, drawing scale is estimated from covalent radii,
not interpreted as angstrom units. This supplies a starting geometry, not
experimentally determined bond lengths. Explicit isotopic masses are retained.
`Atoms.info["cdxml_source"]` records the source hash, original atom IDs,
formal charges, isotopes, radicals, hydrogen labels and directed bond types.
This is a source snapshot, not live connectivity for an edited model; transient
index-based bonds are removed before handing the model to a structure editor.
The snapshot and masses survive Extended XYZ export. Formal charges are not
partial atomic charges and are not automatically translated into calculation
charge or spin settings.

Periodic membership uses `BracketedObjectIDs`, `InnerAtomID` and **all** crossing
bonds, not a coordinate clip or the distance between bracket graphics. Cleanup
preserves identified site equivalences while allowing the cell vector to relax.
Conjugated boundary Kekulé phases can be aligned only while preserving each
site's source H and valence; this is reported and recorded. Ambiguous drawings
show atom indices and allow two equivalent atoms in adjacent cells to be chosen
by fields or by clicking the numbered preview. Incompatible source chemistry
still requires correction in the drawing. Component/mixture/monomer brackets
stay finite; unsupported copolymer and other bracket usages are diagnosed.

Hydrogens are kept by their bonded parent when building a repeat or finite
termination. Close contacts never remove required H. Finite `N units` adds N
copies of the selected repeat between the original drawing's end fragments;
`Infinite` keeps one repeat with 1D PBC. Only positive integers are accepted.
Labels are assigned one-to-one within the same page and a bond-scaled proximity
limit. Unlabelled structures remain numbered, and selected documents retain
only their assigned caption.

For AiiDA storage, call `enable_cdxml_metadata_storage(manager, importer)` on a
widgets-base `StructureManagerWidget`. This uses its normal store action and
records a new reviewed model through `record_cdxml_import`, linked to immutable
original/reviewed `SinglefileData` documents and a `Dict` of source/generation
settings. It records the reviewed geometry; it is not a force-field calculation.
Existing stored nodes are not retrofitted. `restore_cdxml_metadata(atoms, uuid)`
reloads the immutable record at the point of use, without retaining ORM nodes.
Generation mappings and hashes identify the original output; manual edits are
marked so those indices are not presented as current model connectivity.

The public 7AGNR test drawing explicitly marks bond 101 as double. Previously
that omitted order meant single, requiring two methine H which coordinate
clipping silently discarded. Tests also cover the original missing-double case
(C14H6), the corrected independently ASE-checked repeat (C14H4), and retention
of every finite end-fragment H.
