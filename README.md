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
models continue to use the existing planar importer.
