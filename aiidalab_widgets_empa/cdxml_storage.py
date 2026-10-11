"""Preserve reviewed CDXML source metadata when storing a new atomic model."""

from __future__ import annotations

import copy
import hashlib
import json
from io import BytesIO
from pathlib import Path

from aiida.engine import calcfunction


@calcfunction
def record_cdxml_import(
    original_document, reviewed_document, parameters, reviewed_model
):
    """Record a user-reviewed model and its source documents without changing it."""
    return reviewed_model.clone()


def model_fingerprint(atoms):
    """Associate generation metadata with one atom order, geometry and cell."""
    payload = {
        "numbers": atoms.numbers.tolist(),
        "positions": atoms.positions.round(8).tolist(),
        "masses": atoms.get_masses().round(8).tolist(),
        "cell": atoms.cell.array.round(8).tolist(),
        "pbc": atoms.pbc.tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def restore_cdxml_metadata(atoms, node_uuid):
    """Reload source snapshots at the point of use; return only plain ASE data."""
    from aiida.orm import load_node

    node = load_node(node_uuid)
    creator = node.creator
    record = (
        creator.inputs.parameters.get_dict()
        if creator is not None and creator.process_label == "record_cdxml_import"
        else node.base.extras.get("cdxml_import", None)
    )
    del creator, node
    if not isinstance(record, dict) or record.get("schema") != 1:
        return atoms
    result = atoms.copy()
    result.info["cdxml_source"] = copy.deepcopy(record["source"])
    result.info["cdxml_generation"] = copy.deepcopy(record["generation"])
    result.info["cdxml_document_refs"] = copy.deepcopy(record["documents"])
    return result


def enable_cdxml_metadata_storage(manager, importer):
    """Wrap a StructureManager store callback, preserving its normal PK feedback.

    Only newly stored nodes receive metadata. Source documents are immutable
    SinglefileData nodes linked through an explicit recording calcfunction.
    That function copies the user-reviewed geometry; it does not claim an
    energy calculation or an automatic reconstruction of manual edits.
    No ORM node is retained by this hook.
    """
    if getattr(manager, "_cdxml_metadata_storage_enabled", False):
        return
    manager._cdxml_metadata_storage_enabled = True
    manager.btn_store.on_click(manager.store_structure, remove=True)

    def document(content, filename):
        from aiida.common.exceptions import NotExistent
        from aiida.manage import get_manager
        from aiida.orm import SinglefileData, load_node

        profile_uuid = str(get_manager().get_profile().uuid)
        checksum = hashlib.sha256(content).hexdigest()
        cache = getattr(importer, "_cdxml_document_uuids", {})
        key = f"{profile_uuid}:{checksum}"
        identifier = cache.get(key)
        if identifier is not None:
            try:
                load_node(identifier)
            except NotExistent:
                identifier = None
        if identifier is None:
            node = SinglefileData(file=BytesIO(content), filename=Path(filename).name)
            node.store()
            identifier = str(node.uuid)
            del node
            cache[key] = identifier
            importer._cdxml_document_uuids = cache
        return identifier

    def store(event=None):
        from aiida.orm import Dict, load_node

        if manager.structure_node is None:
            return
        already_stored = manager.structure_node.is_stored
        atoms = manager.structure
        metadata = copy.deepcopy(atoms.info) if atoms is not None else {}
        manager.store_structure(event)
        if already_stored or "cdxml_source" not in metadata:
            return
        node_uuid = str(manager.structure_node.uuid)
        # All caches/closures carry only plain source data and UUIDs. The fresh
        # ORM instance is loaded in this immediate storing operation only.
        try:
            source = metadata["cdxml_source"]
            documents = metadata.get("cdxml_document_refs")
            if documents is None:
                original = importer._cdxml_content
                selected = importer._selected_cdxml_content()
                selected = selected.encode() if isinstance(selected, str) else selected
                if hashlib.sha256(selected).hexdigest() != source["sha256"]:
                    raise ValueError(
                        "The model no longer matches the currently loaded CDXML source."
                    )
                reviewed = bytes(importer._preview_cdxml) or selected
                filename = getattr(importer, "_cdxml_filename", "drawing.cdxml")
                documents = {
                    "original_uuid": document(original, filename),
                    "reviewed_uuid": document(
                        reviewed, f"{Path(filename).stem}-reviewed.cdxml"
                    ),
                }
            generation = metadata.get("cdxml_generation", {})
            stored_hash = model_fingerprint(load_node(node_uuid).get_ase())
            record = {
                "schema": 1,
                "source": source,
                "generation": generation,
                "documents": documents,
                "stored_model_hash": stored_hash,
                "edited_after_generation": bool(
                    generation.get("model_hash")
                    and generation["model_hash"] != stored_hash
                ),
            }
            recorded = record_cdxml_import(
                original_document=load_node(documents["original_uuid"]),
                reviewed_document=load_node(documents["reviewed_uuid"]),
                parameters=Dict(dict=record),
                reviewed_model=load_node(node_uuid),
            )
            recorded.label = manager.structure_label.value
            recorded.description = manager.structure_description.value
            recorded.base.extras.set("cdxml_import", record)
            if hasattr(manager, "has_trait") and manager.has_trait("input_structure"):
                # Use the manager's public input interface; structure_node is
                # a derived read-only trait in widgets-base.
                manager.input_structure = recorded
            else:
                manager.structure_node = recorded
            manager.output.value = f"Stored in AiiDA [{recorded}]"
            del recorded
        except Exception as exc:  # noqa: BLE001 -- report post-store failures in the widget
            import html

            manager.output.value += (
                f" Source metadata could not be saved: {html.escape(str(exc))}."
            )

    def restore(change):
        # The incoming framework event can contain a node; extract its UUID
        # immediately and independently reload at this point of consumption.
        identifier = getattr(change.get("new"), "uuid", None)
        if identifier is not None and manager.structure is not None:
            restored = restore_cdxml_metadata(manager.structure, str(identifier))
            manager.structure.info.update(restored.info)

    manager.btn_store.on_click(store)
    manager.observe(restore, names="input_structure")
