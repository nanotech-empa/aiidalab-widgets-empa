"""Store source records and immutable documents in a temporary AiiDA profile."""

from pathlib import Path

import ipywidgets as ipw
import pytest
from aiida.orm import StructureData, load_node

from aiidalab_widgets_empa import (
    CdxmlUploadWidget,
    enable_cdxml_metadata_storage,
    restore_cdxml_metadata,
)

pytest_plugins = ("aiida.tools.pytest_fixtures",)


class Manager:
    """A UUID-only stand-in for the standard manager's public store interface."""

    def __init__(self, atoms):
        self.structure = atoms
        self._uuid = None
        self.btn_store = ipw.Button()
        self.btn_store.on_click(self.store_structure)
        self.output = ipw.HTML()
        self.structure_label = ipw.Text(value="reviewed molecule")
        self.structure_description = ipw.Text(value="CDXML test")
        self.observers = []

    @property
    def structure_node(self):
        return (
            load_node(self._uuid) if self._uuid else StructureData(ase=self.structure)
        )

    @structure_node.setter
    def structure_node(self, node):
        self._uuid = str(node.uuid)

    def store_structure(self, _=None):
        if self._uuid:
            return
        node = StructureData(ase=self.structure).store()
        self._uuid = str(node.uuid)
        self.output.value = f"Stored in AiiDA [{node}]"
        del node

    def observe(self, callback, names):
        self.observers.append(callback)


def importer():
    widget = CdxmlUploadWidget()
    content = (Path(__file__).parent / "benzene.cdxml").read_bytes()
    widget._on_file_upload({"new": ({"name": "benzene.cdxml", "content": content},)})
    widget.create_button.click()
    return widget


def test_store_roundtrip_preserves_metadata_documents_and_pk_feedback(aiida_profile):
    widget = importer()
    manager = Manager(widget.structure.copy())
    enable_cdxml_metadata_storage(manager, widget)
    manager.btn_store.click()
    identifier = manager._uuid
    record = load_node(identifier).base.extras.get("cdxml_import")
    assert str(load_node(identifier).pk) in manager.output.value
    assert load_node(identifier).label == "reviewed molecule"
    assert record["source"]["formal_charge"] == 0
    assert record["generation"]["implicit_hydrogens_included"]
    assert not record["edited_after_generation"]
    assert load_node(identifier).creator.process_label == "record_cdxml_import"
    assert (
        load_node(identifier).creator.inputs.original_document.uuid
        == record["documents"]["original_uuid"]
    )
    assert (
        load_node(identifier).creator.inputs.reviewed_document.uuid
        == record["documents"]["reviewed_uuid"]
    )
    assert (
        load_node(record["documents"]["original_uuid"]).get_content().encode()
        == widget._cdxml_content
    )
    reloaded = restore_cdxml_metadata(load_node(identifier).get_ase(), identifier)
    assert reloaded.info["cdxml_source"] == widget.structure.info["cdxml_source"]
    assert (
        reloaded.info["cdxml_generation"] == widget.structure.info["cdxml_generation"]
    )


def test_manual_edits_are_recorded_without_claiming_unchanged_generation(aiida_profile):
    widget = importer()
    edited = widget.structure.copy()
    edited.positions[0, 2] += 0.1
    manager = Manager(edited)
    enable_cdxml_metadata_storage(manager, widget)
    manager.btn_store.click()
    record = load_node(manager._uuid).base.extras.get("cdxml_import")
    assert record["edited_after_generation"]
    assert load_node(manager._uuid).get_ase().positions[0, 2] == pytest.approx(
        edited.positions[0, 2]
    )


def test_export_includes_immutable_source_documents(aiida_profile, tmp_path):
    from aiida.orm import Dict, SinglefileData, QueryBuilder
    from aiida.tools.archive import create_archive
    from aiida.tools.archive.implementations.sqlite_zip.main import ArchiveFormatSqlZip

    widget = importer()
    manager = Manager(widget.structure.copy())
    enable_cdxml_metadata_storage(manager, widget)
    manager.btn_store.click()
    identifier = manager._uuid
    record = load_node(identifier).creator.inputs.parameters.get_dict()
    # A mutable extra cannot override the immutable import record.
    load_node(identifier).base.extras.set("cdxml_import", {"schema": 0})
    assert (
        restore_cdxml_metadata(load_node(identifier).get_ase(), identifier).info[
            "cdxml_source"
        ]
        == record["source"]
    )
    archive = tmp_path / "reviewed.aiida"
    create_archive([load_node(identifier)], filename=archive)
    with ArchiveFormatSqlZip().open(archive) as reader:
        backend = reader.get_backend()
        files = (
            QueryBuilder(backend=backend)
            .append(SinglefileData, project="uuid")
            .all(flat=True)
        )
        assert set(record["documents"].values()).issubset(set(map(str, files)))
        records = (
            QueryBuilder(backend=backend)
            .append(Dict, project="attributes")
            .all(flat=True)
        )
        assert record in records


def test_isotope_charge_and_radical_roundtrip(aiida_profile):
    widget = CdxmlUploadWidget()
    content = b'<CDXML><page><fragment><n id="c" p="0 0" Element="6" Isotope="13" Radical="Doublet"/><n id="o" p="30 0" Element="8" Charge="-1"/><b id="b" B="c" E="o"/></fragment></page></CDXML>'
    widget._on_file_upload({"new": ({"name": "radical.cdxml", "content": content},)})
    widget.create_button.click()
    manager = Manager(widget.structure.copy())
    enable_cdxml_metadata_storage(manager, widget)
    manager.btn_store.click()
    result = restore_cdxml_metadata(load_node(manager._uuid).get_ase(), manager._uuid)
    assert result[0].mass == pytest.approx(13.00335484)
    assert result.info["cdxml_source"]["formal_charge"] == -1
    assert result.info["cdxml_source"]["atoms"]["c"]["radical_electrons"] == 1
    assert "initial_charges" not in result.arrays


def test_an_already_stored_node_is_not_retrofitted(aiida_profile):
    widget = importer()
    manager = Manager(widget.structure.copy())
    manager.store_structure()
    identifier = manager._uuid
    enable_cdxml_metadata_storage(manager, widget)
    manager.btn_store.click()
    assert manager._uuid == identifier
    assert load_node(identifier).base.extras.get("cdxml_import", None) is None


def test_actual_structure_manager_store_path(aiida_profile):
    base = pytest.importorskip("aiidalab_widgets_base")
    widget = importer()
    manager = base.StructureManagerWidget(
        importers=[widget], node_class="StructureData", storable=True
    )
    manager.input_structure = widget.structure.copy()
    enable_cdxml_metadata_storage(manager, widget)
    manager.btn_store.click()
    identifier = str(manager.structure_node.uuid)
    assert "metadata could not" not in manager.output.value, manager.output.value
    assert manager.structure.info.get("cdxml_source") is not None, (
        manager.structure.info
    )
    assert (
        load_node(identifier).base.extras.get("cdxml_import")["source"]["formal_charge"]
        == 0
    )
    assert str(load_node(identifier).pk) in manager.output.value
    manager.structure = None
    manager.input_structure = None
    manager.close()
