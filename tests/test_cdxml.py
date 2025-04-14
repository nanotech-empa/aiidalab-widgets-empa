from pathlib import Path

import ase
# import pytest

import aiidalab_widgets_empa as awe


# @pytest.mark.usefixtures("aiida_profile_clean")
def test_structure_upload_widget():
    """Test the `StructureUploadWidget`."""
    widget = awe.CdxmlUploadWidget()
    assert widget.structure is None

    filename = Path(__file__).parent / "7AGNR.cdxml"

    with open(filename, "rb") as f:
        content = f.read()

    # Simulate the structure upload.
    widget._on_file_upload(
        change={
            "new": {
                "7AGNR.cdxml": {
                    "content": content,
                }
            }
        }
    )
    widget.create_button.click()
    assert isinstance(widget.structure, ase.Atoms)
    assert widget.structure.get_chemical_formula() == "C14H4"
    # Simulate the structure upload.
    widget._on_file_upload(
        change={
            "new": {
                "7AGNR.cdxml": {
                    "content": content,
                }
            }
        }
    )    
    # sets 2 units, finite size
    widget.nunits.value = '2'
    widget.create_button.click()
    assert isinstance(widget.structure, ase.Atoms)
    assert widget.structure.get_chemical_formula() == "C56H22"
    
    # case of benzene molecule
    filename = Path(__file__).parent / "benzene.cdxml"

    with open(filename, "rb") as f:
        content = f.read()
    # Simulate the structure upload.
    widget._on_file_upload(
        change={
            "new": {
                "benzene.cdxml": {
                    "content": content,
                }
            }
        }
    )    
    widget.create_button.click()
    assert isinstance(widget.structure, ase.Atoms)
    assert widget.structure.get_chemical_formula() == "C6H6"