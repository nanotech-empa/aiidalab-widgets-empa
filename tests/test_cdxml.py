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
    # assert np.all(widget.structure[0].position == [0, 0, 0])
