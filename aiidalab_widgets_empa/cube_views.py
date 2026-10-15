"""Named cube views; applications own persistence and CalcJob submission."""

import html

import ipywidgets as ipw


class SavedCubeViews(ipw.VBox):
    """Manage serializable view records through application callbacks.

    ``save(name, identifier)`` snapshots the current viewer and returns its ID.
    ``list_views()`` returns records with ``id``, ``name`` and ``label`` keys.
    ``load(identifier)``, ``remove(identifier)`` and ``render(identifiers)``
    resolve records in the application. No scientific data or ORM nodes live here.
    """

    def __init__(self, *, save, list_views, load, remove, render):
        self._save, self._list, self._load = save, list_views, load
        self._remove, self._render = remove, render
        self._records = {}
        self.name = ipw.Text(
            description="View name",
            placeholder="e.g. Side view",
            layout=ipw.Layout(width="350px"),
        )
        self.save_button = ipw.Button(
            description="Save current view",
            icon="save",
            layout=ipw.Layout(width="auto"),
        )
        self.saved = ipw.SelectMultiple(
            description="Saved views", rows=4, layout=ipw.Layout(width="95%")
        )
        self.load_button = ipw.Button(description="Restore view", icon="eye")
        self.update_button = ipw.Button(description="Update selected", icon="save")
        self.remove_button = ipw.Button(description="Remove selected", icon="trash")
        self.render_button = ipw.Button(
            description="Render selected views",
            icon="image",
            button_style="primary",
            layout=ipw.Layout(width="auto"),
        )
        self.refresh_button = ipw.Button(description="Refresh", icon="refresh")
        self.message = ipw.HTML()
        self.save_button.on_click(lambda _: self._act(self._save_current))
        self.update_button.on_click(
            lambda _: self._act(lambda: self._save_current(update=True))
        )
        self.load_button.on_click(
            lambda _: self._act(lambda: self._load(self._single()))
        )
        self.remove_button.on_click(lambda _: self._act(self._remove_selected))
        self.render_button.on_click(lambda _: self._act(self._render_selected))
        self.refresh_button.on_click(lambda _: self._act(self.refresh))
        self.saved.observe(self._selection_changed, names="value")
        super().__init__(
            [
                ipw.HTML(
                    "<h4>Saved visualizations</h4><p>Save camera, isosurfaces, rendering options and slice settings for later. Select several views with Ctrl/Cmd to render them together.</p>"
                ),
                ipw.HBox([self.name, self.save_button]),
                self.saved,
                ipw.HBox([self.load_button, self.update_button, self.remove_button]),
                ipw.HBox([self.render_button, self.refresh_button]),
                self.message,
            ]
        )
        self.refresh()

    def _act(self, action):
        try:
            self.message.value = ""
            action()
        except Exception as exc:
            self.message.value = (
                '<p style="color:#a02020">' + html.escape(str(exc)) + "</p>"
            )

    def refresh(self, selected=None):
        selected = self.saved.value if selected is None else selected
        records = self._list()
        self._records = {record["id"]: record for record in records}
        self.saved.options = [(record["label"], record["id"]) for record in records]
        self.saved.value = tuple(key for key in selected if key in self._records)
        self._selection_changed()

    def _selection_changed(self, _=None):
        count = len(self.saved.value)
        self.load_button.disabled = self.update_button.disabled = count != 1
        self.remove_button.disabled = self.render_button.disabled = count == 0
        if count == 1:
            self.name.value = self._records[self.saved.value[0]]["name"]

    def _single(self):
        if len(self.saved.value) != 1:
            raise ValueError("Select one visualization to restore or update")
        return self.saved.value[0]

    def _save_current(self, *, update=False):
        name = self.name.value.strip()
        if not name:
            raise ValueError("Give this visualization a name")
        identifier = self._save(name, self._single() if update else None)
        self.refresh(selected=[identifier])
        self.message.value = (
            "<p>Visualization saved. No rendering job was submitted.</p>"
        )

    def _remove_selected(self):
        for identifier in self.saved.value:
            self._remove(identifier)
        self.refresh()
        self.message.value = (
            "<p>Saved views removed; existing rendered images are kept.</p>"
        )

    def _render_selected(self):
        if not self.saved.value:
            raise ValueError("Select at least one saved visualization")
        self._render(list(self.saved.value))
