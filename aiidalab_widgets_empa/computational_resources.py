import ipywidgets as ipw
import traitlets as tr
from aiida import orm
import timelength

STYLE = {
    "description_width": "120px",
}
LAYOUT = {"width": "200px"}


def closest_multiple(nodes, tasks, groups):
    """Returns the integer M closest to N such that M*T is a multiple of G."""
    if nodes % groups == 0 or tasks % groups == 0:
        return nodes
    else:
        int_division = nodes // groups
        bigger = (int_division + 1) * groups
        smaller = (int_division - 1) * groups
        if bigger - nodes < nodes - smaller:
            return bigger
        else:
            return smaller


class ProcessResourcesWidget(ipw.VBox):
    """Setup metadata for an AiiDA process."""

    nproc_replica_trait = tr.Int()
    n_replica_trait = tr.Int()
    n_replica_per_group_trait = tr.Int()
    walltime_seconds = tr.Int(allow_none=True)
    neb = tr.Bool()
    phonons = tr.Bool()

    def __init__(self):
        """Metadata widget to generate metadata"""

        self.walltime_widget = ipw.Text(
            description="Walltime:",
            placeholder="10:30:00",
            style=STYLE,
            layout=LAYOUT,
        )

        self.wrong_syntax = ipw.HTML(
            value="""<i class="fa fa-times" style="color:red;font-size:2em;" ></i> wrong syntax""",
            layout={"visibility": "hidden"},
        )
        self.time_info = ipw.HTML(layout=LAYOUT)
        self.walltime_widget.observe(self.parse_time_string, "value")

        self.nodes_widget = ipw.IntText(
            value=2, description="# Nodes", style=STYLE, layout=LAYOUT
        )
        self.tasks_per_node_widget = ipw.IntText(
            value=32, description="# Tasks per node", style=STYLE, layout=LAYOUT
        )
        self.threads_per_task_widget = ipw.IntText(
            value=4, description="# Threads per task", style=STYLE, layout=LAYOUT
        )

        self.nodes_widget.observe(self.on_cores_change, "value")
        self.tasks_per_node_widget.observe(self.on_cores_change, "value")

        children = [
            self.nodes_widget,
            self.tasks_per_node_widget,
            self.threads_per_task_widget,
            ipw.HBox([self.walltime_widget, self.wrong_syntax]),
            self.time_info,
        ]

        super().__init__(children=children)

        self.walltime_widget.value = "24:00:00"  # To trigger the traitlet obeserver.

    @property
    def nodes(self):
        return int(self.nodes_widget.value)

    @property
    def tasks_per_node(self):
        return int(self.tasks_per_node_widget.value)

    @property
    def threads_per_task(self):
        return int(self.threads_per_task_widget.value)

    @tr.observe("n_replica_trait")
    def on_n_replica_trait_change(self, change):
        if change["old"] != 0:
            if self.neb or self.phonons:
                self.nodes_widget.value = int(
                    self.nodes_widget.value * change["new"] / change["old"]
                )

    @tr.observe("n_replica_per_group_trait")
    def on_n_replica_per_group_trait_change(self, change):
        if change["old"] != 0 and self.neb:
            self.nodes_widget.value = int(
                self.nodes_widget.value * change["old"] / change["new"]
            )

    def on_cores_change(self, _=None):
        if self.neb:
            ngroups = int(self.n_replica_trait / self.n_replica_per_group_trait)
            self.nodes_widget.value = closest_multiple(
                self.nodes_widget.value, self.tasks_per_node_widget.value, ngroups
            )
            self.nproc_replica_trait = int(
                self.nodes_widget.value
                * self.tasks_per_node_widget.value
                * self.n_replica_per_group_trait
                / self.n_replica_trait
            )
        elif self.phonons:
            self.nodes_widget.value = closest_multiple(
                self.nodes_widget.value,
                self.tasks_per_node_widget.value,
                self.n_replica_trait,
            )
            self.nproc_replica_trait = int(
                self.nodes_widget.value
                * self.tasks_per_node_widget.value
                / self.n_replica_trait
            )

    def parse_time_string(self, _=None):
        """Parse the time string and set the time in seconds"""
        self.wrong_syntax.layout.visibility = "hidden"
        self.time_info.value = ""

        t_length = timelength.TimeLength(self.walltime_widget.value)

        if t_length.result.success:
            self.walltime_seconds = int(t_length.result.seconds)
            self.time_info.value = f"Total walltime: {t_length.result.delta}"
        else:
            self.wrong_syntax.layout.visibility = "visible"
            self.time_info.value = ""
            self.walltime_seconds = None


class ResourcesEstimatorWidget(ipw.VBox):
    details = tr.Dict()
    uks = tr.Bool()
    nodes = tr.Int()
    walltime_hours = tr.Float(allow_none=True)
    selected_code = tr.Union([tr.Unicode(), tr.Instance(orm.Code)], allow_none=True)

    def __init__(
        self,
        calculation_type="dft",
        price_per_hour=1.0,
        currency="CHF",
        price_link=None,
    ):
        """Widget to estimate the resources needed for a calculation."""
        self.max_tasks_per_node = 1
        self.calculation_type = calculation_type
        self.estimate_resources_button = ipw.Button(
            description="Estimate resources", button_style="warning", style=STYLE
        )
        self.estimate_resources_button.on_click(self.estimate_resources)
        self.info_cost = ipw.HTML(
            value="",
            style=STYLE,
        )
        self.price_per_hour = price_per_hour
        self.currency = currency
        self.price_link = price_link

        super().__init__([self.estimate_resources_button, self.info_cost])

    def link_to_resources_widget(self, resources_widget):
        self.resources = resources_widget
        tr.dlink((self.resources.nodes_widget, "value"), (self, "nodes"))

        tr.dlink(
            (self.resources, "walltime_seconds"),
            (self, "walltime_hours"),
            transform=lambda x: x / 3600 if x is not None else None,
        )

    @tr.observe("nodes")
    def _observe_nodes(self, _=None):
        self.update_cost_info()

    @tr.observe("walltime_hours")
    def _observe_walltime_hours(self, _=None):
        self.update_cost_info()

    @tr.observe("details")
    def _observe_details(self, _=None):
        try:
            self.system_type = (
                "Slab"
                if "Slab" in self.details["system_type"]
                else self.details["system_type"]
            )
            self.element_list = self.details["all_elements"]
        except KeyError:
            self.system_type = "Other"
            self.element_list = []

    @tr.observe("selected_code")
    def _observe_code(self, _=None):
        try:
            self.max_tasks_per_node = orm.load_code(
                self.selected_code
            ).computer.get_default_mpiprocs_per_machine()
        except (ValueError, AttributeError):
            self.max_tasks_per_node = 1
        self.update_cost_info()

    def update_cost_info(self):
        """Update the cost information displayed in the widget."""
        self.info_cost.value = ""
        if not self.selected_code or self.walltime_hours is None:
            return

        try:
            code = orm.load_node(self.selected_code).full_label
        except Exception as e:
            self.info_cost.value = f"<b style='color:red;'>Error loading code: {e}</b>"
            return

        if "daint" in code:
            price_info = (
                f"""<i class='fa fa-info-circle' style='color:blue;font-size:2em;' ></i> """
                f"""Total Estimated Cost: {self.resources.nodes_widget.value * self.price_per_hour * self.walltime_hours:.2f} {self.currency}.<br>"""
            )

            if self.price_link:
                price_info += f"""The price was computed according to the following <a href='{self.price_link}' target='_blank'>link</a>."""

            self.info_cost.value = price_info

        elif "localhost" in code:
            if self.cost() > 50:
                self.info_cost.value = (
                    """<i class='fa fa-info-circle' style='color:orange;font-size:2em;' ></i> """
                    """The system may be too big for the localhost."""
                )

    def cost(self):
        cost_per_element = {
            "H": 1,
            "C": 4,
            "Si": 4,
            "N": 5,
            "O": 6,
            "Au": 11,
            "Cu": 11,
            "Ag": 11,
            "Pt": 18,
            "Tb": 19,
            "Co": 11,
            "Zn": 10,
            "Pd": 18,
            "Ga": 10,
        }
        the_cost = 0
        if self.element_list is not None:
            for element in self.element_list:
                s = "".join(i for i in element if not i.isdigit())
                if isinstance(s[-1], int):
                    s = s[:-1]
                if s in cost_per_element.keys():
                    the_cost += cost_per_element[s]
                else:
                    the_cost += 4
            if self.system_type == "Slab" or self.system_type == "Bulk":
                the_cost = int(the_cost / 11)
            else:
                the_cost = int(the_cost / 4)
            if self.uks:
                the_cost = the_cost * 1.26
        return the_cost

    def estimate_resources(self, _=None):
        """Determine the resources needed for the calculation."""

        if not self.selected_code:
            self.info_cost.value = (
                """<i class='fa fa-info-circle' style='color:red;font-size:2em;' ></i> """
                """<b style='color:red;'>Please select a code</b>"""
            )
            return

        if self.calculation_type == "dft":
            resources = self._estimate_resources_dft()
        elif self.calculation_type == "gw":
            resources = self._estimate_resources_gw()
        elif self.calculation_type == "gw_ic":
            resources = self._estimate_resources_gw_ic()

        theone = min(resources, key=lambda x: abs(x - self.cost()))

        if self.resources.n_replica_trait:
            self.resources.nodes_widget.value = (
                resources[theone]["nodes"] * self.resources.n_replica_trait
            )
        else:
            self.resources.nodes_widget.value = resources[theone]["nodes"]

        self.resources.tasks_per_node_widget.value = resources[theone]["tasks_per_node"]
        self.resources.threads_per_task_widget.value = resources[theone]["threads"]
        code = orm.load_node(self.selected_code).full_label
        if "localhost" in code:
            self.resources.nodes_widget.value = 1
            self.resources.tasks_per_node_widget.value = 1
            self.resources.threads_per_task_widget.value = 1

        self.update_cost_info()

    def _estimate_resources_dft(self):
        """Determine the resources needed for the DFT calculation."""

        resources = {
            "Slab": {
                50: {
                    "nodes": 1,
                    "tasks_per_node": 36,
                    "threads": 4,
                },
                200: {
                    "nodes": 2,
                    "tasks_per_node": 32,
                    "threads": 4,
                },
                1400: {
                    "nodes": 4,
                    "tasks_per_node": 36,
                    "threads": 4,
                },
                3000: {
                    "nodes": 6,
                    "tasks_per_node": 24,
                    "threads": 4,
                },
                4000: {
                    "nodes": 8,
                    "tasks_per_node": 32,
                    "threads": 2,
                },
                10000: {
                    "nodes": 12,
                    "tasks_per_node": 12,
                    "threads": 8,
                },
            },
            "Wire": {
                50: {
                    "nodes": 1,
                    "tasks_per_node": 36,
                    "threads": 4,
                },
                200: {
                    "nodes": 2,
                    "tasks_per_node": 32,
                    "threads": 4,
                },
                1400: {
                    "nodes": 4,
                    "tasks_per_node": 36,
                    "threads": 4,
                },
                3000: {
                    "nodes": 6,
                    "tasks_per_node": 24,
                    "threads": 4,
                },
                4000: {
                    "nodes": 8,
                    "tasks_per_node": 32,
                    "threads": 2,
                },
                10000: {
                    "nodes": 12,
                    "tasks_per_node": 12,
                    "threads": 8,
                },
            },
            "Bulk": {
                50: {
                    "nodes": 1,
                    "tasks_per_node": 36,
                    "threads": 4,
                },
                200: {
                    "nodes": 2,
                    "tasks_per_node": 32,
                    "threads": 4,
                },
                1400: {
                    "nodes": 4,
                    "tasks_per_node": 36,
                    "threads": 4,
                },
                3000: {
                    "nodes": 6,
                    "tasks_per_node": 24,
                    "threads": 4,
                },
                4000: {
                    "nodes": 8,
                    "tasks_per_node": 32,
                    "threads": 2,
                },
                10000: {
                    "nodes": 12,
                    "tasks_per_node": 12,
                    "threads": 8,
                },
            },
            "Molecule": {
                50: {
                    "nodes": 1,
                    "tasks_per_node": 36,
                    "threads": 4,
                },
                200: {
                    "nodes": 2,
                    "tasks_per_node": 32,
                    "threads": 4,
                },
                1400: {
                    "nodes": 4,
                    "tasks_per_node": 36,
                    "threads": 4,
                },
                3000: {
                    "nodes": 6,
                    "tasks_per_node": 24,
                    "threads": 4,
                },
                4000: {
                    "nodes": 8,
                    "tasks_per_node": 32,
                    "threads": 2,
                },
                10000: {
                    "nodes": 12,
                    "tasks_per_node": 12,
                    "threads": 8,
                },
            },
            "Other": {
                50: {
                    "nodes": 1,
                    "tasks_per_node": 36,
                    "threads": 4,
                },
                200: {
                    "nodes": 2,
                    "tasks_per_node": 32,
                    "threads": 4,
                },
                1400: {
                    "nodes": 4,
                    "tasks_per_node": 36,
                    "threads": 4,
                },
                3000: {
                    "nodes": 6,
                    "tasks_per_node": 24,
                    "threads": 4,
                },
                4000: {
                    "nodes": 8,
                    "tasks_per_node": 32,
                    "threads": 2,
                },
                10000: {
                    "nodes": 12,
                    "tasks_per_node": 12,
                    "threads": 8,
                },
            },
        }

        return resources[self.system_type]

    def _estimate_resources_gw(self):
        resources = {
            10: {
                "nodes": 2,
                "tasks_per_node": max(self.max_tasks_per_node / 4, 1),
                "threads": 1,
            },
            20: {
                "nodes": 6,
                "tasks_per_node": max(self.max_tasks_per_node / 4, 1),
                "threads": 1,
            },
            50: {
                "nodes": 12,
                "tasks_per_node": self.max_tasks_per_node,
                "threads": 1,
            },
            100: {
                "nodes": 256,
                "tasks_per_node": int(max(self.max_tasks_per_node / 3, 1)),
                "threads": 1,
            },
            180: {
                "nodes": 512,
                "tasks_per_node": int(max(self.max_tasks_per_node / 3, 1)),
                "threads": 1,
            },
            400: {
                "nodes": 1024,
                "tasks_per_node": int(max(self.max_tasks_per_node / 3, 1)),
                "threads": 1,
            },
        }
        return resources

    def _estimate_resources_gw_ic(self):
        resources = {
            10: {
                "nodes": 2,
                "tasks_per_node": max(self.max_tasks_per_node / 4, 1),
                "threads": 1,
            },
            20: {
                "nodes": 6,
                "tasks_per_node": max(self.max_tasks_per_node / 4, 1),
                "threads": 1,
            },
            50: {
                "nodes": 12,
                "tasks_per_node": self.max_tasks_per_node,
                "threads": 1,
            },
            100: {
                "nodes": 256,
                "tasks_per_node": int(max(self.max_tasks_per_node / 3, 1)),
                "threads": 1,
            },
            180: {
                "nodes": 512,
                "tasks_per_node": int(max(self.max_tasks_per_node / 3, 1)),
                "threads": 1,
            },
            400: {
                "nodes": 1024,
                "tasks_per_node": int(max(self.max_tasks_per_node / 3, 1)),
                "threads": 1,
            },
        }
        return resources
