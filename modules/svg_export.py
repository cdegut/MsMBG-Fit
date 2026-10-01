"""
Export a dearpygui plot as SVG: dearpygui cannot save vector graphics, so the plot
as currently displayed (axis limits, visible series with their theme colours,
drawn lines, drag lines, annotations) is redrawn with matplotlib.
"""

import dearpygui.dearpygui as dpg
import matplotlib
from matplotlib.figure import Figure  # no pyplot: no GUI backend, no global state
import numpy as np

from modules.utils import log

EXPORT_DIALOG = "svg_export_dialog"
INCLUDE_LABELS = "svg_export_include_labels"
# Pixels of the dearpygui plot per inch of the figure: same size and proportions
DPI = 100
DEFAULT_COLORS = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]


def _rgba(color) -> tuple:
    """Colour as 0-1 RGBA for matplotlib (dearpygui can hold values slightly above 1)."""
    values = [float(np.clip(c, 0, 1)) for c in color]
    return tuple(values + [1.0] * (4 - len(values)))


def _item_type(item) -> str:
    return dpg.get_item_type(item).split("::")[-1]


def _theme_values(item, item_type_constant) -> dict:
    """Line / fill colours (0-1 RGBA) and line weight from the theme bound to item."""
    theme = dpg.get_item_info(item).get("theme")
    values = {}
    if not theme:
        return values
    # Components for this series type override those for all items
    components = sorted(
        dpg.get_item_children(theme, 1) or [],
        key=lambda c: dpg.get_item_configuration(c).get("item_type", 0) != dpg.mvAll,
    )
    for component in components:
        if dpg.get_item_configuration(component).get("item_type", 0) not in (
            dpg.mvAll,
            item_type_constant,
        ):
            continue
        for entry in dpg.get_item_children(component, 1) or []:
            config = dpg.get_item_configuration(entry)
            if config.get("category") != dpg.mvThemeCat_Plots:
                continue
            value = dpg.get_value(entry)
            if _item_type(entry) == "mvThemeColor":
                if config["target"] == dpg.mvPlotCol_Line:
                    values["line"] = _rgba([v / 255 for v in value])
                elif config["target"] == dpg.mvPlotCol_Fill:
                    values["fill"] = _rgba([v / 255 for v in value])
            elif config["target"] == dpg.mvPlotStyleVar_LineWeight:
                values["weight"] = value[0]
    return values


def _visible(item) -> bool:
    return bool(dpg.get_item_configuration(item).get("show", True))


def _limits(axis, values: list[np.ndarray]):
    """Displayed limits of an axis, or the data range if the plot was never drawn."""
    low, high = dpg.get_axis_limits(axis)
    if high > low:
        return low, high
    data = np.concatenate([v for v in values if len(v)]) if values else np.array([])
    if not len(data):
        return None
    return float(np.min(data)), float(np.max(data))


def export_plot_svg(plot: str, path: str, labels: bool = True) -> None:
    """labels=False leaves out the annotations (peak and charge-state labels)."""
    axes = [a for a in dpg.get_item_children(plot, 1) or [] if _item_type(a) == "mvPlotAxis"]
    if len(axes) < 2:
        log(f"SVG export: {plot} has no axes")
        return
    x_axis, y_axis = axes[0], axes[1]
    width, height = dpg.get_item_rect_size(plot)
    if width <= 0 or height <= 0:
        config = dpg.get_item_configuration(plot)
        width, height = config.get("width", 1430), config.get("height", 600)
    fig = Figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    ax = fig.subplots()

    xs, ys = [], []
    color_index = 0
    for series in dpg.get_item_children(y_axis, 1) or []:
        if not _visible(series):
            continue
        kind = _item_type(series)
        value = dpg.get_value(series)
        if not value or not len(value[0]):
            continue
        x = np.asarray(value[0], dtype=float)
        label = dpg.get_item_configuration(series).get("label", "")
        if kind == "mvLineSeries":
            theme = _theme_values(series, dpg.mvLineSeries)
            color = theme.get("line", DEFAULT_COLORS[color_index % len(DEFAULT_COLORS)])
            color_index += "line" not in theme
            y = np.asarray(value[1], dtype=float)
            ax.plot(x, y, color=color, linewidth=theme.get("weight", 1.0), label=label, zorder=3)
        elif kind == "mvShadeSeries":
            theme = _theme_values(series, dpg.mvShadeSeries)
            color = theme.get("fill", theme.get("line", (0.5, 0.5, 0.5, 0.4)))
            y = np.asarray(value[1], dtype=float)
            y2 = np.asarray(value[2], dtype=float) if len(value) > 2 and len(value[2]) else 0 * y
            ax.fill_between(x, y2, y, color=color, linewidth=0, zorder=2)
        else:
            continue
        xs.append(x)
        ys.append(y)

    for item in dpg.get_item_children(plot, 2) or []:
        if _item_type(item) != "mvDrawLine" or not _visible(item):
            continue
        config = dpg.get_item_configuration(item)
        (x1, y1), (x2, y2) = config["p1"][:2], config["p2"][:2]
        color = _rgba(config["color"])
        if x1 == x2:
            # Inside a plot the thickness is in x-axis units (e.g. matching tolerance)
            half = config["thickness"] / 2
            ax.fill_between(
                [x1 - half, x1 + half], [y1, y1], [y2, y2],
                color=color, linewidth=0, zorder=1,
            )
        else:
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=1, zorder=4)

    for item in dpg.get_item_children(plot, 0) or []:
        if not _visible(item):
            continue
        kind = _item_type(item)
        config = dpg.get_item_configuration(item)
        if kind == "mvDragLine":
            ax.axvline(dpg.get_value(item), color=_rgba(config["color"]), linewidth=1, zorder=4)
        elif kind == "mvAnnotation" and labels:
            x, y = dpg.get_value(item)[:2]
            ox, oy = config.get("offset", (0, 0))
            color = _rgba(config["color"])
            text_color = "black" if sum(color[:3]) > 1.8 else "white"
            ax.annotate(
                config.get("label", ""),
                (x, y),
                xytext=(ox, -oy),  # dearpygui offsets are in pixels, y down
                textcoords="offset points",
                fontsize=7,
                color=text_color,
                ha="center",
                va="center",
                annotation_clip=True,
                bbox=dict(boxstyle="square,pad=0.2", fc=color, ec="none"),
                zorder=5,
            )

    x_limits, y_limits = _limits(x_axis, xs), _limits(y_axis, ys)
    if x_limits:
        ax.set_xlim(*x_limits)
    if y_limits:
        ax.set_ylim(*y_limits)
    ax.set_xlabel(dpg.get_item_configuration(x_axis).get("label", ""))
    ax.set_ylabel(dpg.get_item_configuration(y_axis).get("label", ""))
    ax.set_title(dpg.get_item_configuration(plot).get("label", ""))
    fig.tight_layout()
    fig.savefig(path, format="svg")
    log(f"Plot saved as SVG: {path}")


def _export_callback(sender, app_data):
    path = app_data["file_path_name"]
    if not path.lower().endswith(".svg"):
        path += ".svg"
    export_plot_svg(
        dpg.get_item_user_data(EXPORT_DIALOG), path, labels=dpg.get_value(INCLUDE_LABELS)
    )


def create_svg_dialog():
    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=_export_callback,
        tag=EXPORT_DIALOG,
        width=700,
        height=400,
        default_filename="plot",
    ):
        dpg.add_file_extension(".svg", color=(54, 92, 45, 255), custom_text="[svg]")
        dpg.add_checkbox(
            label="Include labels (peak and charge-state annotations)",
            default_value=True,
            tag=INCLUDE_LABELS,
        )


def add_export_button(plot: str):
    """'Export SVG' button for plot (the dialog must exist: create_svg_dialog)."""

    def show_dialog():
        dpg.set_item_user_data(EXPORT_DIALOG, plot)
        dpg.show_item(EXPORT_DIALOG)

    dpg.add_button(label="Export SVG", callback=show_dialog)
