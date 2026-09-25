import re
import dearpygui.dearpygui as dpg
from modules.rendercallback import RenderCallback
from modules.matching import (
    show_matched_peaks,
    draw_mz_lines,
    redraw_blocks,
    print_to_terminal,
    refine_matching,
    check_integral_ratio,
)
from modules.data_structures import get_global_msdata_ref
from modules.var import colors_list


def matching_window(render_callback: RenderCallback):
    with dpg.child_window(label="Peak matching", tag="Peak matching"):
        # Create a plot for the raw data
        with dpg.plot(
            label="Peak matching", width=1430, height=600, tag="peak_matching_plot"
        ) as plot2:
            # Add x and y axes
            dpg.add_plot_axis(dpg.mvXAxis, label="m/z", tag="x_axis_plot3")
            dpg.add_plot_axis(dpg.mvYAxis, label="Y Axis", tag="y_axis_plot3")
            dpg.add_line_series(
                [],
                [],
                label="Corrected Data Series",
                parent="y_axis_plot3",
                tag="corrected_series_plot3",
            )
        with dpg.group(horizontal=True, horizontal_spacing=25):
            with dpg.group(horizontal=False):
                dpg.add_button(
                    label="Show matched peaks",
                    callback=show_matched_peaks_callback,
                    user_data=render_callback,
                    tag="show_matched_peaks_button",
                )
                dpg.add_button(
                    label="Draw MBG",
                    callback=draw_mbg_callback,
                    user_data=render_callback,
                )
                dpg.add_button(
                    label="Hide lines",
                    callback=hide_blocks_line_callback,
                    user_data=render_callback,
                    tag="hide_lines_button",
                )
                dpg.add_input_int(
                    label="Peaks series",
                    default_value=5,
                    min_value=1,
                    max_value=10,
                    tag="nb_peak_series",
                    width=70,
                    callback=peak_series_callback,
                    user_data=render_callback,
                )
                dpg.add_text("Tolerance:")
                dpg.add_input_int(
                    label="m/z",
                    default_value=25,
                    min_value=1,
                    max_value=200,
                    tag="block_width",
                    width=100,
                    callback=tolerance_callback,
                    user_data=render_callback,
                )
                dpg.add_text("Refine within:")
                dpg.add_input_int(
                    label="Da",
                    default_value=200,
                    min_value=1,
                    max_value=1000,
                    tag="refinement_width",
                    width=100,
                )
                dpg.add_checkbox(
                    label="Hide High error",
                    default_value=True,
                    tag="hide_high_error",
                    callback=redraw_blocks,
                    user_data=render_callback,
                )
                dpg.add_button(
                    label="Print to term",
                    callback=print_to_terminal,
                    user_data=render_callback,
                )

            with dpg.child_window(
                height=240,
                width=1225,
                tag=f"theorical_peaks_windows_container",
                show=True,
            ):
                with dpg.group(horizontal=False, horizontal_spacing=25):
                    with dpg.group(horizontal=True, horizontal_spacing=25):
                        for i in range(5):
                            peak_matching_window(render_callback, i)
                    with dpg.group(horizontal=True, horizontal_spacing=25):
                        for i in range(5, 10):
                            peak_matching_window(render_callback, i)

    dpg.bind_item_theme("corrected_series_plot3", "data_theme")
    peak_series_callback(
        sender="nb_peak_series", app_data=None, user_data=render_callback
    )


def tolerance_callback(sender, app_data, user_data):
    redraw_blocks()


def peak_matching_window(render_callback: RenderCallback, i):
    with dpg.child_window(
        height=200, width=220, tag=f"theorical_peaks_window_{i}", show=True
    ):
        with dpg.theme(tag=f"theme_peak_window_{i}"):
            with dpg.theme_component():
                dpg.add_theme_color(
                    dpg.mvThemeCol_Border, colors_list[i], category=dpg.mvThemeCat_Core
                )

        dpg.add_text(f"Peak Set {i}")

        with dpg.group(horizontal=True):
            dpg.add_input_int(
                label="MW",
                default_value=549000,
                tag=f"molecular_weight_{i}",
                step=100,
                width=125,
                callback=draw_mz_lines,
                user_data=i,
            )
            dpg.add_button(label="Refine", callback=refine_matching, user_data=i)

        dpg.add_input_int(
            label="Charges",
            default_value=52,
            tag=f"charges_{i}",
            width=125,
            callback=draw_mz_lines,
            user_data=i,
        )
        dpg.add_input_int(
            label="# Peaks",
            default_value=5,
            tag=f"nb_peak_show_{i}",
            step=1,
            width=125,
            callback=draw_mz_lines,
            user_data=i,
        )

        with dpg.group(horizontal=True):
            dpg.add_text("Compare with set:")
            dpg.add_input_int(
                default_value=0,
                tag=f"compare_set_{i}",
                width=70,
                callback=draw_mz_lines,
                user_data=i,
            )
        dpg.add_text("", tag=f"MW_diff_{i}")
        dpg.add_checkbox(
            default_value=False,
            tag=f"compare_checkbox_{i}",
            callback=compare_integral_callback,
            label="Use for comparison",
        )
        dpg.add_text("GeoMean of integral ratios:")
        dpg.add_text("", tag=f"Integral_ratio_{i}")

        dpg.add_text(f"Series quality:")
        dpg.add_text("", tag=f"series_quality_score_{i}")
        dpg.add_text("", tag=f"series_quality_rmsd_{i}")
        dpg.add_text("", tag=f"series_quality_modes_{i}")
        dpg.add_text("", tag=f"series_quality_width_{i}")
        dpg.add_table(
            header_row=False, row_background=True, tag=f"theorical_peak_table_{i}"
        )
        dpg.bind_item_theme(f"theorical_peaks_window_{i}", f"theme_peak_window_{i}")


def peak_series_callback(sender, app_data, user_data):
    nb_peak_series = dpg.get_value(sender)
    for alias in dpg.get_aliases():
        if alias.startswith(f"theorical_peaks_window_"):
            index = int(alias.split("_")[-1])
            if index >= nb_peak_series:
                dpg.hide_item(alias)
            else:
                dpg.show_item(alias)
    redraw_blocks()


def draw_mbg_callback(sender: str):
    for alias in dpg.get_aliases():
        if alias.startswith("MBG_plot3"):
            dpg.delete_item("MBG_plot3")
            dpg.set_item_label(sender, "Draw MBG")
            return

    spectrum = get_global_msdata_ref()
    x_data = spectrum.working_data[:, 0]
    mbg = spectrum.calculate_mbg(x_data)
    dpg.add_line_series(
        x_data.tolist(),
        mbg.tolist(),
        label="MBG",
        parent="y_axis_plot3",
        tag="MBG_plot3",
        show=True,
    )
    dpg.bind_item_theme("MBG_plot3", "matching_MBG_theme")
    dpg.set_item_label(sender, "Hide MBG")


def show_matched_peaks_callback(sender, app_data, user_data: RenderCallback):
    show_matched_peaks(user_data)


def hide_blocks_line_callback(sender, app_data, user_data: RenderCallback):
    hide = False
    for alias in dpg.get_aliases():
        if (
            alias.startswith("fitted_peak_matching")
            or alias.startswith(f"mz_lines_")
            or alias.startswith("peak_annotation_matching_not")
        ):
            hide = True
            dpg.delete_item(alias)
    if hide:
        dpg.set_item_label("hide_lines_button", "Show lines")
    else:
        redraw_blocks()
        dpg.set_item_label("hide_lines_button", "Hide lines")


def compare_integral_callback(sender, app_data, user_data):
    check_integral_ratio()
    return
