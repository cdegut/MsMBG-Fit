from math import log
from dearpygui import dearpygui as dpg
from modules.data_structures import MSData, get_global_msdata_ref
from modules.finding import peaks_clear_callback, peaks_finder_callback, add_peak
from modules.rendercallback import RenderCallback, get_global_render_callback_ref
import numpy as np
from modules.finding_callback import get_smoothing_window, set_smoothing_window
from modules.svg_export import add_export_button


def finding_window(render_callback):
    spectrum: MSData = render_callback.spectrum
    with dpg.child_window(
        label="Data Filtering and peak finding", tag="Data Filtering"
    ):
        with dpg.group(horizontal=True, horizontal_spacing=40):
            dpg.add_text("Data Clipping:")
            with dpg.group(horizontal=True, horizontal_spacing=5):
                dpg.add_text("L:")
                dpg.add_slider_int(
                    width=400,
                    default_value=0,
                    min_value=0,
                    max_value=12000,
                    tag="L_data_clipping",
                    callback=data_clipper,
                    user_data=render_callback,
                )
                dpg.add_button(
                    label="+",
                    width=30,
                    callback=data_clipper_button,
                    user_data=render_callback,
                    tag="L_data_clipping_plus",
                )
                dpg.add_button(
                    label="-",
                    width=30,
                    callback=data_clipper_button,
                    user_data=render_callback,
                    tag="L_data_clipping_minus",
                )
            with dpg.group(horizontal=True, horizontal_spacing=5):
                dpg.add_text("R:")
                dpg.add_slider_int(
                    width=400,
                    default_value=12000,
                    min_value=0,
                    max_value=12000,
                    tag="R_data_clipping",
                    callback=data_clipper,
                    user_data=render_callback,
                )
                dpg.add_button(
                    label="+",
                    width=30,
                    callback=data_clipper_button,
                    user_data=render_callback,
                    tag="R_data_clipping_plus",
                )
                dpg.add_button(
                    label="-",
                    width=30,
                    callback=data_clipper_button,
                    user_data=render_callback,
                    tag="R_data_clipping_minus",
                )
        # Create a plot for the data
        with dpg.plot(
            label="Data Filtering", width=1430, height=600, tag="data_plot"
        ) as plot1:
            # Add x and y axes
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="m/z", tag="x_axis_plot1")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Y Axis", tag="y_axis_plot1")
            dpg.add_line_series(
                [],
                [],
                label="Original Data Series",
                parent=y_axis,
                tag="original_series",
            )
            dpg.add_line_series(
                [], [], label="Snip Baseline", parent=y_axis, tag="baseline"
            )
            dpg.add_line_series(
                [], [], label="2nd Order Derivative", parent=y_axis, tag="derivative2nd"
            )
        add_export_button("data_plot")

        with dpg.group(horizontal=True, horizontal_spacing=50):
            with dpg.child_window(height=230, width=300):
                dpg.add_checkbox(
                    default_value=False,
                    label="Show Smoothed Data",
                    callback=show_smoothed_data,
                    user_data=render_callback,
                    tag="show_smoothed_data_checkbox",
                )
                dpg.add_checkbox(
                    default_value=False,
                    label="Show 2nd Order Derivative",
                    callback=sec_order_derivative,
                    user_data=render_callback,
                    tag="show_derivative2nd_checkbox",
                )
                dpg.add_text("Data Filtering:")
                with dpg.group(horizontal=True, horizontal_spacing=50):
                    dpg.add_text("Lambda smoother (10^x):")
                    dpg.add_button(
                        label="Auto Window",
                        callback=auto_window_callback,
                        user_data=render_callback,
                    )
                dpg.add_slider_float(
                    label="",
                    default_value=2.0,
                    min_value=1,
                    max_value=10,
                    width=250,
                    callback=filter_data,
                    tag="smoothing_window",
                )
                dpg.add_text("")
                dpg.add_text("Baseline estimation:")
                with dpg.group(horizontal=True, horizontal_spacing=50):
                    dpg.add_checkbox(
                        label="Correct baseline",
                        callback=toggle_baseline,
                        user_data=spectrum,
                        default_value=False,
                        tag="baseline_correction_checkbox",
                    )
                dpg.add_text("Baseline window:")
                dpg.add_slider_int(
                    label="",
                    default_value=1000,
                    min_value=10,
                    max_value=2000,
                    width=250,
                    callback=spectrum.request_baseline_update,
                    user_data=spectrum,
                    tag="baseline_window",
                )
                dpg.add_button(
                    label="Show Fit Residual",
                    tag="fitting_residual_plot1_button",
                    callback=get_residual,
                    user_data=render_callback,
                )
            with dpg.child_window(height=230, width=300):
                dpg.add_text("Peak detection:")
                dpg.add_text("Peak detection threshold:")
                dpg.add_slider_int(
                    label="",
                    width=200,
                    default_value=100,
                    min_value=0,
                    max_value=1000,
                    tag="peak_detection_threshold",
                    callback=move_threshold_callback,
                    user_data=spectrum,
                )
                with dpg.group(horizontal=True, horizontal_spacing=50):
                    dpg.add_checkbox(
                        label="x100",
                        default_value=False,
                        tag="threshold_x100",
                        callback=move_threshold_callback,
                    )
                    dpg.add_button(
                        label="Show",
                        callback=show_threshold_callback,
                        tag="show_threshold_button",
                        user_data=spectrum,
                    )
                dpg.add_button(
                    label="Estimate Optimal Width",
                    callback=estimate_optimal_width_callback,
                )
                dpg.add_text("Peak detection width:")
                dpg.add_slider_int(
                    label="",
                    width=200,
                    default_value=10,
                    min_value=0,
                    max_value=50,
                    tag="peak_detection_width",
                )
                dpg.add_text("Min Peak Distance:")
                dpg.add_slider_int(
                    label="",
                    width=200,
                    default_value=10,
                    min_value=1,
                    max_value=50,
                    tag="peak_detection_distance",
                )
                dpg.add_checkbox(
                    label="Use 2nd order derivative",
                    default_value=True,
                    tag="use_2nd_derivative_checkbox",
                )
                dpg.add_button(
                    label="Find Peaks",
                    callback=peaks_finder_callback,
                    user_data=render_callback,
                )
                dpg.add_button(
                    label="Clear Peaks",
                    callback=peaks_clear_callback,
                    user_data=render_callback,
                )

            with dpg.child_window(height=230, width=300):
                dpg.add_text("Peaks:")
                dpg.add_table(
                    header_row=True,
                    tag="found_peak_table",
                    resizable=False,
                    policy=dpg.mvTable_SizingStretchProp,
                )

            with dpg.child_window(height=230, width=300):
                dpg.add_text("User Peaks:")
                dpg.add_button(
                    label="Add Peak", callback=add_peak, user_data=render_callback
                )
                dpg.add_table(
                    header_row=True,
                    tag="user_peak_table",
                    policy=dpg.mvTable_SizingStretchProp,
                )

    dpg.bind_item_theme("original_series", "data_theme")
    dpg.bind_item_theme("baseline", "baseline_theme")
    dpg.bind_item_theme("derivative2nd", "derivative2nd_theme")


def data_clipper():
    spectrum = get_global_msdata_ref()
    L_clip = dpg.get_value("L_data_clipping")
    R_clip = dpg.get_value("R_data_clipping")
    if L_clip >= R_clip:
        return
    if len(spectrum.original_data) <= 2:
        return

    spectrum.clip_data(L_clip, R_clip)
    spectrum.request_baseline_update()

    dpg.set_value(
        "original_series",
        [spectrum.working_data[:, 0].tolist(), spectrum.working_data[:, 1].tolist()],
    )
    dpg.set_value(
        "baseline", [spectrum.baseline[:, 0].tolist(), spectrum.baseline[:, 1].tolist()]
    )
    dpg.set_value(
        "corrected_series_plot2",
        [
            spectrum.baseline_corrected[:, 0].tolist(),
            spectrum.baseline_corrected[:, 1].tolist(),
        ],
    )
    dpg.set_value(
        "corrected_series_plot3",
        [
            spectrum.baseline_corrected[:, 0].tolist(),
            spectrum.baseline_corrected[:, 1].tolist(),
        ],
    )
    filter_data()
    dpg.fit_axis_data("y_axis_plot1")
    dpg.fit_axis_data("x_axis_plot1")
    dpg.fit_axis_data("y_axis_plot2")
    dpg.fit_axis_data("x_axis_plot2")
    dpg.fit_axis_data("y_axis_plot3")
    dpg.fit_axis_data("x_axis_plot3")


def filter_data():
    if not dpg.get_value("show_smoothed_data_checkbox"):
        return
    spectrum = get_global_msdata_ref()
    window_length = get_smoothing_window()
    filtered = spectrum.get_filtered_data(window_length)
    dpg.set_value("filtered_series", [spectrum.working_data[:, 0].tolist(), filtered])
    if dpg.get_value("show_derivative2nd_checkbox"):
        get_global_render_callback_ref().display_2nd_derivative()


def toggle_baseline():
    spectrum = get_global_msdata_ref()
    spectrum.baseline_toggle = not spectrum.baseline_toggle
    spectrum.request_baseline_update()


def get_residual():
    spectrum = get_global_msdata_ref()
    if dpg.does_item_exist("fitting_residual_plot1"):
        dpg.delete_item("fitting_residual_plot1")
        dpg.set_item_label("fitting_residual_plot1_button", "Show Fitting Residual")
        return
    else:
        value = dpg.get_value("residual")
        new_value = np.array(value[1]) + spectrum.baseline[:, 1]
        dpg.add_line_series(
            value[0],
            new_value.tolist(),
            label="Fitting Residual",
            parent="y_axis_plot1",
            tag="fitting_residual_plot1",
        )
        dpg.bind_item_theme("fitting_residual_plot1", "residual_theme_plot1")
        dpg.set_item_label("fitting_residual_plot1_button", "Hide Fitting Residual")


def sec_order_derivative():
    if dpg.get_value("show_derivative2nd_checkbox"):
        dpg.show_item("derivative2nd")
        get_global_render_callback_ref().display_2nd_derivative()
    else:
        dpg.hide_item("derivative2nd")


def show_threshold_callback():
    threshold = dpg.get_value("peak_detection_threshold")
    if dpg.get_value("threshold_x100"):
        threshold = threshold * 100

    if dpg.does_item_exist("threshold_line_plot1"):
        dpg.delete_item("threshold_line_plot1")
        dpg.set_item_label("show_threshold_button", "Show")
        return

    dpg.add_line_series(
        [], [], label="Threshold", parent="y_axis_plot1", tag="threshold_line_plot1"
    )
    if not dpg.does_item_exist("threshold_line_theme"):
        with dpg.theme(tag="threshold_line_theme"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line, (255, 116, 108), category=dpg.mvThemeCat_Plots
                )
    if not dpg.get_value("baseline_correction_checkbox"):
        dpg.set_value(
            "threshold_line_plot1",
            [
                [
                    dpg.get_axis_limits("x_axis_plot1")[0],
                    dpg.get_axis_limits("x_axis_plot1")[1],
                ],
                [threshold, threshold],
            ],
        )
    else:
        spectrum = get_global_msdata_ref()
        dpg.set_value(
            "threshold_line_plot1",
            [
                spectrum.baseline[:, 0].tolist(),
                (spectrum.baseline[:, 1] + threshold).tolist(),
            ],
        )
    dpg.bind_item_theme("threshold_line_plot1", "threshold_line_theme")
    dpg.set_item_label("show_threshold_button", "Hide")


def move_threshold_callback():
    threshold = float(dpg.get_value("peak_detection_threshold"))
    if dpg.get_value("threshold_x100"):
        threshold = threshold * 100

    for alias in dpg.get_aliases():
        if alias.startswith("threshold_line_"):
            if dpg.get_value("baseline_correction_checkbox"):
                spectrum = get_global_msdata_ref()
                dpg.set_value(
                    "threshold_line_plot1",
                    [
                        spectrum.baseline[:, 0].tolist(),
                        (spectrum.baseline[:, 1] + threshold).tolist(),
                    ],
                )
            else:
                dpg.set_value(
                    "threshold_line_plot1",
                    [
                        [
                            dpg.get_axis_limits("x_axis_plot1")[0],
                            dpg.get_axis_limits("x_axis_plot1")[1],
                        ],
                        [threshold, threshold],
                    ],
                )
            return


def auto_window_callback():
    spectrum = get_global_msdata_ref()
    optimal_window = spectrum.get_smoothing_window()
    print(f"Optimal smoothing window: {optimal_window}")
    optimal_window = log(optimal_window, 10)
    set_smoothing_window(optimal_window)
    filter_data()
    get_global_render_callback_ref().display_2nd_derivative()


def data_clipper_button(sender: str):
    if sender == "L_data_clipping_plus":
        dpg.set_value("L_data_clipping", dpg.get_value("L_data_clipping") + 5)
    elif sender == "L_data_clipping_minus":
        dpg.set_value("L_data_clipping", dpg.get_value("L_data_clipping") - 5)
    elif sender == "R_data_clipping_plus":
        dpg.set_value("R_data_clipping", dpg.get_value("R_data_clipping") + 5)
    elif sender == "R_data_clipping_minus":
        dpg.set_value("R_data_clipping", dpg.get_value("R_data_clipping") - 5)
    data_clipper()


def estimate_optimal_width_callback(sender=None, app_data=None, user_data=None):
    spectrum: MSData = get_global_msdata_ref()
    avg_length, spike_count, spike_info = spectrum.get_average_spike_length(
        min_spike_width=1, use_baseline_corrected=True
    )
    if spike_count == 0:
        print("No spikes detected for width estimation.")
        return
    sampling_rate = spectrum.guess_sampling_rate()
    if sampling_rate is None:
        print("Unable to estimate sampling rate for width estimation.")
        return

    optimal_width = int(avg_length * sampling_rate * 0.5)
    optimal_width = optimal_width if optimal_width >= 2 else 2
    print(
        f"Estimated optimal peak detection width: {optimal_width} (based on {spike_count} spikes)"
    )
    dpg.set_value("peak_detection_width", optimal_width)
    dpg.set_value("peak_detection_distance", optimal_width)


def show_smoothed_data():
    if not dpg.get_value("show_smoothed_data_checkbox"):
        dpg.delete_item("filtered_series")
    else:
        window = get_smoothing_window()
        spectrum = get_global_msdata_ref()
        filtered_serie = spectrum.get_filtered_data(window)
        dpg.add_line_series(
            spectrum.working_data[:, 0].tolist(),
            filtered_serie,
            label="Filtered Data Series",
            parent="y_axis_plot1",
            tag="filtered_series",
        )
        dpg.bind_item_theme("filtered_series", "filtered_data_theme")
