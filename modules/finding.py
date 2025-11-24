import time
import dearpygui.dearpygui as dpg
from modules import rendercallback
from modules.data_structures import MSData, get_global_msdata_ref, peak_params
from typing import Tuple
from scipy.signal import find_peaks
import numpy as np
from modules.utils import log
from modules.rendercallback import RenderCallback, get_global_render_callback_ref
from modules.finding_callback import get_smoothing_window


peak_width_color = (99, 143, 169, 64)


def add_peak():
    spectrum = get_global_msdata_ref()

    width_l = [spectrum.peaks[peak].width for peak in spectrum.peaks]
    std_width = np.std(width_l)
    med_width = np.median(width_l) if len(width_l) > 0 else 0

    width = int(med_width if med_width != 0 else dpg.get_value("peak_detection_width"))

    existing_peak_indexes: list[int] = []
    for peak in spectrum.peaks:
        if spectrum.peaks[peak].user_added:
            existing_peak_indexes.append(peak)

    new_peak_index = 500
    while new_peak_index in existing_peak_indexes:
        new_peak_index += 1

    limits = dpg.get_axis_limits("x_axis_plot1")
    position = (limits[0] + limits[1]) / 2
    max_y = get_peak_A_init(spectrum, position)
    new_peak = peak_params(A_init=max_y, x0_init=position, width=width, user_added=True)

    spectrum.peaks[new_peak_index] = new_peak

    dpg.add_drag_line(
        label=f"{new_peak_index}",
        tag=f"drag_line_{new_peak_index}",
        parent="data_plot",
        color=[255, 0, 255, 255],
        default_value=position,
        callback=drag_peak_callback,
        user_data=(spectrum, new_peak_index),
    )
    update_user_peaks_table()
    dpg.draw_line(
        (position, 0),
        (position, max_y),
        parent="data_plot",
        tag=f"peak_width_{new_peak_index}",
        color=peak_width_color,
        thickness=width,
    )


def redraw_user_peaks(render_callback: RenderCallback):
    spectrum = render_callback.spectrum
    for alias in dpg.get_aliases():
        if alias.startswith(f"peak_width_") or alias.startswith(f"drag_line_"):
            dpg.delete_item(alias)

    for peak in spectrum.peaks:
        if spectrum.peaks[peak].user_added:
            x0 = spectrum.peaks[peak].x0_init
            center_slice = spectrum.working_data[:, 1][
                (spectrum.working_data[:, 0] > x0 - 0.5)
                & (spectrum.working_data[:, 0] < x0 + 0.5)
            ]
            max_y = max(center_slice) if len(center_slice) > 0 else 0
            width = spectrum.peaks[peak].width

            dpg.add_drag_line(
                label=f"{peak}",
                tag=f"drag_line_{peak}",
                parent="data_plot",
                color=[255, 0, 255, 255],
                default_value=x0,
                callback=drag_peak_callback,
                user_data=(spectrum, peak),
            )
            update_user_peaks_table()
            dpg.draw_line(
                (x0, 0),
                (x0, max_y),
                parent="data_plot",
                tag=f"peak_width_{peak}",
                color=peak_width_color,
                thickness=width,
            )


drag_peak_last_call = (
    time.time()
)  # there is perf issue and crash with this callback if too many calls


def drag_peak_callback(sender, app_data, user_data: Tuple[MSData, int]):
    global drag_peak_last_call
    if time.time() - drag_peak_last_call < 0.1:
        return
    drag_peak_last_call = time.time()

    spectrum = get_global_msdata_ref()
    peak = user_data[1]
    x0_init = dpg.get_value(sender)

    max_y = get_peak_A_init(spectrum, x0_init)
    spectrum.peaks[peak].A_init = max_y
    spectrum.peaks[peak].x0_init = x0_init
    width = spectrum.peaks[peak].width
    line_height = get_peak_A_init(spectrum, x0_init, baseline=False)

    dpg.delete_item(f"peak_width_{peak}")
    dpg.draw_line(
        (x0_init, 0),
        (x0_init, line_height),
        parent="data_plot",
        tag=f"peak_width_{peak}",
        color=peak_width_color,
        thickness=width,
    )
    update_user_peaks_table()


def get_peak_A_init(spectrum: MSData, x0_init, baseline=True):

    window_length = get_smoothing_window()
    filtered_data = np.array(
        spectrum.get_filtered_data(window_length=window_length, baseline=baseline)
    )
    x_data = spectrum.working_data[:, 0]

    mask = (x_data > x0_init - 0.5) & (x_data < x0_init + 0.5)
    center_slice = filtered_data[mask]
    max_y = max(center_slice) if len(center_slice) > 0 else 0
    return max_y


def update_user_peaks_table():
    spectrum = get_global_msdata_ref()
    dpg.delete_item("user_peak_table", children_only=True)

    dpg.add_table_column(label="Peak", parent="user_peak_table", width=100)
    dpg.add_table_column(label="x0", parent="user_peak_table")
    dpg.add_table_column(label="Width", parent="user_peak_table")
    dpg.add_table_column(label="Del", parent="user_peak_table")

    for peaks in spectrum.peaks:
        if spectrum.peaks[peaks].user_added:
            with dpg.table_row(parent="user_peak_table"):
                dpg.add_text(f"{peaks}")
                dpg.add_text(f"{spectrum.peaks[peaks].x0_init:.1f}")
                # dpg.add_text(f"{spectrum.peaks[peaks].width:.2f}")
                dpg.add_input_int(
                    label="",
                    default_value=int(spectrum.peaks[peaks].width),
                    width=80,
                    tag=f"width_{peaks}",
                    callback=update_peak_width_callback,
                    user_data=(spectrum, peaks),
                )
                dpg.add_button(
                    label="Del",
                    callback=delete_user_peak_callback,
                    user_data=(spectrum, peaks),
                )


def update_peak_width_callback(sender, app_data, user_data: Tuple[RenderCallback, int]):
    spectrum = get_global_msdata_ref()
    peak = user_data[1]
    spectrum.peaks[peak].width = app_data

    if spectrum.peaks[peak].user_added:
        name = f"peak_width_{peak}"
    else:
        name = f"found_peak_line_{peak}_width"

    dpg.delete_item(name)

    x0_init = spectrum.peaks[peak].x0_init
    y = spectrum.peaks[peak].A_init
    dpg.draw_line(
        (x0_init, 0),
        (x0_init, y),
        parent="data_plot",
        tag=name,
        color=(99, 143, 169, 64),
        thickness=app_data,
    )
    update_user_peaks_table()


def delete_user_peak_callback(sender, app_data, user_data: Tuple[MSData, int]):
    spectrum = get_global_msdata_ref()
    peak = user_data[1]
    del spectrum.peaks[peak]
    update_user_peaks_table()
    dpg.delete_item(f"drag_line_{peak}")
    dpg.delete_item(f"peak_width_{peak}")


def peaks_finder_callback(sender, app_data, user_data: RenderCallback):

    spectrum = get_global_msdata_ref()
    threshold = dpg.get_value("peak_detection_threshold")
    width = dpg.get_value("peak_detection_width")
    distance = dpg.get_value("peak_detection_distance")
    filter_window = get_smoothing_window()
    baseline_window = dpg.get_value("baseline_window")
    use_derivative2nd = dpg.get_value("use_2nd_derivative_checkbox")
    if dpg.get_value("threshold_x100"):
        threshold = threshold * 100
    # Save parameters
    spectrum.peak_detection_parameters["threshold"] = int(threshold)
    spectrum.peak_detection_parameters["width"] = int(width)
    spectrum.peak_detection_parameters["distance"] = int(distance)
    spectrum.peak_detection_parameters["use_2nd_derivative"] = use_derivative2nd

    # Scale with sampling rate
    sampling_rate = np.mean(np.diff(spectrum.working_data[:, 0]))
    max_width = 4 * width
    width = width / sampling_rate
    distance = distance / sampling_rate
    peaks_finder(
        spectrum,
        threshold,
        width,
        max_width,
        distance,
        filter_window,
        baseline_window,
        use_derivative2nd,
    )
    update_found_peaks_table(user_data)


def peaks_clear_callback(sender, app_data, user_data: RenderCallback):
    spectrum = get_global_msdata_ref()
    peak_to_delete = []
    for old_peak in spectrum.peaks:
        if spectrum.peaks[old_peak].user_added:
            continue
        peak_to_delete.append(old_peak)
    for peak in peak_to_delete:
        del spectrum.peaks[peak]
    update_found_peaks_table(user_data=get_global_render_callback_ref())
    draw_found_peaks(spectrum)


def update_found_peaks_table(user_data: RenderCallback):
    spectrum = get_global_msdata_ref()
    dpg.delete_item("found_peak_table", children_only=True)

    dpg.add_table_column(label="#", parent="found_peak_table")
    dpg.add_table_column(label="x0", parent="found_peak_table")
    dpg.add_table_column(label="Width", parent="found_peak_table")
    dpg.add_table_column(label="Use", parent="found_peak_table")

    for peaks in spectrum.peaks:
        if spectrum.peaks[peaks].user_added:
            continue
        with dpg.table_row(parent="found_peak_table"):
            dpg.add_text(f"{peaks}")
            dpg.add_text(f"{spectrum.peaks[peaks].x0_init:.1f}")
            # dpg.add_text(f"{spectrum.peaks[peaks].width:.2f}")
            dpg.add_input_int(
                label="",
                default_value=int(spectrum.peaks[peaks].width),
                tag=f"width_{peaks}",
                callback=update_peak_width_callback,
                user_data=(user_data, peaks),
            )
            checked = not spectrum.peaks[peaks].do_not_fit
            dpg.add_checkbox(
                label="",
                default_value=checked,
                callback=tick_peak_callback,
                user_data=(user_data, peaks),
            )


def tick_peak_callback(sender, app_data, user_data: Tuple[RenderCallback, int]):
    spectrum = get_global_msdata_ref()
    peak = user_data[1]
    spectrum.peaks[peak].do_not_fit = not spectrum.peaks[peak].do_not_fit
    draw_found_peaks(spectrum)


def peaks_finder(
    spectrum: MSData,
    threshold: int,
    width: int,
    max_width: int,
    distance: int,
    filter_window: int,
    baseline_window: int,
    use_derivative2nd: bool = True,
):

    spectrum.correct_baseline(baseline_window)
    baseline = spectrum.baseline[:, 1]

    if use_derivative2nd:
        filtered = spectrum.get_2nd_derivative(filter_window)
    else:
        filtered = spectrum.get_filtered_data(filter_window)
        filtered = np.where(
            np.abs(filtered - baseline) <= threshold, 0, (filtered - baseline)
        )
    peaks, peaks_data = find_peaks(filtered, width=width, distance=distance)

    # Delete previous peaks
    peak_to_delete = []
    for old_peak in spectrum.peaks:
        # Do not delete custom peaks
        if spectrum.peaks[old_peak].user_added:
            continue

        if (
            spectrum.peaks[old_peak].x0_init > spectrum.working_data[:, 0][0]
            and spectrum.peaks[old_peak].x0_init < spectrum.working_data[:, 0][-1]
        ):
            peak_to_delete.append(old_peak)
    for peak in peak_to_delete:
        del spectrum.peaks[peak]

    try:
        new_peak_index = (
            max(
                [
                    key
                    for key in spectrum.peaks.keys()
                    if not spectrum.peaks[key].user_added
                ]
            )
            + 1
        )
    except ValueError:
        new_peak_index = 0

    filtered_data = spectrum.get_filtered_data(
        window_length=filter_window, baseline=True
    )
    # Iterate over the peaks and add them to the dictionary
    i = 0
    for peak in peaks:
        sample = spectrum.working_data[:, 0][peak - 250 : peak + 250]
        sampling_rate = np.mean(np.diff(sample))
        width = peaks_data["widths"][i] * sampling_rate

        if width > max_width:
            width = max_width
        if width <= 0:
            width = 1
        if np.isnan(width):
            width = 1

        A_init = filtered_data[peak]
        if A_init < threshold:
            continue

        width_half = int(peaks_data["widths"][i] / 2)
        search_start = max(0, peak - width_half)
        search_end = min(len(filtered_data), peak + width_half)

        local_region_x = spectrum.working_data[search_start:search_end, 0]
        local_region_y = filtered_data[search_start:search_end]

        # Find the index of maximum intensity in the local region
        local_max_idx = np.argmax(local_region_y)
        corrected_x0_init = (
            local_region_x[local_max_idx] + spectrum.working_data[peak, 0]
        ) / 2

        # if use_derivative2nd:
        #     width = width * 2  # 2nd derivative makes peaks thinner

        new_peak = peak_params(
            A_init=A_init, x0_init=float(corrected_x0_init), width=width
        )
        spectrum.peaks[new_peak_index] = new_peak
        new_peak_index += 1
        i += 1

    peaks_centers = spectrum.working_data[:, 0][peaks]
    log(f"Detected {len(peaks)} peaks at x = {peaks_centers}")
    draw_found_peaks(spectrum)


def draw_found_peaks(spectrum: MSData):
    for alias in dpg.get_aliases():
        if alias.startswith(f"found_peak_line") or alias.startswith(
            f"found_peak_annotation"
        ):
            dpg.delete_item(alias)

    for peak in spectrum.peaks:
        if spectrum.peaks[peak].user_added:
            continue
        x0 = spectrum.peaks[peak].x0_init
        if spectrum.peaks[peak].do_not_fit:
            color = (64, 64, 90)
        else:
            color = (264, 24, 24)
        center_slice = spectrum.working_data[:, 1][
            (spectrum.working_data[:, 0] > x0 - 0.5)
            & (spectrum.working_data[:, 0] < x0 + 0.5)
        ]
        max_y = max(center_slice) if len(center_slice) > 0 else 0

        width = int(spectrum.peaks[peak].width)

        line_thickness = int(1.5 if width > 4 else width / 8)
        dpg.draw_line(
            (x0, 0),
            (x0, max_y),
            parent="data_plot",
            tag=f"found_peak_line_{peak}",
            color=color,
            thickness=line_thickness,
        )
        dpg.draw_line(
            (x0, 0),
            (x0, max_y),
            parent="data_plot",
            tag=f"found_peak_line_{peak}_width",
            color=peak_width_color,
            thickness=width,
        )
        dpg.add_plot_annotation(
            label=f"{peak}",
            default_value=(x0, max_y),
            parent="data_plot",
            tag=f"found_peak_annotation_{peak}",
            color=color,
        )
