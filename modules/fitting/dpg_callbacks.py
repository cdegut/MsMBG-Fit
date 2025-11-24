import numpy as np
import dearpygui.dearpygui as dpg
from sklearn import base
from modules.fitting.MBGfit import MBG_fit, run_advanced_statistical_analysis
from modules.fitting.draw_MBG import show_MBG
from modules.matching import redraw_blocks
from modules.rendercallback import RenderCallback
from modules.data_structures import MSData, get_global_msdata_ref
from modules.math import (
    bi_Lorentzian_integral_numerical,
    bi_gaussian,
    bi_gaussian_integral,
    bi_Lorentzian,
    bi_Lorentzian_integral,
)
import seaborn as sns


def draw_fitted_peaks_callback():
    draw_fitted_peaks(False)


def show_residual_callback(sender, app_data):
    if app_data:
        dpg.show_item("residual")
    else:
        dpg.hide_item("residual")
    return


def stop_fitting(sender, app_data, user_data: RenderCallback):
    user_data.stop_fitting = True
    print("Fitting stopped")
    return


def draw_initial_peaks_callback(sender, app_data, user_data: RenderCallback):
    spectrum = get_global_msdata_ref()
    for alias in dpg.get_aliases():
        if alias.startswith("initial_peak_"):
            dpg.delete_item(alias)

    i = 0
    for peak in spectrum.peaks:
        x0_init = spectrum.peaks[peak].x0_init
        if (
            x0_init < spectrum.working_data[:, 0][0]
            or x0_init > spectrum.working_data[:, 0][-1]
        ):
            continue

        A = spectrum.peaks[peak].A_init
        sigma_L = spectrum.peaks[peak].width / 2
        sigma_R = spectrum.peaks[peak].width / 2

        x_individual_fit = np.linspace(
            x0_init - 4 * sigma_L, x0_init + 4 * sigma_R, 500
        )
        if spectrum.peak_model == "lorentzian":
            y_individual_fit = bi_Lorentzian(
                x_individual_fit, A, x0_init, sigma_L, sigma_R
            )
        else:
            y_individual_fit = bi_gaussian(
                x_individual_fit, A, x0_init, sigma_L, sigma_R
            )

        dpg.add_line_series(
            x_individual_fit.tolist(),
            y_individual_fit.tolist(),
            label=f"Peak {peak}",
            parent="y_axis_plot2",
            tag=f"initial_peak_{peak}",
        )
        dpg.bind_item_theme(f"initial_peak_{peak}", f"initial_peaks_theme_{peak}")
        dpg.add_plot_annotation(
            label=f"{peak}",
            default_value=(x0_init, A),
            offset=(-15, -15),
            color=[120, 120, 120],
            clamped=False,
            parent="gaussian_fit_plot",
            tag=f"initial_peak_annotation_{peak}",
        )
        i += 1


def run_fitting_callback(sender, app_data, user_data: RenderCallback):
    render_callback = user_data
    dpg.show_item("Fitting_indicator")
    k = dpg.get_value("fitting_iterations")
    theta_threshold = dpg.get_value("theta_threshold_selector") * 1e-5
    dpg.set_value("stop_fitting_checkbox", False)
    dpg.hide_item("start_fitting_button")
    dpg.show_item("stop_fitting_checkbox")
    dpg.hide_item("advanced_statistical_analysis_button")
    use_gaussian = dpg.get_value("use_gaussian")

    if dpg.get_value("show_residual_checkbox"):
        dpg.show_item("residual")

    if dpg.does_alias_exist("MBG_plot2"):
        dpg.hide_item("MBG_plot2")

    if dpg.does_alias_exist("fitting_residual_plot1"):
        dpg.delete_item("fitting_residual_plot1")
        dpg.set_item_label("fitting_residual_plot1_button", "Show Fitting Residual")

    use_filtered = dpg.get_value("use_filtered")
    user_data.stop_fitting = False
    draw_fitted_peaks(delete=True)
    MBG_fit(render_callback, k, theta_threshold, use_filtered, use_gaussian)
    draw_fitted_peaks()
    redraw_blocks()
    dpg.hide_item("stop_fitting_checkbox")
    dpg.hide_item("Fitting_indicator")
    dpg.show_item("start_fitting_button")
    dpg.show_item("advanced_statistical_analysis_button")


def draw_base_projection():
    if not dpg.get_value("show_projection_checkbox"):
        for alias in dpg.get_aliases():
            if alias.startswith("fitted_regression_"):
                dpg.delete_item(alias)
        return

    spectrum: MSData = get_global_msdata_ref()

    for peak in spectrum.peaks:
        if not spectrum.peaks[peak].fitted:
            continue
        spectrum.peaks[peak].matched_with = []  # Reset matched_with for all peaks

        regression_0 = (
            -spectrum.peaks[peak].regression_fct[1]
            / spectrum.peaks[peak].regression_fct[0]
        )
        regression_x = regression_0 + spectrum.peaks[peak].sigma_L * 5
        regression_y = (
            spectrum.peaks[peak].regression_fct[0] * regression_x
            + spectrum.peaks[peak].regression_fct[1]
        )

        dpg.draw_line(
            (regression_0, 0),
            (regression_x, regression_y),
            parent="gaussian_fit_plot",
            color=(237, 43, 43),
            thickness=1,
            tag=f"fitted_regression_{peak}",
        )


def draw_residual(x_data, residual):
    dpg.show_item("residual")
    dpg.set_value("residual", [x_data, residual])


def draw_fitted_peaks(delete=False):
    spectrum = get_global_msdata_ref()
    # Delete previous peaks
    for alias in dpg.get_aliases():
        if (
            alias.startswith("fitted_peak_")
            or alias.startswith("peak_annotation_")
            or alias.startswith("fitted_peaks_theme_")
        ):
            dpg.delete_item(alias)
    if delete:
        return
    # Generate fitted curve
    peak_list = []
    mbg_param = []
    colors = sns.color_palette("plasma", 20)

    i = 0
    for peak in spectrum.peaks:
        x0_fit = spectrum.peaks[peak].x0_refined
        if (
            x0_fit < spectrum.working_data[:, 0][0]
            or x0_fit > spectrum.working_data[:, 0][-1]
        ):
            continue
        if not spectrum.peaks[peak].fitted:
            continue

        peak_error = spectrum.peaks[peak].fit_quality.relative_error * 3
        normalized_error = np.clip(peak_error, 0, 1)
        color_idx = int(normalized_error * (len(colors) - 1))
        color = [int(c * 255) for c in colors[color_idx]]

        shade_color = color.copy()
        shade_color.append(100)

        with dpg.theme(tag=f"fitted_peaks_theme_{peak}"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 3, category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_Fill, shade_color, category=dpg.mvThemeCat_Plots
                )

        A = spectrum.peaks[peak].A_refined
        sigma_L_fit = spectrum.peaks[peak].sigma_L
        sigma_R_fit = spectrum.peaks[peak].sigma_R
        peak_list.append(peak)

        x_individual_fit = np.linspace(
            x0_fit - 4 * sigma_L_fit, x0_fit + 4 * sigma_R_fit, 500
        )
        if spectrum.peak_model == "lorentzian":
            y_individual_fit = bi_Lorentzian(
                x_individual_fit, A, x0_fit, sigma_L_fit, sigma_R_fit
            )
        else:
            y_individual_fit = bi_gaussian(
                x_individual_fit, A, x0_fit, sigma_L_fit, sigma_R_fit
            )
        mbg_param.extend([A, x0_fit, sigma_L_fit, sigma_R_fit])

        dpg.add_line_series(
            x_individual_fit.tolist(),
            y_individual_fit.tolist(),
            label=f"Peak {peak}",
            parent="y_axis_plot2",
            tag=f"fitted_peak_{peak}",
        )
        dpg.bind_item_theme(f"fitted_peak_{peak}", f"fitted_peaks_theme_{peak}")
        dpg.add_shade_series(
            x_individual_fit.tolist(),
            y_individual_fit.tolist(),
            label=f"Peak {peak} area",
            parent="y_axis_plot2",
            tag=f"fitted_peak_{peak}_area",
            show=True,
        )
        dpg.bind_item_theme(f"fitted_peak_{peak}_area", f"fitted_peaks_theme_{peak}")
        dpg.add_plot_annotation(
            label=f"Peak {peak}",
            default_value=(x0_fit, A),
            offset=(-15, -15),
            color=[120, 120, 120],
            clamped=False,
            parent="gaussian_fit_plot",
            tag=f"peak_annotation_{peak}",
        )
        i += 1

    show_MBG(spectrum)
    update_peak_table(spectrum)


def update_peak_table(spectrum: MSData):
    children = dpg.get_item_children("peak_table")
    if children and len(children) > 1 and isinstance(children[1], list):
        for tag in children[1]:
            dpg.delete_item(tag)

    for peak in spectrum.peaks:
        if not spectrum.peaks[peak].fitted:
            continue
        apex = spectrum.peaks[peak].x0_refined
        sigma_L = spectrum.peaks[peak].sigma_L
        sigma_R = spectrum.peaks[peak].sigma_R
        A_refined = spectrum.peaks[peak].A_refined
        start = spectrum.peaks[peak].start_end[0]
        end = spectrum.peaks[peak].start_end[1]

        if spectrum.peak_model == "lorentzian":
            integral = bi_Lorentzian_integral(A_refined, sigma_L, sigma_R)
        else:
            integral = bi_gaussian_integral(A_refined, sigma_L, sigma_R)

        spectrum.peaks[peak].integral = integral
        rel_error = spectrum.peaks[peak].fit_quality.relative_error

        # Get standard errors if available
        se_x0 = spectrum.peaks[peak].se_x0 if spectrum.peaks[peak].se_x0 > 0 else None
        se_integral = (
            spectrum.peaks[peak].se_integral
            if spectrum.peaks[peak].se_integral > 0
            else None
        )

        if spectrum.peaks[peak].regression_fct[0] == 0:
            regression_0 = 0
        else:
            regression_0 = (
                -spectrum.peaks[peak].regression_fct[1]
                / spectrum.peaks[peak].regression_fct[0]
            )

        with dpg.table_row(parent="peak_table"):
            dpg.add_text(f"Peak {peak}")
            base_text = f"{regression_0:.2f}" + (
                f" ± {spectrum.peaks[peak].se_base:.2f}"
                if spectrum.peaks[peak].se_base > 0
                else ""
            )
            dpg.add_text(base_text)
            apex_text = f"{apex:.2f}" + (f" ± {se_x0:.2f}" if se_x0 else "")
            dpg.add_text(apex_text)
            integral_text = f"{integral:.0f}" + (
                f" ± {(se_integral / integral * 100):.2f}%" if se_integral else ""
            )
            dpg.add_text(integral_text)
            sigma_L_text = f"{sigma_L:.2f}" + (
                f" ± {spectrum.peaks[peak].se_sigma_L:.2f}"
                if spectrum.peaks[peak].se_sigma_L > 0
                else ""
            )
            sigma_R_text = f"{sigma_R:.2f}" + (
                f" ± {spectrum.peaks[peak].se_sigma_R:.2f}"
                if spectrum.peaks[peak].se_sigma_R > 0
                else ""
            )
            dpg.add_text(f"{sigma_L_text}")
            dpg.add_text(f"{sigma_R_text}")
            dpg.add_text(f"{rel_error:.4f}")
            dpg.add_text(f"{spectrum.peaks[peak].fit_quality.r_squared:.4f}")


def run_advanced_statistical_analysis_callback():
    spectrum = get_global_msdata_ref()
    dpg.show_item("Fitting_indicator")
    dpg.set_value("stop_fitting_checkbox", False)
    dpg.hide_item("start_fitting_button")
    dpg.show_item("stop_fitting_checkbox")
    dpg.hide_item("advanced_statistical_analysis_button")
    dpg.set_value(
        "Fitting_indicator_text",
        "Running bootstrap and randomisation, this might take a while...",
    )
    run_advanced_statistical_analysis()
    update_peak_table(get_global_msdata_ref())
    dpg.hide_item("Fitting_indicator")
    dpg.show_item("start_fitting_button")
    dpg.hide_item("stop_fitting_checkbox")
    dpg.show_item("advanced_statistical_analysis_button")
    if dpg.does_alias_exist("noise"):
        dpg.delete_item("noise")


def toggle_lorentzian_peak_model(sender, app_data):
    spectrum = get_global_msdata_ref()
    if app_data:
        spectrum.peak_model = "lorentzian"
    else:
        spectrum.peak_model = "gaussian"
