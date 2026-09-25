import numpy as np
import dearpygui.dearpygui as dpg
from sklearn import base
from modules.fitting.MBGfit import MBG_fit, run_advanced_statistical_analysis
from modules.fitting.draw_MBG import show_MBG
from modules.matching import redraw_blocks
from modules.rendercallback import RenderCallback
from modules.data_structures import (
    FitSummary,
    MSData,
    get_global_msdata_ref,
    peak_params,
)
from modules.math import (
    bi_Lorentzian_integral_numerical,
    bi_gaussian,
    bi_gaussian_integral,
    bi_Lorentzian,
    bi_Lorentzian_integral,
)
from modules.fitting.fitting_quality import (
    CONVERGENCE_WINDOW,
    R2_FLAT_WINDOW,
    laplace_covariance_analysis,
)
from modules.rendercallback import get_global_render_callback_ref
import seaborn as sns

# Fitted peaks are coloured by relative error, from 0 to ERROR_COLOR_MAX
PEAK_COLORS = sns.color_palette("plasma", 20)
ERROR_COLOR_MAX = 1 / 3

# (header, sort key, width weight) ; sort key None = not sortable
PEAK_TABLE_COLUMNS = [
    ("", None, 0.15),
    ("Peak", "peak", 0.5),
    ("Apex m/z", "apex", 0.9),
    ("Start m/z", "start", 0.9),
    ("Sigma L", "sigma_L", 0.7),
    ("Sigma R", "sigma_R", 0.7),
    ("Integral", "integral", 0.9),
    ("Share %", "share", 0.6),
    ("Rel. error", "rel_error", 0.6),
    ("R²", "r2", 0.6),
    ("SNR", "snr", 0.5),
    ("Area corr. L / R", "area_corr", 1.1),
    ("Flags", None, 1.75),
]


def peak_color(peak: peak_params) -> list[int]:
    normalized_error = np.clip(
        peak.fit_quality.relative_error / ERROR_COLOR_MAX, 0, 1
    )
    color_idx = min(int(normalized_error * len(PEAK_COLORS)), len(PEAK_COLORS) - 1)
    return [int(c * 255) for c in PEAK_COLORS[color_idx]]


def quality_theme(value: float, good: float, warn: float, higher_is_better=True):
    if not np.isfinite(value):
        return "text_bad_theme"
    if not higher_is_better:
        value, good, warn = -value, -good, -warn
    if value >= good:
        return "text_good_theme"
    if value >= warn:
        return "text_warn_theme"
    return "text_bad_theme"


def show_stop_button():
    dpg.set_value("stop_fitting_button", False)
    dpg.set_item_label("stop_fitting_button", "Stop fitting")
    dpg.show_item("stop_fitting_frame")


STOP_REASON_SHORT = {
    "max_iter": "iteration limit (not converged)",
    "theta": "parameters stable",
    "r2": "R²",
    "flat": "R² flat",
    "user": "user",
}


def stop_reason_short(reason: str) -> str:
    """Short text of a stop reason; converged reasons are criteria joined by '+'."""
    if reason in STOP_REASON_SHORT:
        return STOP_REASON_SHORT[reason]
    return " + ".join(STOP_REASON_SHORT.get(name, name) for name in reason.split("+"))


def update_stop_criteria_label():
    """Label of the collapsible 'Stop at' panel: settings and last stop reason."""
    label = (
        f"Stop at: {dpg.get_value('fitting_iterations')} it. | "
        f"moves < {dpg.get_value('theta_threshold_selector'):g} SE / {CONVERGENCE_WINDOW} it. | "
        f"R² > {dpg.get_value('fitting_r2'):g}"
    )
    if dpg.get_value("fitting_r2_flat") > 0:
        label += f" | R² gain < {dpg.get_value('fitting_r2_flat'):g} % / {R2_FLAT_WINDOW} it."
    summary: FitSummary | None = get_global_render_callback_ref().fit_summary
    if summary is not None:
        label += f"  ->  stopped by {stop_reason_short(summary.stop_reason)}"
    dpg.set_item_label("stop_criteria_node", label)


def reset_stop_reason():
    for block in ("iter", "theta", "r2", "flat"):
        dpg.bind_item_theme(f"stop_{block}_label", 0)
        dpg.set_value(f"stop_{block}_status", "")
        dpg.bind_item_theme(f"stop_{block}_status", 0)
    dpg.bind_item_theme("Fitting_indicator_text", 0)
    update_stop_criteria_label()


def show_stop_reason(fit_summary: FitSummary | None):
    """Highlight which of the stopping criteria ended the fit."""
    reset_stop_reason()
    if fit_summary is None:
        return
    reason = fit_summary.stop_reason
    criteria = reason.split("+")
    hit = {
        "iter": reason == "max_iter",
        "theta": "theta" in criteria,
        "r2": "r2" in criteria,
        "flat": "flat" in criteria,
    }
    dpg.set_value(
        "stop_iter_status",
        f"used {fit_summary.iterations_done} / {fit_summary.max_iterations}",
    )
    dpg.set_value(
        "stop_theta_status",
        f"largest move: {fit_summary.delta_theta:.2f} error bars"
        if np.isfinite(fit_summary.delta_theta)
        else "largest move: not measured yet",
    )
    dpg.set_value("stop_r2_status", f"last R²: {fit_summary.r_squared:.4f}")
    dpg.set_value(
        "stop_flat_status",
        f"last gain: {100 * fit_summary.r2_flat_gain:.3g} % / {R2_FLAT_WINDOW} it."
        if np.isfinite(fit_summary.r2_flat_gain)
        else "last gain: not measured yet",
    )

    for block, is_hit in hit.items():
        if not is_hit:
            dpg.bind_item_theme(f"stop_{block}_status", "text_muted_theme")
            continue
        # Hitting the iteration cap means the fit did not converge
        theme = "text_warn_theme" if block == "iter" else "text_good_theme"
        dpg.bind_item_theme(f"stop_{block}_label", theme)
        dpg.bind_item_theme(f"stop_{block}_status", theme)
        dpg.set_value(
            f"stop_{block}_status",
            ">> " + dpg.get_value(f"stop_{block}_status") + "  <- stopped the fit",
        )

    indicator_theme = {
        "max_iter": "text_warn_theme",
        "user": "text_muted_theme",
    }.get(reason, "text_good_theme")
    dpg.bind_item_theme("Fitting_indicator_text", indicator_theme)
    update_stop_criteria_label()


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
    theta_threshold = dpg.get_value("theta_threshold_selector")  # in error bars
    dpg.hide_item("start_fitting_button")
    dpg.hide_item("fit_options_button")
    show_stop_button()
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
    render_callback.fit_summary = None
    reset_stop_reason()
    draw_fitted_peaks(delete=True)
    MBG_fit(render_callback, k, theta_threshold, use_filtered, use_gaussian)
    show_stop_reason(render_callback.fit_summary)
    if render_callback.fit_summary is not None:
        laplace_covariance_analysis()  # fast, keeps the Laplace columns up to date
    draw_fitted_peaks()
    redraw_blocks()
    dpg.hide_item("stop_fitting_frame")
    dpg.hide_item("Fitting_indicator")
    dpg.show_item("start_fitting_button")
    dpg.show_item("fit_options_button")
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

        color = peak_color(spectrum.peaks[peak])
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


def peak_flags(spectrum: MSData, peak: int) -> list[tuple[str, str]]:
    """Short warnings (label, explanation) worth checking for a fitted peak."""
    p = spectrum.peaks[peak]
    q = p.fit_quality
    flags = []
    if q.relative_error > 0.15:
        flags.append(
            ("high error", "Relative error > 0.15: hidden in matching when 'hide high error' is on")
        )
    if q.r_squared < 0.85:
        flags.append(("low R²", f"Local R² = {q.r_squared:.3f} (< 0.85)"))
    if q.snr < 3:
        flags.append(("low SNR", f"Peak height / local residual noise = {q.snr:.1f} (< 3)"))
    if p.sampling_rate > 0 and min(p.sigma_L, p.sigma_R) < p.sampling_rate * 4:
        flags.append(("narrow", "A width is close to its lower clamp (3x sampling rate)"))
    if p.width > 0 and abs(p.x0_refined - p.x0_init) > 0.9 * p.width:
        flags.append(("apex drift", "Apex moved close to the allowed limit (one initial width)"))
    if p.laplace_area_corr <= -0.8:
        flags.append(
            (
                "shared area",
                f"Integral anticorrelated with Peak {p.laplace_area_peak} "
                f"(r = {p.laplace_area_corr:+.2f}): area can move between them",
            )
        )
    if p.se_integral > 0 and p.integral > 0 and p.se_integral / p.integral > 0.15:
        flags.append(
            ("uncertain area", f"Integral uncertainty {p.se_integral / p.integral * 100:.1f}% (> 15%)")
        )
    return flags


def with_se(value: float, se: float, fmt: str = ".2f") -> str:
    return f"{value:{fmt}}" + (f" ± {se:{fmt}}" if se > 0 else "")


def add_themed_text(text: str, theme: str | None = None, tooltip: str | None = None):
    if not tooltip:
        item = dpg.add_text(text)
        if theme:
            dpg.bind_item_theme(item, theme)
        return item
    # Group so the tooltip does not take its own table cell
    with dpg.group() as cell:
        item = dpg.add_text(text)
        if theme:
            dpg.bind_item_theme(item, theme)
        with dpg.tooltip(item):
            dpg.add_text(tooltip, wrap=400)
    return cell


def residual_metrics(spectrum: MSData) -> dict | None:
    """Size of what the model does not explain (e.g. unaccounted species)."""
    fitted = [p for p in spectrum.peaks.values() if p.fitted and not p.do_not_fit]
    data = spectrum.baseline_corrected
    if not fitted or len(data) < 3:
        return None
    x, y = data[:, 0], data[:, 1]
    residual = y - spectrum.calculate_mbg(x)
    # Light smoothing (1/10 of a peak width) so a single noise spike is not reported
    width_pts = np.median([p.sigma_L + p.sigma_R for p in fitted]) / np.median(np.diff(x))
    window = int(np.clip(width_pts / 10, 1, len(x)))
    smooth = np.convolve(residual, np.ones(window) / window, mode="same")
    base_peak = float(np.max(y))
    signal = float(np.sum(np.clip(y, 0, None)))
    if base_peak <= 0 or signal <= 0:
        return None
    i_max = int(np.argmax(np.abs(smooth)))
    return {
        "max_pct": abs(smooth[i_max]) / base_peak * 100,
        "max_mz": float(x[i_max]),
        "max_positive": bool(smooth[i_max] > 0),
        "unexplained_pct": float(np.sum(np.clip(smooth, 0, None))) / signal * 100,
    }


def update_fit_summary():
    dpg.delete_item("fit_summary_group", children_only=True)
    spectrum = get_global_msdata_ref()
    summary: FitSummary | None = get_global_render_callback_ref().fit_summary
    with dpg.group(parent="fit_summary_group", horizontal=True, horizontal_spacing=30):
        metrics = residual_metrics(spectrum)
        if metrics is not None:
            add_themed_text(
                f"Max residual {metrics['max_pct']:.1f}% of base peak "
                f"@ {metrics['max_mz']:.0f} m/z",
                quality_theme(metrics["max_pct"], 5, 10, higher_is_better=False),
                (
                    "Largest local residual, relative to the highest data point. "
                    + (
                        "Data above the model: signal not explained by any peak (missing species?)"
                        if metrics["max_positive"]
                        else "Model above the data: a peak is too large or too wide"
                    )
                ),
            )
            add_themed_text(
                f"Unexplained signal {metrics['unexplained_pct']:.1f}%",
                quality_theme(metrics["unexplained_pct"], 5, 10, higher_is_better=False),
                "Positive residual area (data above the model) as a fraction of the total signal",
            )
        if summary is None:
            add_themed_text("No other fit statistics for this session yet.", "text_muted_theme")
            return
        add_themed_text(
            f"R² {summary.r_squared:.4f}",
            quality_theme(summary.r_squared, summary.r_squared_threshold, 0.95),
            "Global coefficient of determination",
        )
        add_themed_text(
            f"wRMSE {summary.weighted_rmse:.4g}",
            tooltip="RMSE with 5x weight on peak regions (used for convergence)",
        )
        add_themed_text(
            f"X²r {summary.chi_squared_reduced:.3f}",
            tooltip="Reduced chi-squared, noise estimated from baseline regions. ~1 means residuals are at noise level",
        )
        add_themed_text(f"SNR {summary.signal_to_noise:.1f}", tooltip="Mean signal / noise std")
        add_themed_text(f"AIC {summary.aic:.0f}", tooltip="Akaike information criterion (lower is better, compare fits of the same data)")
        add_themed_text(f"BIC {summary.bic:.0f}", tooltip="Bayesian information criterion (lower is better, compare fits of the same data)")
        add_themed_text(f"{summary.iterations_done} it. in {summary.time_taken:.1f}s", "text_muted_theme")


def has_error_components(p: peak_params) -> bool:
    return p.se_integral_bootstrap > 0 or p.se_integral_restart > 0


def update_error_analysis_status(spectrum: MSData):
    fitted = [p for p in spectrum.peaks.values() if p.fitted and not p.do_not_fit]
    if not fitted:
        text, theme = "", None
    elif all(has_error_components(p) for p in fitted):
        text, theme = "Done: table ± are the final errors.", "text_good_theme"
    else:
        text = (
            "Not run for this fit: ± are Laplace only (lower bound). "
            "Required before publishing the data."
        )
        theme = "text_warn_theme"
    dpg.set_value("error_analysis_status", text)
    dpg.bind_item_theme("error_analysis_status", theme or 0)


def error_breakdown(total: float, bootstrap: float, laplace: float, restart: float, fmt) -> str:
    def show(v):
        return fmt(v) if v > 0 else "n/a"

    return (
        f"± {fmt(total)} = max(bootstrap {show(bootstrap)}, Laplace {show(laplace)}, "
        f"restarts {show(restart)})"
    )


def update_peak_table(spectrum: MSData):
    update_fit_summary()
    update_error_analysis_status(spectrum)
    dpg.delete_item("peak_table", children_only=True, slot=1)

    fitted_peaks = [peak for peak in spectrum.peaks if spectrum.peaks[peak].fitted]
    fitted_peaks.sort(key=lambda peak: spectrum.peaks[peak].x0_refined)

    for peak in fitted_peaks:
        p = spectrum.peaks[peak]
        if spectrum.peak_model == "lorentzian":
            p.integral = bi_Lorentzian_integral(p.A_refined, p.sigma_L, p.sigma_R)
        else:
            p.integral = bi_gaussian_integral(p.A_refined, p.sigma_L, p.sigma_R)
    total_integral = sum(spectrum.peaks[peak].integral for peak in fitted_peaks)

    for peak in fitted_peaks:
        p = spectrum.peaks[peak]
        q = p.fit_quality

        if p.regression_fct[0] == 0:
            regression_0 = 0
        else:
            regression_0 = -p.regression_fct[1] / p.regression_fct[0]
        share = p.integral / total_integral * 100 if total_integral > 0 else 0.0
        flags = peak_flags(spectrum, peak)

        sort_values = {
            "peak": peak,
            "apex": p.x0_refined,
            "start": regression_0,
            "sigma_L": p.sigma_L,
            "sigma_R": p.sigma_R,
            "integral": p.integral,
            "share": share,
            "rel_error": q.relative_error,
            "r2": q.r_squared,
            "snr": q.snr if np.isfinite(q.snr) else 1e12,
            "area_corr": p.laplace_area_corr,
        }

        with dpg.table_row(parent="peak_table", user_data=sort_values):
            dpg.add_color_button(
                peak_color(p),
                width=16,
                height=16,
                no_alpha=True,
                no_border=True,
                no_drag_drop=True,
                no_tooltip=True,
                callback=zoom_to_peak,
                user_data=peak,
            )
            dpg.add_selectable(
                label=f"Peak {peak}",
                span_columns=True,
                callback=zoom_to_peak,
                user_data=peak,
            )
            add_themed_text(
                with_se(p.x0_refined, p.se_x0),
                tooltip=(
                    error_breakdown(
                        p.se_x0, p.se_x0_bootstrap, p.laplace_se_x0, p.se_x0_restart,
                        lambda v: f"{v:.2f} m/z",
                    )
                    if has_error_components(p)
                    else "Laplace only (noise-limited, a lower bound). Run the final error analysis to include random restarts."
                ),
            )
            add_themed_text(with_se(regression_0, p.se_base))
            add_themed_text(with_se(p.sigma_L, p.se_sigma_L))
            add_themed_text(with_se(p.sigma_R, p.se_sigma_R))
            if p.se_integral > 0 and p.integral > 0:
                se_pct = p.se_integral / p.integral * 100
                breakdown = "Laplace only (noise-limited, a lower bound). Run the final error analysis to include random restarts."
                if has_error_components(p):
                    breakdown = error_breakdown(
                        p.se_integral,
                        p.se_integral_bootstrap,
                        p.laplace_se_integral * p.integral,
                        p.se_integral_restart,
                        lambda v: f"{v / p.integral * 100:.1f}%",
                    )
                add_themed_text(
                    f"{p.integral:.0f} ± {se_pct:.1f}%",
                    quality_theme(se_pct, 5, 15, higher_is_better=False),
                    breakdown,
                )
            else:
                add_themed_text(f"{p.integral:.0f}")
            add_themed_text(f"{share:.1f}")
            add_themed_text(
                f"{q.relative_error:.4f}",
                quality_theme(q.relative_error, 0.05, 0.15, higher_is_better=False),
            )
            add_themed_text(f"{q.r_squared:.4f}", quality_theme(q.r_squared, 0.95, 0.85))
            add_themed_text(f"{q.snr:.1f}", quality_theme(q.snr, 10, 3))
            if p.laplace_se_integral >= 0:
                sides = [
                    (side, other, r)
                    for side, other, r in (
                        ("left", p.laplace_left_peak, p.laplace_corr_left),
                        ("right", p.laplace_right_peak, p.laplace_corr_right),
                    )
                ]
                add_themed_text(
                    " / ".join(f"{r:+.2f}" if other >= 0 else "-" for _, other, r in sides),
                    # Only anticorrelation means area exchange
                    quality_theme(max(0.0, -p.laplace_area_corr), 0.5, 0.8, higher_is_better=False),
                    "Correlation of this integral with the integrals of its m/z neighbours:\n"
                    + "\n".join(
                        f"  {side}: Peak {other}, r = {r:+.2f}"
                        for side, other, r in sides
                        if other >= 0
                    )
                    + "\nClose to -1: the data cannot tell how area is split between the two peaks",
                )
            else:
                add_themed_text("-", "text_muted_theme", "Run the Laplace error analysis")
            if flags:
                add_themed_text(
                    ", ".join(label for label, _ in flags),
                    "text_warn_theme",
                    "\n".join(f"{label}: {why}" for label, why in flags),
                )
            else:
                add_themed_text("ok", "text_good_theme")


def sort_peak_table(sender, sort_specs):
    if not sort_specs:
        return
    column, direction = sort_specs[0]
    key = dpg.get_item_user_data(column)
    rows = dpg.get_item_children(sender, 1)
    rows.sort(key=lambda row: dpg.get_item_user_data(row)[key], reverse=direction < 0)
    dpg.reorder_items(sender, 1, rows)


def zoom_to_peak(sender, app_data, user_data):
    if dpg.get_item_type(sender) == "mvAppItemType::mvSelectable":
        dpg.set_value(sender, False)
    spectrum = get_global_msdata_ref()
    p = spectrum.peaks[user_data]
    x_min = p.x0_refined - 5 * p.sigma_L
    x_max = p.x0_refined + 5 * p.sigma_R
    data = spectrum.baseline_corrected
    in_view = data[(data[:, 0] >= x_min) & (data[:, 0] <= x_max), 1]
    y_max = max(float(np.max(in_view)) if len(in_view) else 0.0, p.A_refined) * 1.1
    dpg.set_axis_limits("x_axis_plot2", x_min, x_max)
    dpg.set_axis_limits("y_axis_plot2", -0.05 * y_max, y_max)
    # Release the limits on the next frame so the user can pan / zoom again
    dpg.split_frame()
    dpg.set_axis_limits_auto("x_axis_plot2")
    dpg.set_axis_limits_auto("y_axis_plot2")


def run_laplace_analysis_callback():
    laplace_covariance_analysis()
    update_peak_table(get_global_msdata_ref())


def run_advanced_statistical_analysis_callback():
    spectrum = get_global_msdata_ref()
    dpg.show_item("Fitting_indicator")
    dpg.hide_item("start_fitting_button")
    dpg.hide_item("fit_options_button")
    show_stop_button()
    dpg.hide_item("advanced_statistical_analysis_button")
    dpg.set_value(
        "Fitting_indicator_text",
        "Running bootstrap and randomisation, this might take a while...",
    )
    run_advanced_statistical_analysis()
    if dpg.get_value("stop_fitting_button"):
        dpg.set_value(
            "Fitting_indicator_text", "Error analysis stopped: previous errors kept"
        )
    update_peak_table(get_global_msdata_ref())
    dpg.hide_item("Fitting_indicator")
    dpg.show_item("start_fitting_button")
    dpg.show_item("fit_options_button")
    dpg.hide_item("stop_fitting_frame")
    dpg.show_item("advanced_statistical_analysis_button")
    if dpg.does_alias_exist("noise"):
        dpg.delete_item("noise")


def toggle_lorentzian_peak_model(sender, app_data):
    spectrum = get_global_msdata_ref()
    if app_data:
        spectrum.peak_model = "lorentzian"
    else:
        spectrum.peak_model = "gaussian"
