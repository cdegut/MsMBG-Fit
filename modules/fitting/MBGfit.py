from modules.data_structures import (
    FitQualityPeakMetrics,
    FitSummary,
    MSData,
    peak_params,
    get_global_msdata_ref,
)
import dearpygui.dearpygui as dpg
from modules.finding_callback import get_smoothing_window
from modules.fitting.fitting_quality import (
    FitQualityMetrics,
    FitQualityMetricsReduced,
    advanced_statistical_analysis,
    calculate_fit_quality_metrics,
    CONVERGENCE_WINDOW,
    ConvergenceMonitor,
    R2FlatnessMonitor,
    laplace_covariance_analysis,
)
from modules.math import combine_errors
from modules.fitting.peak_starting_points import update_peak_starting_points
import numpy as np
from modules.fitting.refiner import refine_iteration
from modules.utils import log
import time
from modules.rendercallback import RenderCallback, get_global_render_callback_ref
from modules.fitting.draw_MBG import show_MBG


def initial_peaks_parameters(spectrum: MSData, asymmetry=1.8) -> None | list[int]:
    # initial_params = []
    working_peak_list = []
    i = 0
    working_peaks: dict[int, peak_params] = {}

    reduce_width_var = dpg.get_value("use_reduced")

    if spectrum.peaks is None:
        log("No peaks are detected. Please run peak detection first")
        return None

    width_l = [spectrum.peaks[peak].width for peak in spectrum.peaks]
    std_width = np.std(width_l)
    med_width = np.median(width_l)

    for peak in spectrum.peaks:

        x0_guess = spectrum.peaks[peak].x0_init

        if spectrum.peaks[peak].do_not_fit:
            spectrum.peaks[peak].fitted = False
            continue

        if (
            x0_guess < spectrum.working_data[:, 0][0]
            or x0_guess > spectrum.working_data[:, 0][-1]
        ):
            log(f"Peak {i} is out of bounds. Skipping")
            i += 1
            continue

        width_init = spectrum.peaks[peak].width
        if reduce_width_var:
            width_init = med_width * 0.8 + 0.2 * width_init

        # Select working data within x0_guess ± width_init
        mask = (spectrum.working_data[:, 0] >= x0_guess - width_init * 2) & (
            spectrum.working_data[:, 0] <= x0_guess + width_init * 2
        )
        working_data_peak = spectrum.working_data[mask]
        if len(working_data_peak) > 1:
            sampling_rate = np.mean(np.diff(working_data_peak[:, 0]))
        else:
            sampling_rate = np.mean(np.diff(spectrum.working_data[:, 0]))

        sigma_L_guess = width_init * 0.7 / 2
        sigma_R_guess = sigma_L_guess * asymmetry
        #     [A_guess, x0_guess, sigma_L_guess, sigma_R_guess, sampling_rate]
        # )
        working_peak_list.append(peak)

        spectrum.peaks[peak].sigma_L_init = float(sigma_L_guess)
        spectrum.peaks[peak].sigma_R_init = float(sigma_R_guess)
        spectrum.peaks[peak].fitted = False
        spectrum.peaks[peak].sampling_rate = float(sampling_rate)

        i += 1

    if working_peak_list == []:
        log(
            "No peaks are within the data range. Please adjust the peak detection parameters"
        )
        return None

    return working_peak_list


def MBG_fit(
    render_callback,
    iterations=1000,
    theta_threshold=5e-5,
    use_filtered=True,
    use_gaussian=False,
):
    spectrum = get_global_msdata_ref()
    baseline_window = dpg.get_value("baseline_window")
    spectrum.correct_baseline(baseline_window)
    working_peak_list = initial_peaks_parameters(spectrum)
    render_callback.working_peak_list = working_peak_list
    if working_peak_list is None:
        return

    spectrum.laplace_integral_corr = None
    i = 0
    for peak in working_peak_list:
        spectrum.peaks[peak].A_refined = spectrum.peaks[peak].A_init
        spectrum.peaks[peak].x0_refined = spectrum.peaks[peak].x0_init
        spectrum.peaks[peak].sigma_L = spectrum.peaks[peak].sigma_L_init
        spectrum.peaks[peak].sigma_R = spectrum.peaks[peak].sigma_R_init
        spectrum.peaks[peak].se_sigma_L = -1
        spectrum.peaks[peak].se_sigma_R = -1
        spectrum.peaks[peak].se_A = -1
        spectrum.peaks[peak].se_x0 = -1
        spectrum.peaks[peak].se_integral = -1
        spectrum.peaks[peak].se_integral_bootstrap = -1.0
        spectrum.peaks[peak].se_integral_restart = -1.0
        spectrum.peaks[peak].se_x0_bootstrap = -1.0
        spectrum.peaks[peak].se_x0_restart = -1.0
        spectrum.peaks[peak].regression_fct = (0.0, 0.0)
        spectrum.peaks[peak].fit_quality = FitQualityPeakMetrics(0.0, 0.0, 1.0, 0.0)
        spectrum.peaks[peak].marked_bad = None  # automatic again: from the new fit's error
        spectrum.peaks[peak].integral = 0.0
        spectrum.peaks[peak].fitted = False
        spectrum.peaks[peak].laplace_se_integral = -1.0
        spectrum.peaks[peak].laplace_se_x0 = -1.0
        spectrum.peaks[peak].laplace_area_corr = 0.0
        spectrum.peaks[peak].laplace_area_peak = -1
        spectrum.peaks[peak].laplace_corr_left = 0.0
        spectrum.peaks[peak].laplace_corr_right = 0.0
        spectrum.peaks[peak].laplace_left_peak = -1
        spectrum.peaks[peak].laplace_right_peak = -1
        spectrum.peaks[peak].laplace_message = ""
        i += 1

    fit = refine_peak_parameters(
        working_peak_list,
        render_callback,
        iterations,
        theta_threshold,
        use_filtered=use_filtered,
        use_gaussian=use_gaussian,
    )
    if fit:
        log("Fitting done with no error")
    else:
        log("Error while fitting")
        return

    update_peak_starting_points(spectrum)


def refine_peak_parameters(
    working_peak_list,
    render_callback: RenderCallback,
    iterations=1000,
    theta_threshold=5e-5,
    use_filtered=True,
    use_gaussian=False,
):
    spectrum = render_callback.spectrum
    original_peaks: dict[int, peak_params] = {
        peak: spectrum.peaks[peak] for peak in working_peak_list
    }
    data_x = spectrum.working_data[:, 0]
    data_y = spectrum.baseline_corrected[:, 1]

    # Store quality metrics history
    quality_history = []
    iterations_list = [i for i in range(len(working_peak_list))]
    start = time.time()

    if use_filtered:
        window_length = get_smoothing_window()
        data_x = spectrum.baseline_corrected[:, 0]
        data_y = np.array(
            spectrum.get_filtered_data(window_length=window_length, baseline=True)
        )

    else:
        data_x = spectrum.baseline_corrected[:, 0]
        data_y = spectrum.baseline_corrected[:, 1]

    ## Iteration loop start here
    ##############################
    k = 0
    sigma_L_mean, sigma_R_mean, sigma_L_std, sigma_R_std = -1, -1, -1, -1
    # theta_threshold: largest allowed move per window, in Laplace error bars
    monitor = ConvergenceMonitor(
        spectrum,
        working_peak_list,
        spectrum.baseline_corrected[:, 0],
        spectrum.baseline_corrected[:, 1],
        threshold=theta_threshold,
    )
    delta_theta = float("nan")
    theta_converged = False
    metric_history = []
    oscillation_detected = False

    current_metric = 0.0
    r_squared = 0.0
    r_squared_convergence = dpg.get_value("fitting_r2")
    flatness = R2FlatnessMonitor(r2_flat_threshold())
    accept_only_improving = bool(dpg.get_value("accept_only_improving"))
    stop_reason = "max_iter"  # overwritten if another criterion ends the loop
    for k in range(iterations + 1):

        iteration_start = time.time()
        render_callback.execute()
        if dpg.get_value("stop_fitting_button") or render_callback.stop_fitting:
            log("Fitting stopped by user")
            dpg.set_item_label("stop_fitting_button", "Stopping...")
            stop_reason = "user"
            break

        quality_metrics: FitQualityMetricsReduced = calculate_fit_quality_metrics(
            data_x, data_y, spectrum, working_peak_list, rmse_only=True
        )
        if dpg.get_value("use_reduced"):
            sigma_L_mean = quality_metrics.sigma_L_mean
            sigma_R_mean = quality_metrics.sigma_R_mean
            sigma_L_std = quality_metrics.sigma_L_std
            sigma_R_std = quality_metrics.sigma_R_std

        widths = (sigma_L_mean, sigma_R_mean, sigma_L_std, sigma_R_std)
        quality_history.append(quality_metrics)
        residual = data_y - spectrum.calculate_mbg(data_x, fitting=True)
        show_MBG(spectrum, True)

        if dpg.get_value("show_residual_checkbox"):
            dpg.set_value("residual", [data_x.tolist(), residual.tolist()])

        # Use weighted RMSE for convergence check
        current_metric = quality_metrics.weighted_rmse
        r_squared = quality_metrics.r_squared
        metric_history.append(current_metric)

        if len(metric_history) >= 4:
            recent = metric_history[-4:]
            if (
                recent[0] < recent[1] > recent[2] < recent[3]
                or recent[0] > recent[1] < recent[2] > recent[3]
            ):
                oscillation_detected = True
                print("Oscillation detected in fitting metrics")

        # check for parameter convergence (every CONVERGENCE_WINDOW iterations)
        if monitor.update(k):
            delta_theta = monitor.last_move
            theta_converged = monitor.converged

        dpg.set_value(
            "Fitting_indicator_text",
            f"Iter {k}: wRMSE: {current_metric:.4f}, R²: {r_squared:.4f}, largest move: "
            + (
                f"{delta_theta:.2f} error bars / {CONVERGENCE_WINDOW} it."
                if np.isfinite(delta_theta)
                else "measuring..."
            )
            + (
                f", R² gain: {100 * flatness.last_gain:.3g} % / {flatness.window} it."
                if np.isfinite(flatness.last_gain)
                else ""
            )
            + f", iteration time: {time.time() - iteration_start:.2f}s",
        )
        alpha_convergence = 0.95
        # Check convergence using multiple criteria
        r2_converged = r_squared > r_squared_convergence
        flat_converged = flatness.update(r_squared)
        criteria_met = [
            name
            for name, met in (
                ("theta", theta_converged),
                ("r2", r2_converged),
                ("flat", flat_converged),
            )
            if met
        ]
        if criteria_met:
            stop_reason = "+".join(criteria_met)
            break

        if oscillation_detected:
            alpha_convergence = 0.4
        # elif k < 25:
        #     alpha_convergence = 0.95
        # elif k < 50:
        #     alpha_convergence = 0.7
        # elif k < 100:
        #     alpha_convergence = 0.5

        if len(metric_history) >= 5 and oscillation_detected:
            recent = metric_history[-5:]
            if all(recent[i] >= recent[i + 1] for i in range(4)):  # Monotonic decrease
                oscillation_detected = False

        # Shuffle iterations order for next pass
        iterations_list = np.random.permutation(
            iterations_list
        )  # do not use the same order every iteration
        for i in iterations_list:
            peak = working_peak_list[i]
            refine_iteration(
                peak=peak,
                data_x=data_x,
                data_y=data_y,
                spectrum=spectrum,
                original_peak_width=original_peaks[peak].width,
                force_gaussian=use_gaussian,
                widths=widths,
                alpha=alpha_convergence,
                accept_only_improving=accept_only_improving,
            )

    ##############################
    # End of iteration loop
    ##############################
    full_quality_metrics: FitQualityMetrics = calculate_fit_quality_metrics(
        data_x,
        data_y,
        spectrum,
        working_peak_list,
    )

    chi_squared = full_quality_metrics.chi_squared_reduced

    signal_to_noise = full_quality_metrics.signal_to_noise
    peaks_error = [
        full_quality_metrics.peak_quality[peak].relative_error
        for peak in working_peak_list
        if peak in full_quality_metrics.peak_quality
    ]
    r_squared = full_quality_metrics.r_squared

    time_taken = time.time() - start
    fit_summary = FitSummary(
        stop_reason=stop_reason,
        iterations_done=k,
        max_iterations=iterations,
        delta_theta=float(delta_theta),
        theta_threshold=theta_threshold,
        r_squared=float(r_squared),
        r_squared_threshold=r_squared_convergence,
        weighted_rmse=float(current_metric),
        chi_squared_reduced=float(chi_squared),
        signal_to_noise=float(signal_to_noise),
        aic=float(full_quality_metrics.aic),
        bic=float(full_quality_metrics.bic),
        residual_autocorr=float(full_quality_metrics.residual_autocorr),
        time_taken=time_taken,
        r2_flat_gain=flatness.last_gain,
        r2_flat_threshold=flatness.threshold,
    )
    render_callback.fit_summary = fit_summary
    log(
        f"{stop_reason_text(fit_summary)}: R²={r_squared:.4f}, X²r={chi_squared:.3f}, SNR={signal_to_noise:.1f}, Time: {time_taken:.2f}s"
    )

    median_error = np.median(peaks_error) if peaks_error else float("nan")
    dpg.set_value(
        "Fitting_indicator_text",
        f"{stop_reason_text(fit_summary)} after {k} iterations. wRMSE={current_metric:.4f}, R²={r_squared:.4f}, "
        f"X²r={chi_squared:.3f}, Median peak error={median_error:.4f}, Time: {time_taken:.2f}s",
    )
    render_callback.iterations_done = k

    for peak in working_peak_list:
        spectrum.peaks[peak].fitted = True
        spectrum.peaks[peak].fit_quality = full_quality_metrics.peak_quality.get(
            peak, FitQualityPeakMetrics(0.0, 0.0, 1.0, 0.0)
        )

    return True


CRITERION_TEXT = {
    "theta": "parameters stable",
    "r2": "R² above threshold",
    "flat": "R² no longer improving",
}


def r2_flat_threshold() -> float:
    """R² flatness setting of the UI, as a fraction (the UI shows %)."""
    return dpg.get_value("fitting_r2_flat") / 100.0


def stop_reason_text(fit_summary: FitSummary) -> str:
    reason = fit_summary.stop_reason
    if reason == "max_iter":
        return "Not converged: iteration limit reached"
    if reason == "user":
        return "Stopped by user"
    return "Converged: " + " and ".join(
        CRITERION_TEXT.get(name, name) for name in reason.split("+")
    )


# Error analysis: number of refits, and iteration cap of a bootstrap refit (it
# starts from the fitted solution); a restart refit is capped like the main fit
BOOTSTRAP_SAMPLES = 512
BOOTSTRAP_MAX_ITERATIONS = 500
# A bootstrap refit starts at the solution and can meet the stop criteria at once:
# it runs at least this, or this fraction of the main fit's iterations if larger
BOOTSTRAP_MIN_ITERATIONS = 10
BOOTSTRAP_MIN_FRACTION = 0.1
RESTART_SAMPLES = 16  # one round of refits on a 16-core machine


def run_advanced_statistical_analysis():
    # Calculate standard errors boot strap and refitting with perturbation
    spectrum = get_global_msdata_ref()
    render_callback = get_global_render_callback_ref()
    working_peak_list = render_callback.working_peak_list
    if not working_peak_list:
        # Fit loaded from a file: no fit ran in this session
        working_peak_list = [
            peak
            for peak in spectrum.peaks
            if spectrum.peaks[peak].fitted and not spectrum.peaks[peak].do_not_fit
        ]
        render_callback.working_peak_list = working_peak_list
    if not working_peak_list:
        log("No fitted peaks: run the fitting first")
        return
    data_x = spectrum.baseline_corrected[:, 0]
    data_y = spectrum.baseline_corrected[:, 1]
    # Refits stop with the same convergence test as the main fit
    convergence_threshold = dpg.get_value("theta_threshold_selector")

    # Laplace errors of the current fit, combined with the bootstrap below
    laplace_covariance_analysis()

    quality_metrics = calculate_fit_quality_metrics(
        data_x, data_y, spectrum, working_peak_list, rmse_only=True
    )

    error_bootstrap = advanced_statistical_analysis(
        spectrum=spectrum,
        working_peak_list=working_peak_list,
        data_x=data_x,
        data_y=data_y,
        render_callback=render_callback,
        method="bootstrap-parametric",
        macro_iteration=BOOTSTRAP_SAMPLES,
        micro_iteration=BOOTSTRAP_MAX_ITERATIONS,
        check_convergence="theta-gradient",
        wRMSE_threshold=quality_metrics.weighted_rmse * 1.05,
        theta_threshold=convergence_threshold,
        r2_threshold=(
            spectrum.fit_summary.r_squared
            if spectrum.fit_summary is not None
            else dpg.get_value("fitting_r2")
        ),
        r2_flat_threshold=r2_flat_threshold(),
        accept_only_improving=bool(dpg.get_value("accept_only_improving")),
        min_iterations=max(
            BOOTSTRAP_MIN_ITERATIONS,
            int(
                np.ceil(
                    BOOTSTRAP_MIN_FRACTION
                    * (
                        spectrum.fit_summary.iterations_done
                        if spectrum.fit_summary is not None
                        else 0
                    )
                )
            ),
        ),
    )

    if error_bootstrap is False:
        return

    errors_random_start = advanced_statistical_analysis(
        spectrum,
        working_peak_list,
        data_x,
        data_y,
        render_callback,
        macro_iteration=RESTART_SAMPLES,
        micro_iteration=dpg.get_value("fitting_iterations"),
        check_convergence="theta-gradient",
        wRMSE_threshold=quality_metrics.weighted_rmse * 1.1,
        theta_threshold=convergence_threshold,
        r2_threshold=(
            spectrum.fit_summary.r_squared
            if spectrum.fit_summary is not None
            else dpg.get_value("fitting_r2")
        ),
        r2_flat_threshold=r2_flat_threshold(),
        accept_only_improving=bool(dpg.get_value("accept_only_improving")),
        method="initial",
    )
    if errors_random_start is False and dpg.get_value("stop_fitting_button"):
        return  # stopped by the user: keep the previous errors

    # Store standard errors in peak parameters.
    # Final error = largest of bootstrap, Laplace and random restarts (see combine_errors).
    for peak in working_peak_list:
        p = spectrum.peaks[peak]
        err_bs = error_bootstrap.get(peak)
        err_rs = errors_random_start.get(peak) if errors_random_start else None
        if err_bs is None and err_rs is None:
            continue

        def component(errors, key):
            if errors is None or not np.isfinite(errors[key]):
                return -1.0
            return float(errors[key])

        p.se_integral_bootstrap = component(err_bs, "integral")
        p.se_integral_restart = component(err_rs, "integral")
        p.se_x0_bootstrap = component(err_bs, "x0")
        p.se_x0_restart = component(err_rs, "x0")
        laplace_integral = (
            p.laplace_se_integral * p.integral if p.laplace_se_integral > 0 else -1.0
        )

        p.se_integral = combine_errors(
            [p.se_integral_bootstrap, laplace_integral], p.se_integral_restart
        )
        p.se_x0 = combine_errors([p.se_x0_bootstrap, p.laplace_se_x0], p.se_x0_restart)
        for key in ("A", "sigma_L", "sigma_R", "base"):
            setattr(
                p,
                f"se_{key}",
                combine_errors([component(err_bs, key)], component(err_rs, key)),
            )


def update_peak_params(peak_list, popt, spectrum: MSData):
    i = 0
    for peak in peak_list:
        A_fit, x0_fit, sigma_L_fit, sigma_R_fit = popt[i * 4 : (i + 1) * 4]
        spectrum.peaks[peak].A_refined = A_fit
        spectrum.peaks[peak].x0_refined = x0_fit
        spectrum.peaks[peak].sigma_L = sigma_L_fit
        spectrum.peaks[peak].sigma_R = sigma_R_fit
        spectrum.peaks[peak].fitted = True
        i += 1
