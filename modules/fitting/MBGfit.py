from modules.data_structures import (
    FitQualityPeakMetrics,
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
)
from modules.math import check_theta_convergence
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
        spectrum.peaks[peak].regression_fct = (0.0, 0.0)
        spectrum.peaks[peak].fit_quality = FitQualityPeakMetrics(0.0, 0.0, 1.0, 0.0)
        spectrum.peaks[peak].integral = 0.0
        spectrum.peaks[peak].fitted = False
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
    theta_new = None
    theta_old, peak_list = spectrum.get_packed_parameters()
    delta_theta = 0
    theta_converged = False
    metric_history = []
    oscillation_detected = False

    current_metric = 0.0
    for k in range(iterations + 1):

        iteration_start = time.time()
        render_callback.execute()
        if dpg.get_value("stop_fitting_checkbox") or render_callback.stop_fitting:
            log("Fitting stopped by user")
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

        # check for parameter convergence
        if theta_new is not None:
            theta_converged, delta_theta = check_theta_convergence(
                theta_old, theta_new, tol_theta=theta_threshold
            )
            theta_old = theta_new

        dpg.set_value(
            "Fitting_indicator_text",
            f"Iter {k}: wRMSE: {current_metric:.4f}, R²: {r_squared:.4f}, Parameters change: {delta_theta:.4e}, iteration time: {time.time() - iteration_start:.2f}s",
        )
        alpha_convergence = 0.95
        r_squared_convergence = dpg.get_value("fitting_r2")
        # Check convergence using multiple criteria
        converged = theta_converged or r_squared > r_squared_convergence
        if converged:
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
            )
        theta_new, peak_list = spectrum.get_packed_parameters()

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
    log(
        f"Converged: R²={r_squared:.4f}, X²r={chi_squared:.3f}, SNR={signal_to_noise:.1f}, Time: {time_taken:.2f}s"
    )

    dpg.set_value(
        "Fitting_indicator_text",
        f"Converged after {k} iterations. wRMSE={current_metric:.4f}, R²={r_squared:.4f}, "
        f"X²r={chi_squared:.3f}, Median peak error={np.median(peaks_error):.4f}, Time: {time_taken:.2f}s",
    )
    render_callback.iterations_done = k
    render_callback.finishing_delta_theta = (
        float(delta_theta) if delta_theta is not 0 else 5e-5
    )

    for peak in working_peak_list:
        spectrum.peaks[peak].fitted = True
        spectrum.peaks[peak].fit_quality = full_quality_metrics.peak_quality.get(
            peak, {}
        )

    return True


def run_advanced_statistical_analysis():
    # Calculate standard errors boot strap and refitting with perturbation
    spectrum = get_global_msdata_ref()
    render_callback = get_global_render_callback_ref()
    working_peak_list = render_callback.working_peak_list
    data_x = spectrum.working_data[:, 0]
    data_y = spectrum.baseline_corrected[:, 1]
    k = render_callback.iterations_done

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
        macro_iteration=512,
        micro_iteration=30,
        check_convergence="theta-gradient",
        wRMSE_threshold=quality_metrics.weighted_rmse * 1.05,
        theta_threshold=render_callback.finishing_delta_theta * 2,
    )

    if error_bootstrap is False:
        return

    errors_random_start = advanced_statistical_analysis(
        spectrum,
        working_peak_list,
        data_x,
        data_y,
        render_callback,
        macro_iteration=64,
        micro_iteration=int(k * 1.5),
        check_convergence="theta-gradient",
        wRMSE_threshold=quality_metrics.weighted_rmse * 1.1,
        theta_threshold=render_callback.finishing_delta_theta * 2,
        method="initial",
    )

    # Store standard errors in peak parameters
    for peak in working_peak_list:
        if errors_random_start is False:
            err_rs = None
        else:
            err_rs = errors_random_start[peak]
        err_bs = error_bootstrap[peak]
        if err_rs is not None and err_bs is not None:
            spectrum.peaks[peak].se_A = max(err_rs["A"], err_bs["A"])
            spectrum.peaks[peak].se_x0 = max(err_rs["x0"], err_bs["x0"])
            spectrum.peaks[peak].se_sigma_L = max(err_rs["sigma_L"], err_bs["sigma_L"])
            spectrum.peaks[peak].se_sigma_R = max(err_rs["sigma_R"], err_bs["sigma_R"])
            spectrum.peaks[peak].se_integral = max(
                err_rs["integral"], err_bs["integral"]
            )
            spectrum.peaks[peak].se_base = max(err_rs["base"], err_bs["base"])
        elif err_rs is not None:
            spectrum.peaks[peak].se_A = err_rs["A"]
            spectrum.peaks[peak].se_x0 = err_rs["x0"]
            spectrum.peaks[peak].se_sigma_L = err_rs["sigma_L"]
            spectrum.peaks[peak].se_sigma_R = err_rs["sigma_R"]
            spectrum.peaks[peak].se_integral = err_rs["integral"]
            spectrum.peaks[peak].se_base = err_rs["base"]
        elif err_bs is not None:
            spectrum.peaks[peak].se_A = err_bs["A"]
            spectrum.peaks[peak].se_x0 = err_bs["x0"]
            spectrum.peaks[peak].se_sigma_L = err_bs["sigma_L"]
            spectrum.peaks[peak].se_sigma_R = err_bs["sigma_R"]
            spectrum.peaks[peak].se_integral = err_bs["integral"]
            spectrum.peaks[peak].se_base = err_bs["base"]
        else:
            continue


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
