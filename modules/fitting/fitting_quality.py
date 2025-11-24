from calendar import c
from dataclasses import dataclass
import os
from typing import Dict, Literal, overload
import numpy as np
from sklearn import base
from modules.data_structures import (
    FitQualityPeakMetrics,
    MSData,
    get_global_msdata_ref,
    peak_params,
)
from modules.fitting.peak_starting_points import update_peak_starting_points
from modules.math import (
    bi_Lorentzian_integral,
    bi_gaussian_integral,
    multi_bi_gaussian,
    multi_bi_gaussian_using_I,
    standard_error,
    check_theta_convergence,
)
from modules.rendercallback import RenderCallback
from modules.utils import log
import dearpygui.dearpygui as dpg
from modules.fitting.refiner import refine_iteration
from copy import deepcopy
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock


@dataclass
class FitQualityMetrics:
    rmse: float
    weighted_rmse: float
    r_squared: float
    chi_squared_reduced: float
    noise_variance: float
    noise_std: float
    signal_to_noise: float
    aic: float
    bic: float
    peak_quality: dict
    residual_autocorr: float


@dataclass
class FitQualityMetricsReduced:
    rmse: float
    weighted_rmse: float
    r_squared: float
    weights: np.ndarray
    sigma_L_mean: float = 0.0
    sigma_R_mean: float = 0.0
    sigma_L_std: float = 0.0
    sigma_R_std: float = 0.0


@overload
def calculate_fit_quality_metrics(
    data_x,
    data_y,
    spectrum,
    working_peak_list,
    weights=None,
    *,
    rmse_only: Literal[True],
) -> FitQualityMetricsReduced: ...


@overload
def calculate_fit_quality_metrics(
    data_x,
    data_y,
    spectrum,
    working_peak_list,
    weights=None,
    *,
    rmse_only: Literal[False] = False,
) -> FitQualityMetrics: ...


def calculate_fit_quality_metrics(
    data_x,
    data_y,
    spectrum,
    working_peak_list,
    weights=None,
    *,
    rmse_only=False,
):  # -> FitQualityMetricsReduced | dict[str, Any]:
    """Calculate comprehensive fit quality metrics"""
    residual = data_y - spectrum.calculate_mbg(data_x, fitting=True)

    sigma_L_list = []
    sigma_R_list = []
    # 1. Weighted RMSE (higher weight for peak regions)
    if weights is None:
        weights = np.ones_like(data_y)
        for peak in working_peak_list:
            x0 = spectrum.peaks[peak].x0_refined
            sigma_L = spectrum.peaks[peak].sigma_L
            sigma_R = spectrum.peaks[peak].sigma_R
            # Create Gaussian weight centered at peak
            peak_mask = (data_x >= x0 - 3 * sigma_L) & (data_x <= x0 + 3 * sigma_R)
            weights[peak_mask] *= 5.0  # 5x weight for peak regions
            sigma_L_list.append(sigma_L)
            sigma_R_list.append(sigma_R)

    weighted_rmse = np.sqrt(np.average(np.square(residual), weights=weights))

    # 2. R-squared (coefficient of determination)
    ss_res = np.sum(np.square(residual))
    ss_tot = np.sum(np.square(data_y - np.mean(data_y)))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    if rmse_only:
        return FitQualityMetricsReduced(
            rmse=np.sqrt(np.mean(np.square(residual))),
            weighted_rmse=weighted_rmse,
            r_squared=r_squared,
            weights=weights,
            sigma_L_mean=float(np.mean(sigma_L_list) if sigma_L_list else 0.0),
            sigma_R_mean=float(np.mean(sigma_R_list) if sigma_R_list else 0.0),
            sigma_L_std=float(np.std(sigma_L_list) if sigma_L_list else 0.0),
            sigma_R_std=float(np.std(sigma_R_list) if sigma_R_list else 0.0),
        )

    # 3. Chi-squared for noisy data
    degrees_of_freedom = len(data_y) - (len(working_peak_list) * 4)
    # Method 1: Estimate noise from baseline regions
    noise_mask = np.ones_like(data_y, dtype=bool)
    for peak in working_peak_list:
        if spectrum.peaks[peak].fitted:
            x0 = spectrum.peaks[peak].x0_refined
            sigma_L = spectrum.peaks[peak].sigma_L
            sigma_R = spectrum.peaks[peak].sigma_R
            # Exclude peak regions from noise calculation
            peak_mask = (data_x >= x0 - 4 * sigma_L) & (data_x <= x0 + 4 * sigma_R)
            noise_mask &= ~peak_mask

    # Estimate noise variance from baseline regions
    if np.sum(noise_mask) > 10:  # Need enough baseline points
        baseline_residual = residual[noise_mask]
        noise_variance = np.var(baseline_residual)
    else:
        # Fallback: robust estimate from all residuals using median absolute deviation
        mad = np.median(np.abs(residual - np.median(residual)))
        noise_variance = (1.4826 * mad) ** 2  # Convert MAD to variance estimate

    # Ensure minimum noise level
    min_noise = np.var(data_y) * 0.001  # At least 0.1% of signal variance
    noise_variance = max(noise_variance, min_noise)

    if degrees_of_freedom > 0:
        chi_squared_reduced = ss_res / (degrees_of_freedom * noise_variance)
    else:
        chi_squared_reduced = np.inf

    # 4. Peak-specific metrics
    peak_quality = {}
    for peak in working_peak_list:
        x0 = spectrum.peaks[peak].x0_refined
        sigma_L = spectrum.peaks[peak].sigma_L
        sigma_R = spectrum.peaks[peak].sigma_R

        # Peak region mask
        peak_mask = (data_x >= x0 - 3 * sigma_L) & (data_x <= x0 + 3 * sigma_R)
        if np.any(peak_mask):
            peak_residual = residual[peak_mask]
            peak_data = data_y[peak_mask]

            # Signal-to-noise ratio at peak
            peak_height = spectrum.peaks[peak].A_refined
            noise_level = np.std(peak_residual)
            snr = peak_height / noise_level if noise_level > 0 else np.inf

            # Peak RMSE
            peak_rmse = np.sqrt(np.mean(np.square(peak_residual)))

            # Per-peak R-squared
            ss_res_peak = np.sum(np.square(peak_residual))
            ss_tot_peak = np.sum(np.square(peak_data - np.mean(peak_data)))
            peak_r_squared = 1 - (ss_res_peak / ss_tot_peak) if ss_tot_peak > 0 else 0

            peak_quality[peak] = FitQualityPeakMetrics(
                snr=snr,
                peak_rmse=peak_rmse,
                relative_error=(peak_rmse / peak_height if peak_height > 0 else np.inf),
                r_squared=peak_r_squared,
            )

    # 5. Akaike Information Criterion (AIC)
    n = len(data_y)
    k = len(working_peak_list) * 4  # number of parameters
    if n > 0 and ss_res > 0:
        aic = n * np.log(ss_res / n) + 2 * k
    else:
        aic = np.inf

    # 6. Bayesian Information Criterion (BIC)
    if n > 0 and ss_res > 0:
        bic = n * np.log(ss_res / n) + k * np.log(n)
    else:
        bic = np.inf

    return FitQualityMetrics(
        rmse=np.sqrt(np.mean(np.square(residual))),
        weighted_rmse=weighted_rmse,
        r_squared=r_squared,
        chi_squared_reduced=chi_squared_reduced,
        noise_variance=float(noise_variance),
        noise_std=np.sqrt(noise_variance),
        signal_to_noise=np.mean(np.abs(data_y)) / np.sqrt(noise_variance),
        aic=aic,
        bic=bic,
        peak_quality=peak_quality,
        residual_autocorr=(
            np.corrcoef(residual[:-1], residual[1:])[0, 1] if len(residual) > 1 else 0
        ),
    )


@dataclass
class PerturbationFitErrorMetrics:
    A: list[float]
    x0: list[float]
    sigma_L: list[float]
    sigma_R: list[float]
    integral: list[float]
    base: list[float]


def advanced_statistical_analysis(
    spectrum: MSData,
    working_peak_list: list[int],
    data_x: np.ndarray,
    data_y: np.ndarray,
    render_callback: RenderCallback,
    macro_iteration=100,
    micro_iteration=10,
    method: Literal[
        "bootstrap-parametric", "bootstrap-residual", "initial"
    ] = "bootstrap-parametric",
    check_convergence: Literal["wRMSE", "theta-gradient", "both", False] = False,
    wRMSE_threshold: float = 100.0,
    theta_threshold: float = 1e-4,
) -> dict | Literal[False]:
    """Perform advanced statistical analysis using bootstrap and random start methods."""

    # Compute fitted values and residuals
    y_fitted = spectrum.calculate_mbg(data_x, fitting=True)
    residuals = data_y - y_fitted
    mad = np.median(np.abs(residuals - np.median(residuals)))
    sigma_hat = 1.4826 * mad

    initial_peaks: dict[int, peak_params] = {}
    for peak in working_peak_list:
        initial_peaks[peak] = peak_params(
            A_refined=spectrum.peaks[peak].A_refined,
            x0_refined=spectrum.peaks[peak].x0_refined,
            sigma_L=spectrum.peaks[peak].sigma_L,
            sigma_R=spectrum.peaks[peak].sigma_R,
        )

    perturbation_results: dict[int, PerturbationFitErrorMetrics] = {
        peak: PerturbationFitErrorMetrics(
            A=[], x0=[], sigma_L=[], sigma_R=[], integral=[], base=[]
        )
        for peak in working_peak_list
    }

    log(f"Starting bootstrap with {macro_iteration} resamples...")
    successful_fits = 0

    task_pool = []
    # Thread-safe counters
    cpu_count = os.cpu_count() or 2
    completed_lock = Lock()
    completed_tasks = {"count": 0, "successful": 0, "rejected": 0}
    BATCH_SIZE = min(cpu_count * 2, 50)  # Process 4x CPU cores or max 50 at once
    num_batches = (macro_iteration + BATCH_SIZE - 1) // BATCH_SIZE

    avg_iterations = None
    batch_iterations = []
    rescale = 1.0

    if method == "bootstrap-parametric" or method == "bootstrap-residual":
        it = 0
        for test in range(0, 10):
            dpg.set_value(
                "Fitting_indicator_sub_text",
                (
                    f"Running initial rescale tests {test}/10 "
                    + (f"last it: {it} rescale: {rescale:.2f}" if it else "")
                ),
            )
            if dpg.does_alias_exist("noise"):
                dpg.delete_item("noise")

            noise = np.random.normal(
                0, sigma_hat, size=len(spectrum.working_data[:, 0])
            )

            noise = noise + np.median(data_y)
            dpg.add_line_series(
                spectrum.working_data[:, 0].tolist(),
                noise.tolist(),
                parent="y_axis_plot2",
                tag=f"noise",
            )

            mini_batch = []
            for n in range(0, 4):
                test_task = _make_bootstrap_task(
                    spectrum=spectrum,
                    working_peak_list=working_peak_list,
                    data_x=data_x,
                    method=method,
                    b=test,
                    y_fitted=y_fitted,
                    wRMSE_threshold=wRMSE_threshold,
                    residuals=residuals,
                    sigma_hat=float(sigma_hat),
                    rescale=rescale,
                    micro_iteration=25,
                    check_convergence=check_convergence,
                    theta_threshold=theta_threshold,
                )
                _, _, it = execute_quick_fit(test_task)
                mini_batch.append(it)
                print(f"Initial rescale test {test} iteration {n}: {it} iterations")
            it = int(np.mean(mini_batch))

            if 5 > it < 10:
                dpg.set_value(
                    "Fitting_indicator_sub_text",
                    (
                        f"Running initial rescale tests {test}/10 "
                        + (f"last it: {it} rescale: {rescale:.2f} exiting test now")
                    ),
                )
                print("Exiting initial rescale tests early")
                break
            if it > 10:
                rescale = rescale * 0.75
            elif it > 24:
                rescale = rescale * 0.5
            elif it < 5:
                rescale = rescale * 1.25
            elif it <= 2:
                rescale = rescale * 2.0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, macro_iteration)

        if batch_iterations != []:
            avg_iterations = int(np.mean(batch_iterations))
            batch_iterations = []

        if avg_iterations is not None:
            if avg_iterations > 15:
                rescale = rescale * 0.75
            elif avg_iterations > 25:
                rescale = rescale * 0.5
            elif avg_iterations < 5:
                rescale = rescale * 1.25
            elif avg_iterations <= 2:
                rescale = rescale * 2.0

        if render_callback:
            dpg.set_value(
                "Fitting_indicator_sub_text",
                (
                    f"Running {start_idx} to {end_idx}"
                    + f" average iter last batch: {avg_iterations} rescale: {rescale:.2f}"
                    if avg_iterations and method != "initial"
                    else ""
                ),
            )

            if method == "bootstrap-parametric":

                if dpg.does_alias_exist("noise"):
                    dpg.delete_item("noise")

                noise = np.random.normal(
                    0, sigma_hat, size=len(spectrum.working_data[:, 0])
                )

                noise = noise + y_fitted
                dpg.add_line_series(
                    spectrum.working_data[:, 0].tolist(),
                    noise.tolist(),
                    parent="y_axis_plot2",
                    tag=f"noise",
                )

        task_pool = []
        for b in range(start_idx, end_idx):
            if method == "bootstrap-parametric" or method == "bootstrap-residual":
                task = _make_bootstrap_task(
                    spectrum=spectrum,
                    working_peak_list=working_peak_list,
                    data_x=data_x,
                    method=method,
                    b=b,
                    y_fitted=y_fitted,
                    wRMSE_threshold=wRMSE_threshold,
                    residuals=residuals,
                    sigma_hat=float(sigma_hat),
                    rescale=rescale,
                    micro_iteration=micro_iteration,
                    check_convergence=check_convergence,
                    theta_threshold=theta_threshold,
                )

            elif method == "initial":
                task = _make_initial_refit_task(
                    spectrum=spectrum,
                    working_peak_list=working_peak_list,
                    data_x=data_x,
                    data_y=data_y,
                    b=b,
                    micro_iteration=micro_iteration,
                    check_convergence=check_convergence,
                    wRMSE_threshold=wRMSE_threshold,
                    theta_threshold=theta_threshold,
                )

            task_pool.append(task)

        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = {
                executor.submit(execute_quick_fit, task): task for task in task_pool
            }

            for future in as_completed(futures):
                task = futures[future]
                fitted_peaks, converged, iteration = future.result()

                # Update progress (thread-safe)
                with completed_lock:
                    completed_tasks["count"] += 1
                    if converged:
                        completed_tasks["successful"] += 1
                    batch_iterations.append((iteration))

                    # Update GUI
                    if render_callback:
                        render_callback.execute()
                        if dpg.get_value("stop_fitting_checkbox"):
                            log("Fitting stopped by user.")
                            dpg.set_value("Fitting_indicator_text", "Stopping ...")
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            executor.shutdown(wait=False, cancel_futures=True)

                            return False

                        if method == "initial":
                            dpg.set_value(
                                "Fitting_indicator_text",
                                f"Refit with randomisation: {completed_tasks['count']}/{macro_iteration} "
                                f"({completed_tasks['successful']} converged, {completed_tasks['rejected']} rejected)",
                            )
                        else:
                            dpg.set_value(
                                "Fitting_indicator_text",
                                f"Bootstrap: {completed_tasks['count']}/{macro_iteration} "
                                f"({completed_tasks['successful']} converged, {completed_tasks['rejected']} rejected)",
                            )

                    if fitted_peaks == {}:
                        completed_tasks["rejected"] += 1
                        continue  # Skip failed fits
                    for peak in fitted_peaks:
                        perturbation_results[peak].A.append(
                            fitted_peaks[peak].A_refined
                        )
                        perturbation_results[peak].x0.append(
                            fitted_peaks[peak].x0_refined
                        )
                        perturbation_results[peak].sigma_L.append(
                            fitted_peaks[peak].sigma_L
                        )
                        perturbation_results[peak].sigma_R.append(
                            fitted_peaks[peak].sigma_R
                        )
                        perturbation_results[peak].integral.append(
                            fitted_peaks[peak].integral
                        )
                        perturbation_results[peak].base.append(
                            -fitted_peaks[peak].regression_fct[1]
                            / fitted_peaks[peak].regression_fct[0]
                        )

    print("integrals:", perturbation_results[working_peak_list[0]].integral)
    # Compute standard errors from bootstrap distribution
    final_result = {}
    for peak in working_peak_list:
        if len(perturbation_results[peak].A) < macro_iteration * 0.5:
            log(
                f"Warning: Peak {peak} had only {len(perturbation_results[peak].A)} successful fits"
            )
            final_result[peak] = None
            continue

        final_result[peak] = {
            "A": standard_error((perturbation_results[peak].A)),
            "x0": standard_error((perturbation_results[peak].x0)),
            "sigma_L": standard_error((perturbation_results[peak].sigma_L)),
            "sigma_R": standard_error((perturbation_results[peak].sigma_R)),
            "integral": standard_error((perturbation_results[peak].integral)),
            "n_samples": len(perturbation_results[peak].A),
            "base": standard_error((perturbation_results[peak].base)),
        }
    if dpg.does_alias_exist("noise"):
        dpg.delete_item("noise")

    log(f"Perturbation complete: {successful_fits}/{macro_iteration} successful fits")

    return final_result


@dataclass
class QuickFitInterface:
    spectrum: MSData
    working_peaks: Dict[int, peak_params]
    data_x: np.ndarray
    data_y: np.ndarray
    n_iterations: int
    check_convergence: Literal["wRMSE", "theta-gradient", "both", False]
    theta_threshold: float
    wRMSE_threshold: float
    width_regularization: bool
    it_index: int


def execute_quick_fit(task: QuickFitInterface):
    return quick_fit_model(
        task.spectrum,
        task.working_peaks,
        task.data_x,
        task.data_y,
        task.n_iterations,
        task.check_convergence,
        task.theta_threshold,
        task.wRMSE_threshold,
        task.width_regularization,
        task.it_index,
    )


def quick_fit_model(
    spectrum: MSData,
    working_peaks: Dict[int, peak_params],
    data_x: np.ndarray,
    data_y: np.ndarray,
    n_iterations=10,
    check_convergence: Literal["wRMSE", "theta-gradient", "both", False] = False,
    theta_threshold: float = 1e-4,
    wRMSE_threshold: float = 1e-4,
    width_regularization: bool = True,
    it_index: int = 0,
) -> tuple[Dict[int, peak_params], bool, int]:
    iteration = 0
    try:
        converged = False
        # Create temporary spectrum copy to avoid modifying the original

        for peak in working_peaks:
            spectrum.peaks[peak].A_refined = working_peaks[peak].A_refined
            spectrum.peaks[peak].x0_refined = working_peaks[peak].x0_refined
            spectrum.peaks[peak].sigma_L = working_peaks[peak].sigma_L
            spectrum.peaks[peak].sigma_R = working_peaks[peak].sigma_R
            spectrum.peaks[peak].width = (
                working_peaks[peak].sigma_L + working_peaks[peak].sigma_R
            )

        widths = (-1, -1, -1, -1)
        quality_metrics: Optional[FitQualityMetricsReduced] = None
        # Run a quick refinement (fewer iterations for speed)
        for iteration in range(n_iterations):
            old_theta, peak_list = spectrum.get_packed_parameters()
            random_list = np.random.permutation(list(working_peaks.keys()))
            if width_regularization and quality_metrics:
                widths = (
                    quality_metrics.sigma_L_mean,
                    quality_metrics.sigma_R_mean,
                    quality_metrics.sigma_L_std,
                    quality_metrics.sigma_R_std,
                )

            for peak in random_list:
                refine_iteration(
                    peak=peak,
                    data_x=data_x,
                    data_y=data_y,
                    spectrum=spectrum,
                    original_peak_width=spectrum.peaks[peak].width,
                    force_gaussian=False,
                    widths=widths,
                )

            quality_metrics = calculate_fit_quality_metrics(
                data_x,
                data_y,
                spectrum,
                list(working_peaks.keys()),
                rmse_only=True,
            )

            if check_convergence:
                rmse_converged = False
                theta_converged = False
                if check_convergence == "wRMSE" or check_convergence == "both":
                    rmse_converged = quality_metrics.weighted_rmse < wRMSE_threshold

                if check_convergence == "theta-gradient" or check_convergence == "both":
                    new_theta, peak_list = spectrum.get_packed_parameters()
                    theta_converged, delta_theta = check_theta_convergence(
                        old_theta,
                        new_theta,
                        tol_theta=theta_threshold,
                    )

                if check_convergence == "both":
                    converged = bool(rmse_converged and theta_converged)
                elif check_convergence == "wRMSE":
                    converged = rmse_converged
                elif check_convergence == "theta-gradient":
                    converged = bool(theta_converged)
                if converged:
                    break

        if check_convergence and not converged:
            full_quality_metrics = calculate_fit_quality_metrics(
                data_x,
                data_y,
                spectrum,
                list(working_peaks.keys()),
            )

            print(
                f"Quick fit did not converge in {n_iterations} iterations \n",
                "chi squared:",
                full_quality_metrics.chi_squared_reduced,
                "\n",
                "R²:",
                full_quality_metrics.r_squared,
                "\n",
                "wRMSE:",
                full_quality_metrics.weighted_rmse,
                f"/ {wRMSE_threshold} \n",
                "RMSE:",
                full_quality_metrics.rmse,
                "\n",
            )
            if (
                full_quality_metrics.chi_squared_reduced > 10.0
                or full_quality_metrics.r_squared < 0.8
                or full_quality_metrics.weighted_rmse > wRMSE_threshold * 4
            ):
                print("Very poor fit detected, rejecting results.")
                return {}, False, iteration

        # Store the fitted parameters
        update_peak_starting_points(spectrum)
        for peak in working_peaks:
            A = spectrum.peaks[peak].A_refined
            x0 = spectrum.peaks[peak].x0_refined
            sigma_L = spectrum.peaks[peak].sigma_L
            sigma_R = spectrum.peaks[peak].sigma_R

            # Sanity check: reject obviously bad fits
            if (
                A > 0
                and sigma_L > 0
                and sigma_R > 0
                and not np.isnan(A)
                and not np.isnan(x0)
                and not np.isnan(sigma_L)
                and not np.isnan(sigma_R)
            ):
                working_peaks[peak].A_refined = A
                working_peaks[peak].x0_refined = x0
                working_peaks[peak].sigma_L = sigma_L
                working_peaks[peak].sigma_R = sigma_R
                working_peaks[peak].regression_fct = spectrum.peaks[peak].regression_fct
                working_peaks[peak].integral = spectrum.peaks[peak].integral

        return working_peaks, converged, iteration

    except Exception as e:
        print(f"Quick fit iteration {it_index} failed: {e}")
        return {}, False, iteration


def pack_parameters(peaks_dict: dict[int, peak_params]) -> np.ndarray:
    """Flatten all peak parameters into a single 1D array."""
    theta = []
    for pid in sorted(peaks_dict.keys()):
        p = peaks_dict[pid]
        theta.extend([p.A_refined, p.x0_refined, p.sigma_L, p.sigma_R])
    return np.array(theta, dtype=float)


def _make_bootstrap_task(
    method: Literal["bootstrap-parametric", "bootstrap-residual"],
    b: int,
    spectrum: MSData,
    working_peak_list: list[int],
    data_x: np.ndarray,
    rescale: float,
    residuals: np.ndarray,
    y_fitted: np.ndarray,
    sigma_hat: float,
    micro_iteration: int,
    check_convergence: Literal["wRMSE", "theta-gradient", "both", False],
    theta_threshold: float,
    wRMSE_threshold: float,
):
    working_peaks: Dict[int, peak_params] = {}
    if method == "bootstrap-residual" or method == "bootstrap-parametric":
        noise_scale_A = 0.01 * rescale
        noise_scale_X0 = 0.02 * rescale
        noise_scale_w = 0.02 * rescale

        if method == "bootstrap-parametric":
            noise = np.random.normal(0, sigma_hat, size=len(residuals))
            task_data_y = y_fitted + noise

        else:  # residual bootstrap
            bootstrap_indices = np.random.choice(
                len(residuals), size=len(residuals), replace=True
            )
            resampled_residuals = residuals[bootstrap_indices]
            task_data_y = y_fitted + resampled_residuals

        for peak in working_peak_list:
            search_width = spectrum.peaks[peak].sigma_L + spectrum.peaks[peak].sigma_R
            working_peaks[peak] = peak_params(
                A_refined=spectrum.peaks[peak].A_refined
                * (1 + np.random.randn() * noise_scale_A),
                x0_refined=spectrum.peaks[peak].x0_refined
                + (np.random.randn() * search_width * noise_scale_X0),
                sigma_L=spectrum.peaks[peak].sigma_L
                * (1 + np.random.randn() * noise_scale_w),
                sigma_R=spectrum.peaks[peak].sigma_R
                * (1 + np.random.randn() * noise_scale_w),
            )
        task_spectrum = deepcopy(spectrum)

        task = QuickFitInterface(
            spectrum=task_spectrum,
            working_peaks=working_peaks,
            data_x=data_x.copy(),
            data_y=task_data_y,
            n_iterations=micro_iteration,
            check_convergence=check_convergence,
            theta_threshold=theta_threshold,
            wRMSE_threshold=wRMSE_threshold,
            width_regularization=True,
            it_index=b,
        )

        return task


def _make_initial_refit_task(
    data_y: np.ndarray,
    b: int,
    spectrum: MSData,
    working_peak_list: list[int],
    data_x: np.ndarray,
    micro_iteration: int,
    check_convergence: Literal["wRMSE", "theta-gradient", "both", False],
    theta_threshold: float,
    wRMSE_threshold: float,
):
    working_peaks: Dict[int, peak_params] = {}
    noise_scale_A = 0.2
    noise_scale_X0 = 0.1
    noise_scale_w = 0.2  # this is the critical value

    task_data_y = data_y.copy()
    task_spectrum = deepcopy(spectrum)

    for peak in working_peak_list:
        search_width = spectrum.peaks[peak].sigma_L + spectrum.peaks[peak].sigma_R
        noise_scale_X0_applied = (
            noise_scale_X0 if peak < 500 else noise_scale_X0 * 2.0
        )  # be harder on user added peaks
        working_peaks[peak] = peak_params(
            A_refined=spectrum.peaks[peak].A_init
            * (1 + np.random.randn() * noise_scale_A),
            x0_refined=spectrum.peaks[peak].x0_init
            + (np.random.randn() * search_width * noise_scale_X0_applied),
            sigma_L=spectrum.peaks[peak].sigma_L_init
            * (1 + np.random.randn() * noise_scale_w),
            sigma_R=spectrum.peaks[peak].sigma_R_init
            * (1 + np.random.randn() * noise_scale_w),
        )
    task = QuickFitInterface(
        spectrum=task_spectrum,
        working_peaks=working_peaks,
        data_x=data_x.copy(),
        data_y=task_data_y,
        n_iterations=micro_iteration,
        check_convergence=check_convergence,
        theta_threshold=theta_threshold,
        wRMSE_threshold=wRMSE_threshold,
        width_regularization=True,
        it_index=b,
    )

    return task


def laplace_covariance_analysis(log_params=True):
    spectrum: MSData = get_global_msdata_ref()
    theta_hat, peak_list = spectrum.get_packed_parameters(integral=True, ordered=True)
    theta_hat = np.asarray(theta_hat)
    x = spectrum.working_data[:, 0]
    y = spectrum.working_data[:, 1]
    results = rolling_three_peak_quality(
        model_func=multi_bi_gaussian_using_I,
        theta_hat=theta_hat,
        x=x,
        y=y,
        n_peaks=len(peak_list),
    )
    for i, result in enumerate(results):
        print(f"Peak {peak_list[i]} \n", f"Score: {result["score"]}")


def _peak_quality_score_from_J_local(J_local, rcond=1e-8, max_log_cond=20.0):
    """
    Compute 0..100 score for a local Jacobian J_local (N x n_params_local).
    Returns dict with components and message.
    """
    # SVD on J_local
    if J_local.size == 0:
        return {
            "score": 0.0,
            "rank_frac": 0.0,
            "kept": 0,
            "cond": np.inf,
            "cond_score": 0.0,
            "max_abs_corr": 1.0,
            "corr_score": 0.0,
            "message": "no data",
            "singular_values": np.array([]),
        }

    U, svals, VT = np.linalg.svd(J_local, full_matrices=False)
    svals = np.array(svals)
    cond = float(svals[0] / (svals[-1] + 1e-30))

    # effective kept singular count by rcond threshold
    cutoff = svals[0] * rcond
    kept = int(np.sum(svals > cutoff))

    # rank fraction normalized to 4 (we score per-peak later but pass generic)
    # if n_params_local>4, we map kept-> fraction of 4 for the peak metric step
    rank_frac = float(min(kept, 4)) / 4.0

    # conditioning component
    logc = np.log10(cond + 1e-30)
    cond_score = 1.0 - np.clip(logc / max_log_cond, 0.0, 1.0)

    # correlation component: largest absolute off-diagonal correlation between columns
    if J_local.shape[0] > 1 and J_local.shape[1] > 1:
        C = np.corrcoef(J_local.T)
        abs_corr = np.abs(C - np.eye(C.shape[0]))
        max_abs_corr = float(np.max(abs_corr))
    else:
        max_abs_corr = 0.0
    corr_score = 1.0 - np.clip(max_abs_corr, 0.0, 1.0)

    # weights (same as before)
    w_rank = 0.50
    w_cond = 0.30
    w_corr = 0.20
    combined = w_rank * rank_frac + w_cond * cond_score + w_corr * corr_score
    score = float(np.clip(combined, 0.0, 1.0) * 100.0)

    # message
    reasons = []
    if kept < 4:
        reasons.append(f"low effective rank ({kept})")
    if cond_score < 0.5:
        reasons.append(f"poor conditioning (cond≈{cond:.2e})")
    if max_abs_corr > 0.8:
        reasons.append(f"strong parameter correlation (corr≈{max_abs_corr:.2f})")
    message = " ; ".join(reasons) if reasons else "well constrained"

    return {
        "score": score,
        "rank_frac": rank_frac,
        "kept": kept,
        "cond": cond,
        "cond_score": cond_score,
        "max_abs_corr": max_abs_corr,
        "corr_score": corr_score,
        "message": message,
        "singular_values": svals,
    }


def rolling_three_peak_quality(
    model_func,
    theta_hat,
    x,
    y,
    n_peaks,
    dlog=1e-3,
    rcond=1e-8,
    window_factor=1.5,
):
    """
    Compute a per-peak quality score using a 3-peak rolling window.
    - model_func: f(x, *theta)
    - theta_hat: full packed parameter vector (len = 4*n_peaks)
    - x, y: data arrays
    - n_peaks: number of peaks
    Returns: list of dicts length n_peaks with entries:
      {
        'score': 0..100,
        'message': str,
        'stderr_I': float,
        'I': float,
        'Cov_local': 4x4 marginal covariance for central peak,
        'block_svals': array of SVD singular values for block,
        'block_kept': int,
        'block_cond': float,
        'block_indices': list of peak indices in block
      }
    """
    results = []
    P = len(theta_hat)
    assert P == 4 * n_peaks, "theta_hat length mismatch"

    # Precompute full model baseline once
    f0 = model_func(x, *theta_hat)
    N = len(x)

    for center in range(n_peaks):
        # define block: center-1, center, center+1 (clipped)
        block_peaks = [i for i in (center - 1, center, center + 1) if 0 <= i < n_peaks]
        n_block = len(block_peaks)
        idxs = []
        for p in block_peaks:
            idxs.extend(list(range(p * 4, p * 4 + 4)))  # global param indices for block

        # choose mask covering union of blocks, local window factor aggregated
        x0s = [theta_hat[p * 4 + 1] for p in block_peaks]
        sLs = [theta_hat[p * 4 + 2] for p in block_peaks]
        sRs = [theta_hat[p * 4 + 3] for p in block_peaks]
        width = max([sLs[i] + sRs[i] for i in range(n_block)] + [1.0])
        mask = np.zeros_like(x, dtype=bool)
        for x0 in x0s:
            mask |= (x >= x0 - window_factor * width) & (
                x <= x0 + window_factor * width
            )
        if mask.sum() < 6:
            mask = slice(None)  # use full data if local too small

        # Build block Jacobian J_block (rows = mask.sum(), cols = 4*n_block)
        rows = x[mask] if not isinstance(mask, slice) else x
        f0_block = f0[mask] if not isinstance(mask, slice) else f0
        J_block = np.zeros((len(rows), 4 * n_block))
        for col_idx, gidx in enumerate(idxs):
            step = theta_hat[gidx] * dlog
            if step == 0:
                step = dlog
            theta_step = theta_hat.copy()
            theta_step[gidx] = theta_hat[gidx] + step
            f1 = model_func(x, *theta_step)
            f1_block = f1[mask] if not isinstance(mask, slice) else f1
            J_block[:, col_idx] = (f1_block - f0_block) / step

        # SVD and truncated pseudo-inverse for block covariance
        U, svals_block, VT = np.linalg.svd(J_block, full_matrices=False)
        cutoff = svals_block[0] * rcond
        kept_block = int(np.sum(svals_block > cutoff))
        s_inv = np.zeros_like(svals_block)
        s_inv[svals_block > cutoff] = 1.0 / svals_block[svals_block > cutoff]
        JTJ_pinv_block = (
            VT.T * s_inv
        ) @ U.T  # pseudo-inverse of J_block (since SVD on J)
        # Convert to parameter covariance: Cov_block = sigma^2 * (J^T J)^+ = sigma^2 * JTJ_pinv_block
        # Need sigma estimate (use global residuals if available)
        residuals = y - f0
        sigma_est = np.sqrt(np.sum(residuals**2) / max(1, (N - len(theta_hat))))
        Cov_block = (sigma_est**2) * JTJ_pinv_block

        # Extract marginal covariance for the central peak within block
        # Find local offset of center in block_peaks
        center_pos_in_block = block_peaks.index(center)
        local_base = center_pos_in_block * 4
        Cov_local = Cov_block[local_base : local_base + 4, local_base : local_base + 4]

        # Compute integral and stderr_I depending on paramization
        # We must detect whether model_func expects amplitude or integral first param.
        # We assume theta ordering is [I/x, x0, sL, sR]; user should ensure consistency.
        I_val = theta_hat[center * 4 + 0]
        stderr_I = np.sqrt(max(Cov_local[0, 0], 0.0))

        # Build J_local for scoring: use block's columns restricted to central peak columns
        # This gives columns with the same scaling as used in the block
        cols_local = list(range(center_pos_in_block * 4, center_pos_in_block * 4 + 4))
        J_local = J_block[:, cols_local]

        # compute the lightweight quality score for this central peak from its local block info
        score_dict = _peak_quality_score_from_J_local(J_local, rcond=rcond)

        # Overwrite kept to reflect block-kept if that is more informative
        score_dict["block_svals"] = svals_block
        score_dict["block_kept"] = kept_block
        score_dict["block_cond"] = float(svals_block[0] / (svals_block[-1] + 1e-30))
        score_dict["block_peaks"] = block_peaks
        score_dict["Cov_local"] = Cov_local
        score_dict["I"] = float(I_val)
        score_dict["stderr_I"] = float(stderr_I)

        cond_block = svals_block[0] / (svals_block[-1] + 1e-30)
        logc = np.log10(cond_block)
        # sigmoid from 0 (good) to 1 (bad) between cond 1e6–1e10
        p = 1 / (1 + np.exp(-(logc - 8) / 0.8))
        score_dict["score"] *= 1 - 0.8 * p
        score_dict["overlap_penalty"] = p
        score_dict["block_cond"] = cond_block

        # Friendly message tweak: if block shows strong cross-coupling (kept_block < 4*n_block)
        if kept_block < 4 * n_block:
            score_dict["message"] = (
                score_dict.get("message", "") + f"; block rank {kept_block}/{4*n_block}"
            )
        results.append(score_dict)

    return results
