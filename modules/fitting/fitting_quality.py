from calendar import c
from dataclasses import dataclass
import os
import time
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
    bi_gaussian,
    bi_Lorentzian,
    bootstrap_std,
    combine_errors,
)
from modules.rendercallback import RenderCallback
from modules.utils import log
import dearpygui.dearpygui as dpg
from modules.fitting.refiner import refine_iteration
from copy import deepcopy
from typing import Optional
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
from threading import Lock


# Convergence of the refiner: parameters averaged over a window of iterations are
# compared with the previous window, in units of their Laplace error bars.
# Bootstrap refits start from the fitted solution and settle quickly.
CONVERGENCE_WINDOW = 50
BOOTSTRAP_CONVERGENCE_WINDOW = 10
# Refits keep going for this fraction of their iterations once R² reached the threshold.
# Restarts start far from the optimum: they cross the threshold well before it
R2_EXTRA_FRACTION = 0.1
RESTART_R2_EXTRA_FRACTION = 0.5
# Iterations over which the tangent of the R² evolution is measured
R2_FLAT_WINDOW = 20


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


# Set in the worker processes: lets running refits stop as soon as the user asks
_worker_stop_event = None


def _init_worker(stop_event):
    global _worker_stop_event
    _worker_stop_event = stop_event
    try:  # one BLAS thread per worker: the refits already run in parallel
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:
        pass


def _stop_requested(render_callback) -> bool:
    if render_callback:
        render_callback.execute()
    return bool(dpg.get_value("stop_fitting_button"))


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
    r2_threshold: Optional[float] = None,
    r2_flat_threshold: float = 0.0,
    min_iterations: int = 0,
    accept_only_improving: bool = False,
) -> dict | Literal[False]:
    """
    Perform advanced statistical analysis using bootstrap and random start methods.
    Refits stop like the main fit: parameters stable (theta_threshold),
    R² above r2_threshold, or R² evolution flat (r2_flat_threshold), but not
    before min_iterations.
    """

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

    task_pool = []
    # Thread-safe counters
    cpu_count = os.cpu_count() or 2
    completed_lock = Lock()
    completed_tasks = {"count": 0, "successful": 0, "rejected": 0}
    BATCH_SIZE = min(cpu_count * 2, 50)  # Process 4x CPU cores or max 50 at once
    started = time.time()
    label = "Refit with randomisation" if method == "initial" else "Bootstrap"

    # Refits run cpu_count at a time: the analysis takes `rounds_total` rounds of
    # about one refit duration each (measured on the first round of each batch)
    rounds_total = sum(
        -(-min(BATCH_SIZE, macro_iteration - b * BATCH_SIZE) // cpu_count)
        for b in range((macro_iteration + BATCH_SIZE - 1) // BATCH_SIZE)
    )
    first_round_durations: list[float] = []

    def show_progress(unfinished_in_batch: int):
        elapsed = time.time() - started
        count = completed_tasks["count"]
        running = min(unfinished_in_batch, cpu_count)
        text = (
            f"{label}: {count}/{macro_iteration} done, {running} running, "
            f"{int(elapsed // 60)} min {int(elapsed % 60):02d} s"
        )
        if first_round_durations:
            total = rounds_total * float(np.median(first_round_durations))
            remaining = max(total - elapsed, 0.0)
            text += f", about {int(remaining // 60)} min {int(remaining % 60):02d} s left"
        dpg.set_value("Fitting_indicator_sub_text", text)
    num_batches = (macro_iteration + BATCH_SIZE - 1) // BATCH_SIZE

    avg_iterations = None
    batch_iterations = []
    rescale = 1.0

    # Starting jitter of the bootstrap refits: fixed (it used to be tuned so that
    # refits "converged" in a few iterations under the old parameter-change test)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, macro_iteration)

        if batch_iterations != []:
            avg_iterations = int(np.mean(batch_iterations))
            batch_iterations = []

        if render_callback:
            dpg.set_value(
                "Fitting_indicator_sub_text",
                (
                    f"Running {start_idx} to {end_idx}"
                    + f" average iterations last batch: {avg_iterations}"
                    if avg_iterations
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

            task.r2_threshold = r2_threshold
            task.r2_flat_threshold = r2_flat_threshold
            task.min_iterations = min(min_iterations, micro_iteration)
            task.accept_only_improving = accept_only_improving
            task_pool.append(task)

        stop_event = multiprocessing.Event()
        batch_started = time.time()
        finished_in_batch = 0
        with ProcessPoolExecutor(
            max_workers=cpu_count, initializer=_init_worker, initargs=(stop_event,)
        ) as executor:
            futures = {
                executor.submit(execute_quick_fit, task): task for task in task_pool
            }

            def finished_futures():
                # Poll so that Stop is noticed even while long refits are running,
                # and show progress: a refit can take minutes before it finishes
                pending = set(futures)
                last_progress = 0.0
                while pending:
                    done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                    if _stop_requested(render_callback):
                        return
                    if render_callback and time.time() - last_progress > 1.0:
                        last_progress = time.time()
                        show_progress(len(pending) + len(done))
                    yield from done

            for future in finished_futures():
                task = futures[future]
                fitted_peaks, converged, iteration = future.result()

                # Update progress (thread-safe)
                with completed_lock:
                    completed_tasks["count"] += 1
                    finished_in_batch += 1
                    if finished_in_batch <= min(cpu_count, len(task_pool)):
                        first_round_durations.append(time.time() - batch_started)
                    if converged:
                        completed_tasks["successful"] += 1
                    batch_iterations.append((iteration))

                    # Update GUI
                    if render_callback:
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

            if _stop_requested(None):
                log("Error analysis stopped by user.")
                dpg.set_value("Fitting_indicator_text", "Stopping ...")
                dpg.set_item_label("stop_fitting_button", "Stopping...")
                stop_event.set()  # running refits exit at their next iteration
                for f in futures:
                    f.cancel()
                return False

    # Compute standard errors from bootstrap distribution
    final_result = {}
    for peak in working_peak_list:
        if len(perturbation_results[peak].A) < macro_iteration * 0.5:
            log(
                f"Warning: Peak {peak} had only {len(perturbation_results[peak].A)} successful fits"
            )
            final_result[peak] = None
            continue

        # The uncertainty of a parameter is the spread of its resampled values
        # (not the standard error of their mean, which shrinks with more resamples)
        final_result[peak] = {
            "A": bootstrap_std(perturbation_results[peak].A),
            "x0": bootstrap_std(perturbation_results[peak].x0),
            "sigma_L": bootstrap_std(perturbation_results[peak].sigma_L),
            "sigma_R": bootstrap_std(perturbation_results[peak].sigma_R),
            "integral": bootstrap_std(perturbation_results[peak].integral),
            "n_samples": len(perturbation_results[peak].A),
            "base": bootstrap_std(perturbation_results[peak].base),
        }
    if dpg.does_alias_exist("noise"):
        dpg.delete_item("noise")

    log(
        f"{'Random restarts' if method == 'initial' else 'Bootstrap'} complete: "
        f"{completed_tasks['count'] - completed_tasks['rejected']}/{macro_iteration} fits used "
        f"({completed_tasks['successful']} converged, {completed_tasks['rejected']} rejected)"
    )

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
    convergence_window: int = CONVERGENCE_WINDOW
    r2_threshold: Optional[float] = None  # stop when R² is above it, as the main fit
    r2_flat_threshold: float = 0.0  # stop when the R² evolution is flat, as the main fit
    r2_extra_fraction: float = R2_EXTRA_FRACTION  # extra iterations once R² reached
    min_iterations: int = 0  # no stopping criterion applies before this
    accept_only_improving: bool = False  # refiner keeps only steps lowering the residual


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
        task.convergence_window,
        task.r2_threshold,
        task.r2_flat_threshold,
        task.r2_extra_fraction,
        task.min_iterations,
        task.accept_only_improving,
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
    convergence_window: int = CONVERGENCE_WINDOW,
    r2_threshold: Optional[float] = None,
    r2_flat_threshold: float = 0.0,
    r2_extra_fraction: float = R2_EXTRA_FRACTION,
    min_iterations: int = 0,
    accept_only_improving: bool = False,
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
        r2_stop_at: Optional[int] = None  # last iteration once R² reached the threshold
        quality_metrics: Optional[FitQualityMetricsReduced] = None
        # theta_threshold: largest allowed move per window, in Laplace error bars
        monitor = ConvergenceMonitor(
            spectrum,
            list(working_peaks.keys()),
            data_x,
            data_y,
            threshold=theta_threshold,
            window=convergence_window,
        )
        flatness = R2FlatnessMonitor(r2_flat_threshold)
        for iteration in range(n_iterations):
            if _worker_stop_event is not None and _worker_stop_event.is_set():
                return {}, False, iteration
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
                    accept_only_improving=accept_only_improving,
                )

            quality_metrics = calculate_fit_quality_metrics(
                data_x,
                data_y,
                spectrum,
                list(working_peaks.keys()),
                rmse_only=True,
            )

            # The criteria are tracked from the start, but none can end the refit
            # before min_iterations (it must first respond to its own data)
            may_stop = iteration + 1 >= min_iterations

            # Same stopping rule as the main fit: R² above the threshold, then
            # r2_extra_fraction more iterations so that the refit moves towards
            # the optimum instead of stopping on the threshold boundary
            if r2_stop_at is None and r2_threshold is not None and quality_metrics.r_squared > r2_threshold:
                converged = True
                r2_stop_at = iteration + max(1, int(np.ceil(r2_extra_fraction * (iteration + 1))))
            if may_stop and r2_stop_at is not None and iteration >= r2_stop_at:
                break

            # R² no longer improving: more iterations would not change the fit
            if flatness.update(quality_metrics.r_squared) and may_stop:
                converged = True
                break

            if check_convergence:
                rmse_converged = False
                theta_converged = False
                if check_convergence == "wRMSE" or check_convergence == "both":
                    rmse_converged = quality_metrics.weighted_rmse < wRMSE_threshold

                if check_convergence == "theta-gradient" or check_convergence == "both":
                    monitor.update(iteration + 1)
                    theta_converged = monitor.converged

                if check_convergence == "both":
                    criterion_met = bool(rmse_converged and theta_converged)
                elif check_convergence == "wRMSE":
                    criterion_met = rmse_converged
                else:
                    criterion_met = bool(theta_converged)
                # During the iterations after R² was reached, a refit that is not
                # yet stable stays converged (R²) and simply runs on
                if criterion_met and may_stop:
                    converged = True
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
            # Starts from the fitted solution: settles quickly, short window
            convergence_window=BOOTSTRAP_CONVERGENCE_WINDOW,
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
    max_z = 2.5  # draws are capped so that rare extreme starts do not dominate the spread
    neighbour_gap_fraction = 0.4  # an apex moves at most this fraction of the gap to a neighbour

    task_data_y = data_y.copy()
    task_spectrum = deepcopy(spectrum)

    def z():
        return float(np.clip(np.random.randn(), -max_z, max_z))

    # Apex bounds keep the peaks in order: a swap of two close peaks would mix
    # their integrals in the spread
    by_position = sorted(working_peak_list, key=lambda p: spectrum.peaks[p].x0_init)
    positions = [spectrum.peaks[p].x0_init for p in by_position]
    x0_bounds = {}
    for i, peak in enumerate(by_position):
        low = (
            positions[i] - neighbour_gap_fraction * (positions[i] - positions[i - 1])
            if i > 0
            else -np.inf
        )
        high = (
            positions[i] + neighbour_gap_fraction * (positions[i + 1] - positions[i])
            if i < len(positions) - 1
            else np.inf
        )
        x0_bounds[peak] = (low, high)

    for peak in working_peak_list:
        p = spectrum.peaks[peak]
        # Same reference as the other parameters: the starting widths
        search_width = p.sigma_L_init + p.sigma_R_init
        noise_scale_X0_applied = (
            noise_scale_X0 * 2.0 if p.user_added else noise_scale_X0
        )  # be harder on user added peaks
        # Log-normal jitter: symmetric in ratio (x0.6 as likely as x1.6) and positive
        working_peaks[peak] = peak_params(
            A_refined=p.A_init * np.exp(z() * noise_scale_A),
            x0_refined=float(
                np.clip(
                    p.x0_init + z() * search_width * noise_scale_X0_applied,
                    *x0_bounds[peak],
                )
            ),
            sigma_L=p.sigma_L_init * np.exp(z() * noise_scale_w),
            sigma_R=p.sigma_R_init * np.exp(z() * noise_scale_w),
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
        r2_extra_fraction=RESTART_R2_EXTRA_FRACTION,
    )

    return task


def _peak_profile(x, peak_model, integral, x0, sigma_L, sigma_R):
    """Single peak parametrised by its integral instead of its amplitude."""
    if peak_model == "lorentzian":
        A = integral / bi_Lorentzian_integral(1.0, sigma_L, sigma_R)
        return bi_Lorentzian(x, A, x0, sigma_L, sigma_R)
    A = integral / bi_gaussian_integral(1.0, sigma_L, sigma_R)
    return bi_gaussian(x, A, x0, sigma_L, sigma_R)


def _laplace_jacobian(spectrum: MSData, peaks: list[int], x: np.ndarray, step=1e-4):
    """
    Jacobian of the model w.r.t. log(integral), x0 / width, log(sigma_L),
    log(sigma_R) of every peak (one block of 4 columns per peak: each peak only
    affects its own term). Also returns the integrals and widths.
    """
    J = np.zeros((len(x), 4 * len(peaks)))
    widths = np.zeros(len(peaks))
    integral_values = np.zeros(len(peaks))
    for i, peak in enumerate(peaks):
        p = spectrum.peaks[peak]
        if spectrum.peak_model == "lorentzian":
            integral = bi_Lorentzian_integral(p.A_refined, p.sigma_L, p.sigma_R)
        else:
            integral = bi_gaussian_integral(p.A_refined, p.sigma_L, p.sigma_R)
        theta = [integral, p.x0_refined, p.sigma_L, p.sigma_R]
        integral_values[i] = integral
        width = (p.sigma_L + p.sigma_R) / 2
        widths[i] = width

        def profile(t):
            return _peak_profile(x, spectrum.peak_model, *t)

        J[:, 4 * i] = profile(theta)  # d/dlog(I) = f
        for j in (1, 2, 3):
            plus, minus = list(theta), list(theta)
            if j == 1:  # x0, in units of the peak width
                plus[1] += step * width
                minus[1] -= step * width
            else:  # log sigma
                plus[j] *= np.exp(step)
                minus[j] *= np.exp(-step)
            J[:, 4 * i + j] = (profile(plus) - profile(minus)) / (2 * step)
    return J, integral_values, widths




def parameter_state(spectrum: MSData, peaks: list[int]) -> np.ndarray:
    """Per peak: log(integral), x0, log(sigma_L), log(sigma_R)."""
    integral = bi_Lorentzian_integral if spectrum.peak_model == "lorentzian" else bi_gaussian_integral
    return np.array(
        [
            [
                np.log(max(integral(q.A_refined, q.sigma_L, q.sigma_R), 1e-300)),
                q.x0_refined,
                np.log(max(q.sigma_L, 1e-300)),
                np.log(max(q.sigma_R, 1e-300)),
            ]
            for q in (spectrum.peaks[p] for p in peaks)
        ]
    )


def parameter_sd(spectrum: MSData, peaks: list[int], x, y, rcond=1e-10) -> np.ndarray:
    """
    Laplace standard errors (MAD noise, as laplace_covariance_analysis) of the
    quantities of parameter_state, per peak; the apex error is in m/z.
    """
    J, _, widths = _laplace_jacobian(spectrum, peaks, x)
    residual = y - spectrum.calculate_mbg(x, fitting=True)
    sigma = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    U, svals, VT = np.linalg.svd(J, full_matrices=False)
    keep = svals > svals[0] * rcond
    variance = sigma**2 * np.sum((VT[keep] / svals[keep, None]) ** 2, axis=0)
    sd = np.sqrt(variance).reshape(-1, 4)
    sd[:, 1] *= widths
    return sd


class ConvergenceMonitor:
    """
    The refiner is converged when the average parameters over the last `window`
    iterations differ from the average over the window before by less than
    `threshold` error bars, for every peak's integral, apex and widths.

    - Scale free: changes are in units of each parameter's Laplace error bar.
    - Averaging over a window removes the iteration-to-iteration jitter of the
      refiner (random peak order, damped steps), which otherwise sets a floor on
      the change measured between two single iterations; a slow drift remains
      fully visible.
    """

    def __init__(self, spectrum: MSData, peaks, x, y, threshold, window=CONVERGENCE_WINDOW):
        self.spectrum, self.peaks, self.x, self.y = spectrum, list(peaks), x, y
        self.threshold, self.window = threshold, window
        self.states = []  # parameter states of the current window
        self.previous_mean = None
        self.last_move = float("nan")  # largest move between window averages, in error bars
        self.converged = False

    def update(self, iteration: int) -> bool:
        """Call once per iteration; True when a window was completed and compared."""
        self.states.append(parameter_state(self.spectrum, self.peaks))
        if iteration == 0 or iteration % self.window:
            return False
        mean = np.mean(self.states, axis=0)
        self.states = []
        previous, self.previous_mean = self.previous_mean, mean
        if previous is None:
            return False
        sd = parameter_sd(self.spectrum, self.peaks, self.x, self.y)
        with np.errstate(divide="ignore", invalid="ignore"):
            moves = np.where(sd > 0, np.abs(mean - previous) / sd, 0.0)
        self.last_move = float(np.nanmax(moves)) if moves.size else 0.0
        self.converged = self.last_move < self.threshold
        return True


class R2FlatnessMonitor:
    """
    The fit is converged when the R² evolution has become flat.

    - The tangent is the least squares slope of log(1 - R²) over the last `window`
      iterations: it averages out the iteration-to-iteration jitter of the refiner.
    - It is relative to the part of the data still unexplained, so the same setting
      works at R² = 0.95 and R² = 0.999 (the slope of R² itself shrinks with 1 - R²).
    - `last_gain` is the fraction by which the residual shrinks over one window at
      the current slope; converged when it is below `threshold`. A residual that
      grows (negative gain) also counts as flat: iterating does not help.
    - threshold <= 0 disables the criterion.
    """

    def __init__(self, threshold: float, window: int = R2_FLAT_WINDOW):
        self.threshold, self.window = threshold, window
        self.history: list[float] = []
        self.last_gain = float("nan")
        self.converged = False

    def update(self, r_squared: float) -> bool:
        """Call once per iteration; True when the R² curve is flat."""
        self.history.append(float(np.log(max(1.0 - r_squared, 1e-300))))
        self.history = self.history[-self.window :]
        if len(self.history) < self.window:
            return False
        slope = np.polyfit(np.arange(self.window), self.history, 1)[0]
        self.last_gain = float(1.0 - np.exp(slope * self.window))
        self.converged = self.threshold > 0 and self.last_gain < self.threshold
        return self.converged


def laplace_covariance_analysis(step=1e-4, rcond=1e-10) -> dict[int, dict]:
    """
    Laplace (linearised) parameter covariance of the current fit:
    Cov = s² (JᵀJ)⁻¹, with J the Jacobian of the model w.r.t. every peak parameter.

    Parameters are log(integral), x0 / width, log(sigma_L), log(sigma_R), so the
    covariance of log(integral) is directly the relative error of the integral and
    the result does not depend on the units of each parameter.
    s is the robust (MAD) noise estimate of the residual, as in the parametric
    bootstrap. The errors are therefore the precision limited by noise: systematic
    misfit of the peak shape (structured residual) is not included.
    The correlations do not depend on s.

    Per peak, stores the integral relative standard error, the apex standard error
    and the strongest correlation between its integral and another peak's integral
    (strongly negative: area can move between the two peaks).
    """
    spectrum: MSData = get_global_msdata_ref()
    spectrum.laplace_integral_corr = None
    peak_list = sorted(
        (
            peak
            for peak in spectrum.peaks
            if spectrum.peaks[peak].fitted and not spectrum.peaks[peak].do_not_fit
        ),
        key=lambda peak: spectrum.peaks[peak].x0_refined,
    )
    if not peak_list:
        log("Laplace analysis: no fitted peaks")
        return {}

    x = spectrum.baseline_corrected[:, 0]
    y = spectrum.baseline_corrected[:, 1]
    residual = y - spectrum.calculate_mbg(x)
    n_points, n_params = len(x), 4 * len(peak_list)
    if n_points <= n_params:
        log("Laplace analysis: not enough data points")
        return {}

    J, integral_values, widths = _laplace_jacobian(spectrum, peak_list, x, step)
    sigma = 1.4826 * np.median(np.abs(residual - np.median(residual)))

    U, svals, VT = np.linalg.svd(J, full_matrices=False)
    keep = svals > svals[0] * rcond
    cov = sigma**2 * (VT[keep].T / svals[keep] ** 2) @ VT[keep]
    # Parameters involved in directions the data cannot constrain at all
    degenerate = np.zeros(n_params, dtype=bool)
    if not np.all(keep):
        degenerate = np.any(np.abs(VT[~keep]) > 0.1, axis=0)

    std = np.sqrt(np.clip(np.diag(cov), 0, None))
    integral_idx = np.arange(len(peak_list)) * 4
    I_std = std[integral_idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        I_corr = cov[np.ix_(integral_idx, integral_idx)] / np.outer(I_std, I_std)
    I_corr = np.nan_to_num(I_corr)
    np.fill_diagonal(I_corr, 1.0)
    spectrum.laplace_integral_corr = {"peaks": list(peak_list), "matrix": I_corr}

    results = {}
    for i, peak in enumerate(peak_list):
        se_integral = float(I_std[i])  # relative (log scale)
        se_x0 = float(std[4 * i + 1] * widths[i])
        # Integral correlation with the m/z neighbours (peak_list is sorted by apex):
        # overlap, hence area exchange, happens between adjacent peaks
        left_peak = peak_list[i - 1] if i > 0 else -1
        right_peak = peak_list[i + 1] if i < len(peak_list) - 1 else -1
        corr_left = float(I_corr[i, i - 1]) if left_peak >= 0 else 0.0
        corr_right = float(I_corr[i, i + 1]) if right_peak >= 0 else 0.0
        # Strongest anticorrelation of the two, for flags and sorting
        if left_peak >= 0 and (right_peak < 0 or corr_left <= corr_right):
            area_corr, area_peak = corr_left, left_peak
        else:
            area_corr, area_peak = corr_right, right_peak

        if degenerate[4 * i : 4 * i + 4].any():
            message = "Not identifiable: the data cannot constrain some of this peak's parameters"
            se_integral = np.inf
        else:
            message = f"Integral ± {se_integral * 100:.1f}%, apex ± {se_x0:.2f} m/z"
            for side, other, r in (("left", left_peak, corr_left), ("right", right_peak, corr_right)):
                if other >= 0:
                    message += f"; r with {side} neighbour Peak {other} = {r:+.2f}"
                    if r <= -0.8:
                        message += " (area can move between them)"

        p = spectrum.peaks[peak]
        p.laplace_se_integral = se_integral
        p.laplace_se_x0 = se_x0
        # Best available error: Laplace alone until the bootstrap has run
        integral = integral_values[i]
        p.se_integral = combine_errors(
            [p.se_integral_bootstrap, se_integral * integral], p.se_integral_restart
        )
        p.se_x0 = combine_errors([p.se_x0_bootstrap, se_x0], p.se_x0_restart)
        p.laplace_area_corr = area_corr
        p.laplace_area_peak = area_peak
        p.laplace_corr_left = corr_left
        p.laplace_corr_right = corr_right
        p.laplace_left_peak = left_peak
        p.laplace_right_peak = right_peak
        p.laplace_message = message
        results[peak] = {
            "se_integral_rel": se_integral,
            "se_x0": se_x0,
            "area_corr": area_corr,
            "area_peak": area_peak,
        }

    log(
        f"Laplace analysis done (noise {sigma:.3g})"
        + (f", {int(np.sum(~keep))} degenerate direction(s)" if not np.all(keep) else "")
    )
    return results
