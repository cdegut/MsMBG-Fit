from modules.data_structures import MSData
import numpy as np

from modules.math import (
    bi_Lorentzian,
    bi_Lorentzian_integral,
    bi_gaussian,
    bi_gaussian_integral,
)

# Multiply the refiner's steps before the step limits are applied (1.0 = original
# behaviour). The integral step is already about the full correction (blended in at
# 40 %), so INTEGRAL_GAIN above ~2.5 overshoots; the apex and width steps use fixed
# divisors and are much smaller than their natural scale (SHAPE_GAIN).
INTEGRAL_GAIN = 1.0
SHAPE_GAIN = 1.0


# Safeguard: in one iteration a peak's integral may shrink at most to this fraction
# of its current value (and never become negative). Without it, a weak, poorly
# constrained peak can be pushed to zero and its neighbours then run away.
MIN_INTEGRAL_RATIO = 0.8

# Test switch: direction of the apex step from the fit of the whole peak region (see
# _apex_direction). The original rule compares two narrow windows next to the apex;
# for asymmetric peaks it is biased to the right and can move the apex the wrong
# way, a cause of slow drift. The step size is unchanged.
# Off: tested on Um2x / Um4x it makes the whole refiner unstable (a weak peak
# collapses and its neighbours run away).
APEX_DIRECTION_FROM_FIT = False


def _apex_direction(spectrum, peak, data_x, data_y, x0, sigma_L, sigma_R) -> float:
    """
    +1 / -1: the direction of apex shift that reduces the residual over the peak
    region (x0 - 3 sigma_L to x0 + 3 sigma_R), the sign of
    sum(residual * d(peak profile)/d(x0)); 0 if undetermined.
    """
    region = (data_x >= x0 - 3 * sigma_L) & (data_x <= x0 + 3 * sigma_R)
    if np.count_nonzero(region) < 3:
        return 0.0
    x = data_x[region]
    residual = data_y[region] - spectrum.calculate_mbg(x, fitting=True)
    profile = bi_Lorentzian if spectrum.peak_model == "lorentzian" else bi_gaussian
    A = spectrum.peaks[peak].A_refined
    h = 1e-3 * (sigma_L + sigma_R) / 2
    g = (profile(x, A, x0 + h, sigma_L, sigma_R) - profile(x, A, x0 - h, sigma_L, sigma_R)) / (2 * h)
    gradient = float(np.sum(residual * g))
    return float(np.sign(gradient)) if np.isfinite(gradient) else 0.0


# Test switch: keep a peak's update only if it lowers the residual over the peak's
# region (old and new extent, +-3 sigma, including the overlap with its neighbours);
# otherwise try 1/2, 1/4, 1/8 of the step, then keep the old values. Makes every
# accepted step an improvement. Note: width regularisation pulls that cost fit
# quality are then rejected too. False = original behaviour.
ACCEPT_ONLY_IMPROVING = False
BACKTRACK_STEPS = 3


def _accept_only_improving(spectrum, peak, previous, data_x, data_y):
    q = spectrum.peaks[peak]
    new = (q.A_refined, q.x0_refined, q.sigma_L, q.sigma_R, q.integral)
    low = min(previous[1] - 3 * previous[2], new[1] - 3 * new[2])
    high = max(previous[1] + 3 * previous[3], new[1] + 3 * new[3])
    region = (data_x >= low) & (data_x <= high)
    if np.count_nonzero(region) < 3:
        return
    x, y = data_x[region], data_y[region]

    def local_ssr(values):
        q.A_refined, q.x0_refined, q.sigma_L, q.sigma_R, q.integral = values
        return float(np.sum((y - spectrum.calculate_mbg(x, fitting=True)) ** 2))

    ssr_previous = local_ssr(previous)
    step = 1.0
    for _ in range(BACKTRACK_STEPS + 1):
        candidate = tuple(p0 + step * (p1 - p0) for p0, p1 in zip(previous, new))
        if local_ssr(candidate) <= ssr_previous:
            return  # accepted (values already set)
        step *= 0.5
    local_ssr(previous)  # no improving step: keep the old values


def refine_iteration(
    peak: int,
    data_x,
    data_y,
    spectrum: MSData,
    original_peak_width: float,
    force_gaussian=False,
    widths=(-1, -1, -1, -1),
    alpha=0.8,
    integral_gain=None,
    shape_gain=None,
    accept_only_improving=None,
):
    integral_gain = INTEGRAL_GAIN if integral_gain is None else integral_gain
    shape_gain = SHAPE_GAIN if shape_gain is None else shape_gain
    # Passed explicitly by the fits (GUI option): the refits run in worker
    # processes, which would not see a change of the module constant
    accept_only_improving = (
        ACCEPT_ONLY_IMPROVING if accept_only_improving is None else accept_only_improving
    )
    x0_fit = spectrum.peaks[peak].x0_refined
    sigma_L_fit = spectrum.peaks[peak].sigma_L
    sigma_R_fit = spectrum.peaks[peak].sigma_R
    sampling_rate = spectrum.peaks[peak].sampling_rate

    # Check for overlap with neighbors and limit refinement
    min_neighbor_distance = float("inf")
    for other_peak in spectrum.peaks:
        if other_peak != peak and not spectrum.peaks[other_peak].do_not_fit:
            distance = abs(spectrum.peaks[other_peak].x0_refined - x0_fit)
            min_neighbor_distance = min(min_neighbor_distance, distance)

    # If peaks are very close, reduce alpha and constrain widths
    overlap_threshold = (sigma_L_fit + sigma_R_fit) * 1.5
    has_close_neighbor = min_neighbor_distance < overlap_threshold
    if has_close_neighbor:
        alpha = alpha * 0.5  # More conservative updates

    # ######################
    # # Adjust the amplitude
    # ######################
    # R_val = sigma_R_fit if sigma_R_fit > sampling_rate * 20 else sampling_rate * 5
    # L_val = sigma_L_fit if sigma_L_fit > sampling_rate * 20 else sampling_rate * 5
    # mask = (data_x >= x0_fit - R_val) & (data_x <= x0_fit + L_val)
    # data_x_peak = data_x[mask]
    # data_y_peak = data_y[mask]
    # A_fit = spectrum.peaks[peak].A_refined
    # if len(data_x_peak) > 0 and len(data_y_peak) > 0:
    #     peak_error = np.mean(
    #         data_y_peak - spectrum.calculate_mbg(data_x_peak, fitting=True)
    #     )
    #     A_fit = spectrum.peaks[peak].A_refined + (peak_error / 10)

    ######################
    # Adjust integral (instead of amplitude)
    ######################

    A_ref = spectrum.peaks[peak].A_refined
    if spectrum.peak_model == "lorentzian":
        integral_fit = bi_Lorentzian_integral(A_ref, sigma_L_fit, sigma_R_fit)
    else:
        integral_fit = bi_gaussian_integral(A_ref, sigma_L_fit, sigma_R_fit)

    if not np.isfinite(integral_fit) or integral_fit <= 0:
        integral_fit = 1.0
    integral_current = integral_fit

    R_val = sigma_R_fit if sigma_R_fit > sampling_rate * 20 else sampling_rate * 5
    L_val = sigma_L_fit if sigma_L_fit > sampling_rate * 20 else sampling_rate * 5
    mask = (data_x >= x0_fit - L_val) & (data_x <= x0_fit + R_val)
    data_x_peak = data_x[mask]
    data_y_peak = data_y[mask]

    if len(data_x_peak) > 0 and len(data_y_peak) > 0:
        model_y = spectrum.calculate_mbg(data_x_peak, fitting=True)
        residual = data_y_peak - model_y
        # Empirical adjustment: scale integral with local mean residual
        integral_fit += integral_gain * np.mean(residual) * (sigma_L_fit + sigma_R_fit)
        integral_fit = max(integral_fit, MIN_INTEGRAL_RATIO * integral_current)

    ############
    # Move x0
    ############
    # Create symmetric windows around current x0_fit
    window_size = max((sigma_L_fit + sigma_R_fit) / 4, sampling_rate * 3)
    window_half = window_size / 2

    L_mask = (data_x >= x0_fit - window_size) & (data_x <= x0_fit - window_half)
    R_mask = (data_x >= x0_fit + window_half) & (data_x <= x0_fit + window_size)

    data_x_L = data_x[L_mask]
    data_y_L = data_y[L_mask]
    data_x_R = data_x[R_mask]
    data_y_R = data_y[R_mask]

    # Only adjust if we have enough data points
    if len(data_x_L) >= 3 and len(data_x_R) >= 3:
        mbg_L = spectrum.calculate_mbg(data_x_L, fitting=True)
        mbg_R = spectrum.calculate_mbg(data_x_R, fitting=True)

        error_l = np.mean((data_y_L - mbg_L))
        error_r = np.mean((data_y_R - mbg_R))

        # Check for valid errors
        if not (
            np.isnan(error_l)
            or np.isnan(error_r)
            or np.isinf(error_l)
            or np.isinf(error_r)
        ):
            # If right side has more residual error, peak center should move right
            error_diff = error_r - error_l

            # Only adjust if asymmetry is significant
            if abs(error_diff) > abs(error_l + error_r) / 10:
                # Conservative adjustment
                offset = shape_gain * error_diff / (10000) * (sigma_L_fit + sigma_R_fit) / 2
                max_offset = (sigma_L_fit + sigma_R_fit) / 200
                offset = np.clip(offset, -max_offset, max_offset)
                if APEX_DIRECTION_FROM_FIT:
                    offset = abs(offset) * _apex_direction(
                        spectrum, peak, data_x, data_y, x0_fit, sigma_L_fit, sigma_R_fit
                    )
                x0_fit = x0_fit + offset

        # Sharpen the peak
    if spectrum.peak_model == "lorentzian":
        max_iter = 5
    else:
        max_iter = 5

    for iteration in range(1, max_iter):
        min_window = sampling_rate * 3
        L_window = max(sigma_L_fit * iteration, min_window)
        R_window = max(sigma_R_fit * iteration, min_window)
        L_mask = (data_x >= x0_fit - L_window) & (data_x <= x0_fit - L_window / 2)
        R_mask = (data_x >= x0_fit + R_window / 2) & (data_x <= x0_fit + R_window)

        data_x_L = data_x[L_mask]
        data_y_L = data_y[L_mask]
        data_x_R = data_x[R_mask]
        data_y_R = data_y[R_mask]

        # Check if we have enough data points
        if len(data_x_L) < 3 or len(data_x_R) < 3:
            continue

        mbg_L = spectrum.calculate_mbg(data_x_L, fitting=True)
        mbg_R = spectrum.calculate_mbg(data_x_R, fitting=True)

        error_l = np.mean((data_y_L - mbg_L))
        error_r = np.mean((data_y_R - mbg_R))

        if (
            np.isnan(error_l)
            or np.isnan(error_r)
            or np.isinf(error_l)
            or np.isinf(error_r)
        ):
            continue

        val = 1000 * iteration**2

        max_adjustment = original_peak_width * 0.01  # 1% of original width

        sigma_L_adjustment = np.clip(shape_gain * error_l / val, -max_adjustment, max_adjustment)
        sigma_R_adjustment = np.clip(shape_gain * error_r / val, -max_adjustment, max_adjustment)

        sigma_L_fit = sigma_L_fit + sigma_L_adjustment
        sigma_R_fit = sigma_R_fit + sigma_R_adjustment

        if sigma_L_fit < sampling_rate:
            sigma_L_fit = sampling_rate * 3
        if sigma_R_fit < sampling_rate:
            sigma_R_fit = sampling_rate * 3

        if np.isnan(sigma_L_fit):
            sigma_L_fit = sampling_rate * 3
        if np.isnan(sigma_R_fit):
            sigma_R_fit = sampling_rate * 3

        if force_gaussian:
            sigma_L_fit = (sigma_L_fit + sigma_R_fit) / 2
            sigma_R_fit = sigma_L_fit

    # Apply width regularization if enabled
    if any(w != -1 for w in widths):
        factor = 4
        sigma_L_mean, sigma_R_mean, sigma_L_std, sigma_R_std = widths

        # Global regularization
        global_softness = 0.8  # Adjust between 0 (hard) and 1 (very soft)

        if sigma_L_fit > sigma_L_mean + sigma_L_std * factor:
            excess = sigma_L_fit - (sigma_L_mean + sigma_L_std * factor)
            sigma_L_fit = sigma_L_mean + sigma_L_std * factor + excess * global_softness
        if sigma_L_fit < sigma_L_mean - sigma_L_std * factor:
            deficit = (sigma_L_mean - sigma_L_std * factor) - sigma_L_fit
            sigma_L_fit = (
                sigma_L_mean - sigma_L_std * factor - deficit * global_softness
            )
        if sigma_R_fit > sigma_R_mean + sigma_R_std * factor:
            excess = sigma_R_fit - (sigma_R_mean + sigma_R_std * factor)
            sigma_R_fit = sigma_R_mean + sigma_R_std * factor + excess * global_softness
        if sigma_R_fit < sigma_R_mean - sigma_R_std * factor:
            deficit = (sigma_R_mean - sigma_R_std * factor) - sigma_R_fit
            sigma_R_fit = (
                sigma_R_mean - sigma_R_std * factor - deficit * global_softness
            )

        # Neighbor-based regularization with push-away mechanism
        max_width_ratio = 1.7

        distance_factor = 1.5 if spectrum.peak_model == "lorentzian" else 2.0
        neighbor_distance = (sigma_L_fit + sigma_R_fit) * 1.5

        # Find all neighbors and their distances
        neighbors = []
        for close_peak in spectrum.peaks:
            if close_peak != peak and not spectrum.peaks[close_peak].do_not_fit:
                distance = abs(spectrum.peaks[close_peak].x0_refined - x0_fit)
                if distance < neighbor_distance:
                    neighbors.append(
                        {
                            "index": close_peak,
                            "distance": distance,
                            "sigma_L": spectrum.peaks[close_peak].sigma_L,
                            "sigma_R": spectrum.peaks[close_peak].sigma_R,
                            "x0": spectrum.peaks[close_peak].x0_refined,
                        }
                    )

        if neighbors:
            # Sort by distance and get two closest
            neighbors.sort(key=lambda n: n["distance"])
            closest_neighbors = neighbors[:2]

            # Collect sigmas for regularization
            neighbor_sigmas_L = [n["sigma_L"] for n in neighbors]
            neighbor_sigmas_R = [n["sigma_R"] for n in neighbors]

            neighbor_L_median = np.median(neighbor_sigmas_L)
            neighbor_R_median = np.median(neighbor_sigmas_R)

            # Check if peak is getting too constrained
            needs_space = (
                sigma_L_fit < sigma_L_mean / 2 or sigma_R_fit < sigma_R_mean / 2
            )

            # if needs_space:
            #     # Push away the two closest neighbors slightly
            #     for neighbor in closest_neighbors:
            #         neighbor_idx = neighbor["index"]
            #         if neighbor["x0"] < x0_fit:
            #             # Neighbor is to the left, push it left
            #             spectrum.peaks[neighbor_idx].sigma_R = (
            #                 spectrum.peaks[neighbor_idx].sigma_R * 0.99
            #             )
            #         else:
            #             # Neighbor is to the right, push it right
            #             spectrum.peaks[neighbor_idx].sigma_L = (
            #                 spectrum.peaks[neighbor_idx].sigma_L * 0.99
            #             )

            softness = 0.8

            if sigma_L_fit > neighbor_L_median * max_width_ratio:
                excess = sigma_L_fit - neighbor_L_median * max_width_ratio
                sigma_L_fit = neighbor_L_median * max_width_ratio + excess * softness
            elif sigma_L_fit < neighbor_L_median / max_width_ratio:
                deficit = neighbor_L_median / max_width_ratio - sigma_L_fit
                sigma_L_fit = neighbor_L_median / max_width_ratio - deficit * softness

            if sigma_R_fit > neighbor_R_median * max_width_ratio:
                excess = sigma_R_fit - neighbor_R_median * max_width_ratio
                sigma_R_fit = neighbor_R_median * max_width_ratio + excess * softness
            elif sigma_R_fit < neighbor_R_median / max_width_ratio:
                deficit = neighbor_R_median / max_width_ratio - sigma_R_fit
                sigma_R_fit = neighbor_R_median / max_width_ratio - deficit * softness

    alpha_A = 0.4 if alpha > 0.4 else alpha
    q = spectrum.peaks[peak]
    previous = (q.A_refined, q.x0_refined, q.sigma_L, q.sigma_R, q.integral)
    spectrum.peaks[peak].sigma_L = float(
        alpha * sigma_L_fit + (1 - alpha) * spectrum.peaks[peak].sigma_L
    )
    spectrum.peaks[peak].sigma_R = float(
        alpha * sigma_R_fit + (1 - alpha) * spectrum.peaks[peak].sigma_R
    )
    x0_refined = float(alpha * x0_fit + (1 - alpha) * spectrum.peaks[peak].x0_refined)
    if abs(x0_refined - spectrum.peaks[peak].x0_init) > original_peak_width:
        x0_refined = spectrum.peaks[peak].x0_refined
    spectrum.peaks[peak].x0_refined = x0_refined

    if spectrum.peak_model == "lorentzian":
        A_new = integral_fit / (np.pi * (sigma_L_fit + sigma_R_fit))
    else:
        A_new = integral_fit / (np.sqrt(np.pi / 2.0) * (sigma_L_fit + sigma_R_fit))

    spectrum.peaks[peak].A_refined = float(
        alpha_A * A_new + (1 - alpha_A) * spectrum.peaks[peak].A_refined
    )
    spectrum.peaks[peak].integral = integral_fit

    if accept_only_improving:
        _accept_only_improving(spectrum, peak, previous, data_x, data_y)
