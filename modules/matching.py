from dataclasses import dataclass
import re
from turtle import mode
from typing import List, Tuple
import dearpygui.dearpygui as dpg
import numpy as np
from modules.rendercallback import RenderCallback, get_global_render_callback_ref
from modules.math import bi_gaussian, bi_Lorentzian
from modules.var import colors_list
from modules.dpg_style import peak_label_color
from modules.data_structures import MSData, MatchedWith, get_global_msdata_ref


def draw_mz_lines(sender, app_data, user_data: int):
    render_callback: RenderCallback = get_global_render_callback_ref()
    k = int(user_data)

    mw: int = dpg.get_value(f"molecular_weight_{k}")
    charges: int = dpg.get_value(f"charges_{k}")
    nb_peak_show: int = dpg.get_value(f"nb_peak_show_{k}")
    try:
        render_callback.spectrum.matching_data[k] = [mw, charges, nb_peak_show]
    except IndexError:
        new_matching_data = [[], [], [], [], [], [], [], [], [], []]
        for i in range(len(render_callback.spectrum.matching_data)):
            new_matching_data[i] = render_callback.spectrum.matching_data[i]
        render_callback.spectrum.matching_data = new_matching_data
        render_callback.spectrum.matching_data[k] = [mw, charges, nb_peak_show]

    mz_l = []
    z_l = []
    z_mz = []

    for i in range(nb_peak_show):
        z = charges - i
        if z == 0:
            break
        mz = (mw + z * 0.007) / z
        mz_l.append(mz)
        z_l.append(z)
        z_mz.append((z, mz))

    update_theorical_peak_table(k, mz_l, z_l)
    render_callback.mz_lines[k] = z_mz
    redraw_blocks()


def update_theorical_peak_table(k: int, mz_list: List[float], z_list):
    dpg.delete_item(f"theorical_peak_table_{k}", children_only=True)

    dpg.add_table_column(parent=f"theorical_peak_table_{k}")
    dpg.add_table_column(parent=f"theorical_peak_table_{k}")
    dpg.add_table_column(parent=f"theorical_peak_table_{k}")

    row = len(mz_list) // 3 + (len(mz_list) % 3 > 0)
    for r in range(0, row):
        with dpg.table_row(
            parent=f"theorical_peak_table_{k}", tag=f"theorical_peak_table_{k}_{r}"
        ):
            dpg.bind_item_theme(f"theorical_peak_table_{k}_{r}", "table_row_bg_grey")
            for n in range(0, 3):
                try:
                    dpg.add_text(f"{z_list[r*3+n]}+")
                except IndexError:
                    pass

        with dpg.table_row(parent=f"theorical_peak_table_{k}"):
            for n in range(0, 3):
                try:
                    dpg.add_text(f"{int(mz_list[r*3+n])}")
                except IndexError:
                    pass


def hidden_in_matching(spectrum: MSData, peak: int) -> bool:
    """Peaks ticked 'Bad' in the peak table are left out of matching when asked."""
    return bool(dpg.get_value("hide_bad_peaks")) and spectrum.peaks[peak].is_bad


def redraw_blocks():
    render_callback = get_global_render_callback_ref()
    show_matched_peaks(render_callback, clear=True)
    spectrum = get_global_msdata_ref()
    if len(spectrum.baseline_corrected) == 0:
        return
    try:
        block_height = max(spectrum.baseline_corrected[:, 1]) / 20
    except TypeError:
        return
    block_width = dpg.get_value("block_width")
    spectrum.block_width = block_width
    spectrum.matching_series = dpg.get_value("nb_peak_series")

    # Delete previous peaks and annotation
    for alias in dpg.get_aliases():
        if (
            alias.startswith("fitted_peak_matching")
            or alias.startswith("peak_annotation_matching")
            or alias.startswith(f"mz_lines_")
        ):
            dpg.delete_item(alias)

    for peak in spectrum.peaks:
        # Reset before skipping: a hidden peak must not keep an old match (the
        # integral ratios and the matched peak plots read matched_with)
        spectrum.peaks[peak].matched_with = []
        if not spectrum.peaks[peak].fitted or hidden_in_matching(spectrum, peak):
            continue

        if spectrum.peaks[peak].regression_fct[0] == 0:
            regression_0 = 0
        else:
            regression_0 = (
                -spectrum.peaks[peak].regression_fct[1]
                / spectrum.peaks[peak].regression_fct[0]
            )
        regression_x = regression_0 + spectrum.peaks[peak].sigma_L * 5
        regression_y = (
            spectrum.peaks[peak].regression_fct[0] * regression_x
            + spectrum.peaks[peak].regression_fct[1]
        )

        # non matched blocks
        dpg.draw_line(
            (regression_0, block_height),
            (regression_0, -25),
            parent="peak_matching_plot",
            color=(246, 32, 24, 128),
            thickness=block_width,
            tag=f"fitted_peak_matching_{peak}",
        )
        dpg.add_plot_annotation(
            label=f"Peak {peak}",
            default_value=(regression_0, 0),
            offset=(15, 15),
            color=peak_label_color(spectrum.peaks[peak].is_bad),
            clamped=False,
            parent="peak_matching_plot",
            tag=f"peak_annotation_matching_{peak}_gray",
        )

        for k in render_callback.mz_lines.keys():
            if int(k) >= dpg.get_value("nb_peak_series"):
                continue
            mz_lines = render_callback.mz_lines[k]
            color = colors_list[k]
            transparent_color = list(color)
            transparent_color.append(100)
            transparent_color = tuple(transparent_color)

            # find matched blocks
            for z_mz in mz_lines:
                if (
                    z_mz[1] > regression_0 - block_width / 2
                    and z_mz[1] < regression_0 + block_width / 2
                ):
                    dpg.delete_item(f"fitted_peak_matching_{peak}")
                    dpg.delete_item(f"mz_lines_{k}_{z_mz[0]}")
                    dpg.draw_line(
                        (z_mz[1], -50),
                        (z_mz[1], block_height * 1.5),
                        parent="peak_matching_plot",
                        tag=f"mz_lines_{k}_{z_mz[0]}",
                        color=colors_list[k],
                        thickness=1,
                    )
                    dpg.draw_line(
                        (regression_0, block_height),
                        (regression_0, -25),
                        parent="peak_matching_plot",
                        color=transparent_color,
                        thickness=block_width,
                        tag=f"fitted_peak_matching_{peak}",
                    )
                    if spectrum.peaks[peak].matched_with != []:
                        offset = len(spectrum.peaks[peak].matched_with) * 18
                    else:
                        offset = 0
                    spectrum.peaks[peak].matched_with.append(
                        MatchedWith(k, z_mz[0], z_mz[1] * z_mz[0], 1.0)
                    )
                    if not dpg.does_alias_exist(
                        f"peak_annotation_matching_{k}_{z_mz[0]}"
                    ):
                        x0_fit = spectrum.peaks[peak].x0_refined
                        y_mbg = spectrum.calculate_mbg(x0_fit)
                        dpg.add_plot_annotation(
                            label=f"{k}:{z_mz[0]}+",
                            default_value=(x0_fit, y_mbg),
                            offset=(15, -15 - offset),
                            color=color,
                            clamped=False,
                            parent="peak_matching_plot",
                            tag=f"peak_annotation_matching_{k}_{z_mz[0]}",
                        )

    # Draw the line annotations of unmatched peaks
    for k in render_callback.mz_lines.keys():
        if int(k) >= dpg.get_value("nb_peak_series"):
            continue

        mz_lines = render_callback.mz_lines[k]
        color = colors_list[k]
        for z_mz in mz_lines:
            line = z_mz[1]
            center_slice = render_callback.spectrum.baseline_corrected[:, 1][
                (render_callback.spectrum.baseline_corrected[:, 0] > line - 0.5)
                & (render_callback.spectrum.baseline_corrected[:, 0] < line + 0.5)
            ]
            max_y = max(center_slice) if len(center_slice) > 0 else 0

            if dpg.does_alias_exist(f"mz_lines_{k}_{z_mz[0]}"):
                dpg.delete_item(f"mz_lines_{k}_{z_mz[0]}")

            dpg.draw_line(
                (line, -50),
                (line, max_y),
                parent="peak_matching_plot",
                tag=f"mz_lines_{k}_{z_mz[0]}",
                color=colors_list[k],
                thickness=1,
            )
            if not dpg.does_alias_exist(f"peak_annotation_matching_{k}_{z_mz[0]}"):
                dpg.add_plot_annotation(
                    label=f"{k}:{z_mz[0]}+",
                    default_value=(z_mz[1], 0),
                    offset=(-15, -15 - k * 15),
                    color=colors_list[k],
                    clamped=False,
                    parent="peak_matching_plot",
                    tag=f"peak_annotation_matching_not_{k}_{z_mz[1]}",
                )

    matching_quality(render_callback)
    check_mass_difference(render_callback)


def show_projection(user_data: RenderCallback):
    redraw_blocks()


@dataclass
class MatchQualityMetrics:
    peaks: int
    rmsd: float
    score: float
    widths: Tuple[float, float]


def calculate_quality_score(
    render_callback: RenderCallback, k: int, shift=0
) -> MatchQualityMetrics:
    spectrum = get_global_msdata_ref()
    mz_lines = render_callback.mz_lines[k]
    block_width = dpg.get_value("block_width")
    squares = []
    widths = []
    integrals = []

    for z_mz in mz_lines:
        for peak in spectrum.peaks:
            if not spectrum.peaks[peak].fitted or hidden_in_matching(spectrum, peak):
                continue

            integrals.append(spectrum.peaks[peak].integral)
            try:
                regression_0 = (
                    -spectrum.peaks[peak].regression_fct[1]
                    / spectrum.peaks[peak].regression_fct[0]
                )
            except ZeroDivisionError:
                regression_0 = 0
            z_mz_shifted = z_mz[1] + (shift / z_mz[0])
            if (
                z_mz_shifted > regression_0 - block_width / 2
                and z_mz_shifted < regression_0 + block_width / 2
            ):
                square_distance = (z_mz_shifted - regression_0) ** 2
                squares.append(square_distance)
                widths.append(
                    spectrum.peaks[peak].sigma_L + spectrum.peaks[peak].sigma_R
                )

    peaks = len(squares)

    if peaks > 0:
        rmsd = np.sqrt(np.mean(squares))
        score = (1 / rmsd) * 100 + (len(squares) * 20)

        return MatchQualityMetrics(
            peaks=peaks,
            rmsd=rmsd,
            score=score,
            widths=(float(np.median(widths)), float(np.std(widths))),
        )
    else:
        return MatchQualityMetrics(
            peaks=0,
            rmsd=0,
            score=0,
            widths=(0, 0),
        )


def series_integral_quality(k: int):
    spectrum = get_global_msdata_ref()
    render_callback = get_global_render_callback_ref()
    mz_lines = render_callback.mz_lines[k]
    integrals = []

    for z_mz in mz_lines:
        for peak in spectrum.peaks:
            if not spectrum.peaks[peak].fitted or hidden_in_matching(spectrum, peak):
                continue
            matched = spectrum.peaks[peak].matched_with
            for m in matched:
                if m.set == k and m.charge == z_mz[0]:
                    integrals.append(spectrum.peaks[peak].integral)


def analyze_series(k: int):
    """Analyze if matched peak integrals follow a bi-gaussian distribution pattern"""
    spectrum = get_global_msdata_ref()
    render_callback = get_global_render_callback_ref()
    mz_lines = render_callback.mz_lines[k]

    # Collect charge states and their corresponding integrals
    charge_integral_pairs = []

    for z_mz in mz_lines:
        for peak in spectrum.peaks:
            if not spectrum.peaks[peak].fitted or hidden_in_matching(spectrum, peak):
                continue

            matched = spectrum.peaks[peak].matched_with
            for m in matched:
                if m.set == k and m.charge == z_mz[0]:
                    charge_integral_pairs.append(
                        (z_mz[0], spectrum.peaks[peak].integral)
                    )

    if len(charge_integral_pairs) < 3:
        print(
            f"Series {k}: Not enough points for distribution analysis (need at least 3)"
        )
        return

    # Sort by charge state
    charge_integral_pairs.sort(key=lambda x: x[0])
    charges = np.array([x[0] for x in charge_integral_pairs])
    integrals = np.array([x[1] for x in charge_integral_pairs])

    # print(f"\nSeries {k} - Charge State Distribution Analysis:")
    # print(f"Charges: {charges}")
    # print(f"Integrals: {integrals}")

    # Find the peak (maximum intensity)
    max_idx = np.argmax(integrals)
    max_charge = charges[max_idx]
    max_integral = integrals[max_idx]

    # Check for unimodality
    is_unimodal = True
    mode_count = 0

    if integrals[0] > integrals[1]:
        mode_count += 1
    if integrals[-1] > integrals[-2]:
        mode_count += 1

    # Count local maxima (peaks in the distribution)
    for i in range(1, len(integrals) - 1):
        if integrals[i] > integrals[i - 1] and integrals[i] > integrals[i + 1]:
            mode_count += 1
            if mode_count > 1:
                is_unimodal = False

    if not is_unimodal:
        print(f"WARNING: Series {k} is not unimodal")
    return mode_count


def matching_quality(render_callback: RenderCallback):
    for k in render_callback.mz_lines.keys():
        quality_metrics = calculate_quality_score(render_callback, k)

        if quality_metrics.peaks > 1:
            dpg.set_value(
                f"series_quality_score_{k}",
                f"{quality_metrics.peaks} Peaks - Score: {quality_metrics.score:.2f}",
            )
            dpg.set_value(
                f"series_quality_rmsd_{k}", f"RMSD: {quality_metrics.rmsd:.2f}"
            )
            dpg.set_value(
                f"series_quality_width_{k}",
                f"mean Width: {quality_metrics.widths[0]:.2f} ± {quality_metrics.widths[1]:.2f}",
            )

        elif quality_metrics.peaks == 1:
            dpg.set_value(f"series_quality_rmsd_{k}", f"Only 1 peak match")
            dpg.set_value(f"series_quality_rmsd_{k}", f"RMSD:")
            dpg.set_value(
                f"series_quality_width_{k}",
                f"mean Width: {quality_metrics.widths[0]:.2f}",
            )
        else:
            dpg.set_value(f"series_quality_rmsd_{k}", f"No matching peaks")
            dpg.set_value(f"series_quality_rmsd_{k}", f"RMSD centers:")
            dpg.set_value(f"series_quality_width_{k}", f"mean Width:")

        modes = analyze_series(int(k))
        if modes == 1:
            dpg.set_value(f"series_quality_modes_{k}", f"Unimodal distribution")
        else:
            dpg.set_value(
                f"series_quality_modes_{k}", f"Warning! {modes} modes detected"
            )


def refine_matching(sender, app_data, user_data: int):
    refinement_width = dpg.get_value("refinement_width")
    k = user_data
    local_score = []
    render_callback = get_global_render_callback_ref()

    for shift in range(-refinement_width, refinement_width + 1, 1):
        quality_metrics = calculate_quality_score(render_callback, k, shift)
        local_score.append(
            (shift, quality_metrics.peaks, quality_metrics.rmsd, quality_metrics.score)
        )

    max_peaks = max(local_score, key=lambda x: x[1])[1]
    best_rmsd = min([x for x in local_score if x[1] == max_peaks], key=lambda x: x[2])
    new_mw = dpg.get_value(f"molecular_weight_{k}") + best_rmsd[0]

    dpg.set_value(f"molecular_weight_{k}", new_mw)
    draw_mz_lines(None, None, user_data=k)
    analyze_series(k)


def check_mass_difference(render_callback: RenderCallback):
    for k in range(0, 10):
        mw_set_i = dpg.get_value(f"compare_set_{k}")
        mw_set1 = dpg.get_value(f"molecular_weight_{mw_set_i}")
        mw = dpg.get_value(f"molecular_weight_{k}")
        diff = mw - mw_set1
        dpg.set_value(f"MW_diff_{k}", f"d{diff}")
    check_integral_ratio()


# def check_integral_ratio():
#     for k in range(0, 10):
#         compare_to = int(dpg.get_value(f"compare_set_{k}"))
#         integral_set_to = []
#         integral_set_k = []
#         spectrum = get_global_msdata_ref()

#         for peak in spectrum.peaks:
#             if not spectrum.peaks[peak].fitted:
#                 continue
#             matched = spectrum.peaks[peak].matched_with
#             if matched == []:
#                 continue

#             if matched[0].set == compare_to:
#                 integral_set_to.append(
#                     [
#                         matched[0].charge,
#                         spectrum.peaks[peak].integral,
#                         spectrum.peaks[peak].se_integral,
#                     ]
#                 )
#             if matched[0].set == k:
#                 integral_set_k.append(
#                     [
#                         matched[0].charge,
#                         spectrum.peaks[peak].integral,
#                         spectrum.peaks[peak].se_integral,
#                     ]
#                 )

#         ratios = []
#         zs = []

#         for z1, int_to, se_to in integral_set_to:
#             for z2, int_k, se_k in integral_set_k:
#                 if z1 == z2:
#                     # Skip if denominator is zero or either integral is non-positive
#                     if int_to <= 0 or int_k <= 0:
#                         continue

#                     ratio = int_k / int_to

#                     # Calculate ratio SE using error propagation if both SEs are available and non-zero
#                     if se_to > 0 and se_k > 0:
#                         ratio_se = ratio * np.sqrt(
#                             (se_to / int_to) ** 2 + (se_k / int_k) ** 2
#                         )
#                     else:
#                         ratio_se = 0  # Mark as no SE available

#                     ratios.append((ratio, ratio_se))
#                     zs.append(z1)

#         if len(ratios) > 0:
#             # Extract ratio values and their standard errors
#             ratio_values = np.array([r[0] for r in ratios])
#             ratio_ses = np.array([r[1] for r in ratios])

#             # Check for positive values before taking log
#             if np.all(ratio_values > 0):
#                 geometric_mean = np.exp(np.mean(np.log(ratio_values)))

#                 # Calculate geometric mean SE
#                 if np.any(ratio_ses == 0):
#                     # If any SE is missing, use sample standard error of log-transformed values
#                     geometric_mean_se = (
#                         np.std(np.log(ratio_values), ddof=1)
#                         * geometric_mean
#                         / np.sqrt(len(ratio_values))
#                     )
#                 else:

#                     # Combine both sources of uncertainty:
#                     # 1. Measurement uncertainty (from error propagation)
#                     # 2. Sample variance (between-ratio variability)

#                     # Measurement uncertainty contribution
#                     relative_ses = ratio_ses / ratio_values
#                     measurement_var = np.mean(relative_ses**2)

#                     # Sample variance contribution (variability between ratios)
#                     sample_var = np.var(np.log(ratio_values), ddof=1) / len(
#                         ratio_values
#                     )

#                     # Total variance is the sum (assuming independence)
#                     total_var = measurement_var + sample_var

#                     geometric_mean_se = geometric_mean * np.sqrt(total_var)

#                 dpg.set_value(
#                     f"Integral_ratio_{k}",
#                     f"{geometric_mean:.2f} ± {geometric_mean_se:.2f} using {len(zs)} peaks",
#                 )
#             else:
#                 print(f"Warning: Non-positive ratio values found for set {k}")
#                 dpg.set_value(f"Integral_ratio_{k}", "Error: Invalid ratios")


def check_integral_ratio():

    max_set = dpg.get_value("nb_peak_series")
    selected_sets = []
    for k in range(0, max_set):
        if dpg.get_value(f"compare_checkbox_{k}"):
            selected_sets.append(k)
    spectrum = get_global_msdata_ref()

    def matches(peak):  # older files store plain numbers here: skip them
        return [m for m in spectrum.peaks[peak].matched_with if isinstance(m, MatchedWith)]

    ions_lists = [[]]
    for peak_set in selected_sets:
        ions_list = []
        for peak in spectrum.peaks:
            for match in matches(peak):
                if match.set == peak_set:
                    ions_list.append(match.charge)
        ions_lists.append(ions_list)

    useful_ions = set(ions_lists[1]) if len(ions_lists) > 1 else set()
    for ions in ions_lists[2:]:
        useful_ions &= set(ions)

    ions = sorted(useful_ions)
    if not ions:
        for peak_set in selected_sets:
            dpg.set_value(f"Integral_ratio_{peak_set}", "No charge state in common")
        return

    # Peaks of each set at each charge state (ion)
    set_ion_peaks: dict[tuple[int, int], list[int]] = {
        (k, ion): [] for k in selected_sets for ion in ions
    }
    for peak in spectrum.peaks:
        for match in matches(peak):
            if match.set in selected_sets and match.charge in useful_ions:
                set_ion_peaks[(match.set, match.charge)].append(peak)

    shares, errors = closed_geometric_mean_shares(spectrum, selected_sets, ions, set_ion_peaks)
    for k, peak_set in enumerate(selected_sets):
        dpg.set_value(
            f"Integral_ratio_{peak_set}",
            f"{shares[k]:.2f} ± {errors[k]:.3f} using {len(ions)} ions",
        )


def closed_geometric_mean_shares(
    spectrum: MSData,
    sets: list[int],
    ions: list[int],
    set_ion_peaks: dict[tuple[int, int], list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Relative abundance of each set: share of the charge-state total, averaged over
    charge states as a geometric mean, then closed (divided by the sum) so the
    shares add up to 1 (centre of compositional data).

    Standard errors combine
    - the measurement error, propagated from the peak integrals with their full
      covariance (Laplace correlations, see MSData.integral_covariance): overlapping
      species are anticorrelated, and each species is part of the total;
    - the scatter between charge states (sample covariance of the log shares).
    Both are propagated through the closure.
    """
    n, K = len(ions), len(sets)
    peaks = sorted({p for ps in set_ion_peaks.values() for p in ps})
    col = {p: i for i, p in enumerate(peaks)}
    integral = np.array([spectrum.peaks[p].integral for p in peaks])

    # Log shares L[z, k] and gradient of their mean m_k w.r.t. the peak integrals
    L = np.zeros((n, K))
    G = np.zeros((K, len(peaks)))
    for z, ion in enumerate(ions):
        set_totals = [sum(integral[col[p]] for p in set_ion_peaks[(k, ion)]) for k in sets]
        total = sum(set_totals)
        for k, peak_set in enumerate(sets):
            L[z, k] = np.log(set_totals[k] / total)
            for p in set_ion_peaks[(peak_set, ion)]:
                G[k, col[p]] += 1 / (n * set_totals[k])
            for other in sets:
                for p in set_ion_peaks[(other, ion)]:
                    G[k, col[p]] -= 1 / (n * total)

    geo = np.exp(L.mean(axis=0))
    shares = geo / geo.sum()
    # Closure s = geo / sum(geo): ds = diag(s) (I - 1 s^T) dm
    J = np.diag(shares) @ (np.eye(K) - np.outer(np.ones(K), shares))

    cov_m = G @ spectrum.integral_covariance(peaks) @ G.T
    if n > 1:
        cov_m = cov_m + np.atleast_2d(np.cov(L, rowvar=False, ddof=1)) / n
    cov_s = J @ cov_m @ J.T
    return shares, np.sqrt(np.clip(np.diag(cov_s), 0, None))


def print_to_terminal():
    spectrum = get_global_msdata_ref()
    print("Matched Peaks:")
    if not spectrum.peaks:
        print("No peaks found.")
        return
    # Collect peaks grouped by the first value of matched_with
    grouped_peaks = {}
    for peak in spectrum.peaks:
        if not spectrum.peaks[peak].fitted:
            continue
        matched = spectrum.peaks[peak].matched_with
        if matched == []:
            continue

        for m in matched:
            group_key = m.set
            if group_key not in grouped_peaks:
                grouped_peaks[group_key] = []
            base = (
                -spectrum.peaks[peak].regression_fct[1]
                / spectrum.peaks[peak].regression_fct[0]
            )
            grouped_peaks[group_key].append(
                {
                    "peak": peak,
                    "x0": spectrum.peaks[peak].x0_refined,
                    "integral": spectrum.peaks[peak].integral,
                    "se_integral": spectrum.peaks[peak].se_integral,
                    "matched_with": m,
                    "base": base,
                }
            )

    for group, peaks in grouped_peaks.items():
        print(f"Mass group {group}: M: {peaks[0]['matched_with'].mw:.2f}")
        ordered_peaks = sorted(
            peaks, key=lambda x: x["matched_with"].charge, reverse=True
        )
        print("Peak:\t\tz:\t\tM/Z:\t\tx0:\t\tBase:\t\tIntegral:\t\tIntegralError")
        for p in ordered_peaks:
            print(
                f"{p['peak']}\t\t{p['matched_with'].charge}\t\t{p['matched_with'].mw/p['matched_with'].charge:.2f}\t\t{p['x0']:.2f}\t\t{p['base']:.2f}\t\t{p['integral']:.2f}\t\t{p['se_integral']:.2f}"
            )


def show_matched_peaks(user_data: RenderCallback, clear=False):
    draw = True if not clear else False
    for alias in dpg.get_aliases():
        if alias.startswith("Matched_biG_plot3"):
            dpg.delete_item(alias)
            draw = False
    if draw:
        dpg.set_item_label("show_matched_peaks_button", "Hide matched peaks")
        draw_biG_matched_peaks(user_data)
    else:
        dpg.set_item_label("show_matched_peaks_button", "Show matched peaks")
        return


def draw_biG_matched_peaks(user_data: RenderCallback):
    spectrum = get_global_msdata_ref()

    for alias in dpg.get_aliases():
        if alias.startswith("Matched_biG_"):
            dpg.delete_item(alias)

    for k in user_data.mz_lines.keys():
        if int(k) >= dpg.get_value("nb_peak_series"):
            continue

        shade_color = list(colors_list[k])
        shade_color.append(100)

        with dpg.theme(tag=f"Matched_biG_theme_{k}"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 3, category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line, colors_list[k], category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_Fill, shade_color, category=dpg.mvThemeCat_Plots
                )

    for peak in spectrum.peaks:
        if not spectrum.peaks[peak].fitted:
            continue
        matched = spectrum.peaks[peak].matched_with
        if matched == []:
            continue

        x0 = spectrum.peaks[peak].x0_refined
        A = spectrum.peaks[peak].A_refined
        sigma_L = spectrum.peaks[peak].sigma_L
        sigma_R = spectrum.peaks[peak].sigma_R
        set = matched[0].set

        x_data = np.linspace(
            x0 - 5 * sigma_L,
            x0 + 5 * sigma_R,
            100,
        )
        if spectrum.peak_model == "lorentzian":
            y_data = bi_Lorentzian(x_data, A, x0, sigma_L, sigma_R)
        else:
            y_data = bi_gaussian(x_data, A, x0, sigma_L, sigma_R)

        dpg.add_shade_series(
            x_data.tolist(),
            y_data.tolist(),
            label=f"Peak {peak} area",
            parent="y_axis_plot3",
            tag=f"Matched_biG_plot3_shade3_{peak}",
            show=True,
        )
        dpg.bind_item_theme(
            f"Matched_biG_plot3_shade3_{peak}", f"Matched_biG_theme_{set}"
        )
        dpg.add_line_series(
            x_data.tolist(),
            y_data.tolist(),
            label=f"Peak {peak} matched",
            parent="y_axis_plot3",
            tag=f"Matched_biG_plot3_{peak}",
            show=True,
        )
        dpg.bind_item_theme(f"Matched_biG_plot3_{peak}", f"Matched_biG_theme_{set}")

        if len(matched) > 1:
            for extra_match in matched[1:]:
                dpg.add_shade_series(
                    x_data.tolist(),
                    y_data.tolist(),
                    label=f"Peak {peak} area",
                    parent="y_axis_plot3",
                    tag=f"Matched_biG_plot3_shade3_{peak}_extra{extra_match.set}",
                    show=True,
                )
                dpg.bind_item_theme(
                    f"Matched_biG_plot3_shade3_{peak}_extra{extra_match.set}",
                    f"Matched_biG_theme_{extra_match.set}",
                )
