from _collections_abc import dict_keys
from typing import List, Literal, Optional
from pybaselines import Baseline
from scipy.signal import medfilt
import numpy as np
from dataclasses import dataclass
from modules.math import multi_bi_gaussian, multi_bi_Lorentzian
from typing import Dict, Tuple
from dataclasses import MISSING, field
import pickle
import pandas as pd
from whittaker_eilers import WhittakerSmoother


class MSData:
    def __init__(self):
        self.peak_model: Literal["gaussian", "lorentzian"] = "gaussian"
        self.original_data: np.ndarray = np.empty((0, 2))
        self.working_data: np.ndarray = np.empty((0, 2))
        self.baseline: np.ndarray = np.empty((0, 2))
        self.baseline_corrected: np.ndarray = np.empty((0, 2))
        self.peaks: Dict[int, peak_params] = {}
        self.baseline_toggle = False
        self.baseline_window = 1000
        self.baseline_need_update = False
        self.matching_data = [[], [], [], [], [], [], [], [], [], []]
        self.peak_detection_parameters = {
            "threshold": 100,
            "width": 10,
            "distance": 10,
            "use_2nd_derivative": True,
        }
        self.smoothing_window = 10
        self.block_width = 25
        self.matching_series = 3
        # Correlation matrix of peak integrals from the Laplace analysis
        # {"peaks": [peak ids], "matrix": ndarray}, None when not computed
        self.laplace_integral_corr: Optional[dict] = None
        # FitSummary of the last MBG fit, None when no fit was run
        self.fit_summary: Optional[FitSummary] = None

    def import_csv(self, path: str):
        return True

    def initialise_dataframe(self, data: pd.DataFrame):
        data = data.dropna()
        # Ensure original_data is a 2D numpy array as expected
        if isinstance(data, pd.DataFrame) and data.shape[1] == 2:
            self.original_data = data.to_numpy()
        else:
            self.original_data = np.empty((0, 2))
        self.working_data = self.original_data
        self.fit_summary = None
        self.correct_baseline(0)

    def clip_data(self, L_clip: int, R_clip: int):
        if len(self.original_data) <= 2:
            return
        self.working_data = self.original_data[
            (self.original_data[:, 0] > L_clip) & (self.original_data[:, 0] < R_clip)
        ]

    def get_filtered_data(self, window_length, polyorder=2, baseline=False):
        self.smoothing_window = window_length
        if len(self.working_data[:, 1]) >= window_length:
            window_length = window_length - 1
        if window_length % 2 == 0:
            window_length -= 1  # Ensure window_length is odd

        if baseline:
            self.ensure_baseline_consistent()

        whittaker_smoother = WhittakerSmoother(
            lmbda=window_length,
            order=2,
            x_input=self.working_data[:, 0].tolist(),
            data_length=len(self.working_data[:, 1]),
        )

        data = (
            self.working_data[:, 1] if not baseline else self.baseline_corrected[:, 1]
        )
        filtered = whittaker_smoother.smooth(data.tolist())

        return filtered

    def get_smoothing_window(self):
        whittaker_smoother = WhittakerSmoother(
            lmbda=1000,
            order=2,
            x_input=self.working_data[:, 0].tolist(),
            data_length=len(self.working_data[:, 1]),
        )
        optimal_smooth = whittaker_smoother.smooth_optimal(
            self.working_data[:, 1].tolist()
        )
        return float(optimal_smooth.get_optimal().get_lambda())

    def get_2nd_derivative(self, window_length, polyorder=2) -> Optional[List[float]]:
        if len(self.working_data[:, 1]) >= window_length:
            window_length = window_length - 1
        if window_length % 2 == 0:
            window_length -= 1  # Ensure window_length is odd

        whittaker_smoother = WhittakerSmoother(
            lmbda=window_length,
            order=2,
            x_input=self.working_data[:, 0].tolist(),
            data_length=len(self.working_data[:, 1]),
        )

        try:
            filtered = whittaker_smoother.smooth(self.working_data[:, 1].tolist())

            # Calculate 2nd derivative directly
            derivative2nd = np.gradient(
                np.gradient(filtered, self.working_data[:, 0]), self.working_data[:, 0]
            )

            # Apply smoothing and invert sign
            derivative2nd_filtered = (
                np.array(whittaker_smoother.smooth(derivative2nd)) * -1
            )

            # Normalize and clip
            shift = int(np.max(self.working_data[:, 1]) / 2)
            max_val = np.max(derivative2nd_filtered)
            if max_val != 0:
                derivative2nd_filtered = derivative2nd_filtered * (shift / max_val)
            derivative2nd_filtered = np.clip(derivative2nd_filtered, 0, None)
        except:
            return

        return derivative2nd_filtered.tolist()

    def get_average_spike_length(self, min_spike_width=1, use_baseline_corrected=True):
        if len(self.working_data) <= 2:
            return 0, 0, []

        data_to_analyze = self.get_2nd_derivative(self.smoothing_window)

        avg_length, spike_info = analyze_spike_lengths(
            data_to_analyze, min_spike_width=min_spike_width
        )
        return avg_length, len(spike_info), spike_info

    def correct_baseline(self, window):
        if len(self.original_data) <= 2:
            return
        self.baseline_window = window
        if not self.baseline_toggle:
            self.baseline_corrected = self.working_data
            self.baseline = np.column_stack(
                (self.working_data[:, 0], [0] * len(self.working_data))
            )
            self.baseline_need_update = False
            return

        baseline_fitter = Baseline(x_data=self.working_data[:, 0])
        try:
            bkg_4, params_4 = baseline_fitter.snip(
                self.working_data[:, 1],
                max_half_window=window,
                decreasing=True,
                smooth_half_window=3,
            )
            self.baseline = np.column_stack((self.working_data[:, 0], bkg_4))
            self.baseline_corrected = np.column_stack(
                (self.working_data[:, 0], self.working_data[:, 1] - bkg_4)
            )
            self.baseline_need_update = False
        except:
            return

    def ensure_baseline_consistent(self) -> None:
        """
        Recompute the baseline when it no longer matches working_data (clipping or
        loading a session changes working_data before the render loop catches up).
        """
        if len(self.baseline_corrected) == len(self.working_data) and np.array_equal(
            self.baseline_corrected[:, 0], self.working_data[:, 0]
        ):
            return
        self.correct_baseline(self.baseline_window)

    def fft_filter_data(self, cutoff_frequency=0.1):
        if len(self.working_data) <= 2:
            return
        y = self.working_data[:, 1]
        n = len(y)
        y_fft = np.fft.fft(y)
        sampling_rate = self.guess_sampling_rate()
        if sampling_rate is None:
            print("Unable to estimate sampling rate for FFT.")
            return
        frequencies = np.fft.fftfreq(n, d=sampling_rate)
        y_fft[np.abs(frequencies) > cutoff_frequency] = 0
        y_filtered = np.fft.ifft(y_fft).real
        self.baseline_corrected = np.column_stack((self.working_data[:, 0], y_filtered))
        return y_fft.tolist()

    def guess_sampling_rate(self) -> Optional[float]:
        if len(self.original_data) <= 2:
            return
        sampling_rate = float(np.mean(np.diff(self.working_data[:, 0])))
        return sampling_rate

    def request_baseline_update(self) -> None:
        if len(self.original_data) <= 2:
            return
        self.baseline_need_update = True

    def calculate_mbg(self, data_x: np.ndarray | float, fitting=False) -> np.ndarray:
        mbg_params = []

        for peak in self.peaks:
            if self.peaks[peak].do_not_fit:
                continue

            if fitting or self.peaks[peak].fitted:
                mbg_params.append(self.peaks[peak].A_refined)
                mbg_params.append(self.peaks[peak].x0_refined)
                mbg_params.append(self.peaks[peak].sigma_L)
                mbg_params.append(self.peaks[peak].sigma_R)

        if self.peak_model == "lorentzian":
            mbg = multi_bi_Lorentzian(data_x, *mbg_params)
        else:
            mbg = multi_bi_gaussian(data_x, *mbg_params)
        return mbg

    def integral_covariance(self, peaks: List[int]) -> np.ndarray:
        """
        Covariance matrix of the integrals of `peaks`: Laplace correlation scaled by
        each peak's final standard error (se_integral). Peaks without a correlation
        are treated as uncorrelated.
        """
        se = np.array([max(self.peaks[p].se_integral, 0.0) for p in peaks])
        corr = np.eye(len(peaks))
        stored = self.laplace_integral_corr
        if stored is not None:
            index = {peak: i for i, peak in enumerate(stored["peaks"])}
            for a, pa in enumerate(peaks):
                for b, pb in enumerate(peaks):
                    if a != b and pa in index and pb in index:
                        corr[a, b] = stored["matrix"][index[pa], index[pb]]
        return corr * np.outer(se, se)

    def save_to_file(self, path: str):
        """Save the entire MSData object to a file using pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def get_packed_parameters(
        self, integral=False, ordered=False
    ) -> Tuple[List[float], List[int]]:
        packed_params = []
        ordered_peaks_list: List[int] = (
            sorted(self.peaks.keys(), key=lambda peak: self.peaks[peak].x0_init)
            if ordered
            else list(self.peaks.keys())
        )

        packed_peaks: List[int] = []
        for peak in ordered_peaks_list:
            if self.peaks[peak].do_not_fit:
                continue
            packed_peaks.append(peak)

            if integral:
                packed_params.append(self.peaks[peak].integral)
            else:
                packed_params.append(self.peaks[peak].A_refined)
            packed_params.append(self.peaks[peak].x0_refined)
            packed_params.append(self.peaks[peak].sigma_L)
            packed_params.append(self.peaks[peak].sigma_R)
        return packed_params, packed_peaks

    # @staticmethod
    # def load_from_file(path: str) -> "MSData":
    #    """Load an MSData object from a file using pickle."""
    #    with open(path, 'rb') as f:
    #        data: MSData = pickle.load(f)
    #    return data

    def load_from_file(self, path: str):
        with open(path, "rb") as f:
            new_data: MSData = pickle.load(f)
        # Files saved by older versions miss newer attributes: fall back to defaults
        for name, default in MSData().__dict__.items():
            setattr(self, name, getattr(new_data, name, default))
        self.peaks = {
            peak: upgrade_peak_params(params) for peak, params in self.peaks.items()
        }
        self.fit_summary = upgrade_fit_summary(self.fit_summary)
        # Sessions can be saved with a baseline computed on a different clipping
        self.ensure_baseline_consistent()


ms_data_global_ref = MSData()


def get_global_msdata_ref() -> MSData:
    return ms_data_global_ref


@dataclass
class FitQualityPeakMetrics:
    snr: float
    peak_rmse: float
    relative_error: float
    r_squared: float
    # Mean squared residual over the peak's region / noise variance: ~1 = fitted to
    # within the noise; NaN when not computed (older files)
    residual_noise: float = float("nan")


@dataclass
class FitSummary:
    # Which stopping criterion ended the fit: "max_iter", "user", or the criteria met
    # joined by "+" among "theta", "r2" and "flat" (e.g. "theta+r2")
    stop_reason: str
    iterations_done: int
    max_iterations: int
    delta_theta: float
    theta_threshold: float
    r_squared: float
    r_squared_threshold: float
    weighted_rmse: float
    chi_squared_reduced: float
    signal_to_noise: float
    aic: float
    bic: float
    residual_autocorr: float
    time_taken: float
    # Tangent of the R² evolution: relative residual drop over the last window
    r2_flat_gain: float = float("nan")
    r2_flat_threshold: float = 0.0


@dataclass
class MatchedWith:
    set: int
    charge: int
    mw: float
    ratio: float


# Relative error above which a peak counts as bad unless the user decided otherwise
HIGH_ERROR_THRESHOLD = 0.15


@dataclass
class peak_params:
    A_init: float = 0
    x0_init: float = 0
    width: float = 0
    A_refined: float = 0
    x0_refined: float = 0
    sigma_L: float = 0
    sigma_L_init: float = 0
    sigma_R: float = 0
    sigma_R_init: float = 0
    se_A: float = 0
    se_x0: float = 0
    se_sigma_L: float = 0
    se_sigma_R: float = 0
    x0_ema: Optional[float] = None
    sigma_L_ema: Optional[float] = None
    sigma_R_ema: Optional[float] = None
    A_ema: Optional[float] = None
    sampling_rate: float = 0.0
    fitted: bool = False
    integral: float = 0
    se_integral: float = 0
    start_end: Tuple[float, float] = (0.0, 0.0)
    regression_fct: Tuple[float, float] = (1.0, 0.0)
    se_base: float = 0.0
    do_not_fit: bool = False
    user_added: bool = False
    matched_with: List[MatchedWith] = field(default_factory=lambda: [])
    fit_quality: FitQualityPeakMetrics = field(
        default_factory=lambda: FitQualityPeakMetrics(0.0, 0.0, 1.0, 0.0)
    )
    # Bad peak (hidden from matching when 'Hide bad peaks' is on): None = automatic
    # (relative error above HIGH_ERROR_THRESHOLD), True / False = set by the user.
    # Reset to automatic by a new fit
    marked_bad: Optional[bool] = None

    @property
    def is_bad(self) -> bool:
        # bool(): the error is often a numpy float, and dearpygui rejects numpy bools
        if self.marked_bad is not None:
            return bool(self.marked_bad)
        return bool(self.fit_quality.relative_error > HIGH_ERROR_THRESHOLD)
    # Components of se_integral / se_x0 from the error analysis, -1 when not computed.
    # se_integral = max(bootstrap, Laplace, restarts)
    se_integral_bootstrap: float = -1.0
    se_integral_restart: float = -1.0
    se_x0_bootstrap: float = -1.0
    se_x0_restart: float = -1.0
    # Laplace covariance analysis, -1 when not computed
    laplace_se_integral: float = -1.0  # relative standard error of the integral
    laplace_se_x0: float = -1.0
    # Integral correlation with the left / right m/z neighbours (-1 peak: none)
    laplace_corr_left: float = 0.0
    laplace_corr_right: float = 0.0
    laplace_left_peak: int = -1
    laplace_right_peak: int = -1
    laplace_area_corr: float = 0.0  # the stronger anticorrelation of the two
    laplace_area_peak: int = -1  # ... and that neighbour
    laplace_message: str = ""


def upgrade_peak_params(old: peak_params) -> peak_params:
    """Rebuild a peak loaded from an older file so every current field exists."""
    peak = peak_params()
    for name in peak.__dataclass_fields__:
        if name in old.__dict__:
            setattr(peak, name, old.__dict__[name])
    quality = getattr(peak.fit_quality, "__dict__", {})
    peak.fit_quality = FitQualityPeakMetrics(
        snr=quality.get("snr", 0.0),
        peak_rmse=quality.get("peak_rmse", 0.0),
        relative_error=quality.get("relative_error", 1.0),
        r_squared=quality.get("r_squared", 0.0),
        residual_noise=quality.get("residual_noise", float("nan")),
    )
    return peak


def upgrade_fit_summary(old: Optional[FitSummary]) -> Optional[FitSummary]:
    """
    Drop a summary loaded from an older file if it misses current fields; fields
    with a default (added later) are filled with it.
    """
    if old is None:
        return None
    saved = getattr(old, "__dict__", {})
    fields = FitSummary.__dataclass_fields__
    required = [
        name
        for name, f in fields.items()
        if f.default is MISSING and f.default_factory is MISSING
    ]
    if not all(name in saved for name in required):
        return None
    return FitSummary(**{name: saved[name] for name in fields if name in saved})


def fft_filter_data(y_data, cutoff_frequency=0.1, sampling_rate=1.0):
    if len(y_data) <= 2:
        return
    y = y_data
    n = len(y)
    y_fft = np.fft.fft(y)
    frequencies = np.fft.fftfreq(n, d=sampling_rate)
    y_fft[np.abs(frequencies) > cutoff_frequency] = 0
    y_filtered = np.fft.ifft(y_fft).real
    return y_filtered.tolist()


if __name__ == "__main__":
    ms = MSData()
    ms.import_csv(rf"D:\MassSpec\Um_2-1_1x.csv")
    # ms.clip_data(10000, 12000)
    ms.baseline_toggle = True
    ms.correct_baseline(40)
    ms.guess_sampling_rate()
    print(ms.working_data[:, 0])


def remove_spikes_median(y, kernel_len=11):
    """
    Median filter. kernel_len must be odd.
    - kernel_len ~ a bit larger than spike width in samples.
    """
    if kernel_len % 2 == 0:
        kernel_len += 1
    return medfilt(y, kernel_len)


def analyze_spike_lengths(y_data, threshold_factor=2.0, min_spike_width=1):
    """
    Analyze spikes in data and return their average length.

    Parameters:
    - y_data: 1D array of intensity values
    - threshold_factor: Multiplier for standard deviation to define spike threshold
    - min_spike_width: Minimum width in samples to consider as a spike

    Returns:
    - average_spike_length: Average length of detected spikes
    - spike_info: List of tuples (start_idx, end_idx, length) for each spike
    """
    if len(y_data) <= 2:
        return 0, []

    y = np.array(y_data)

    # Calculate threshold for spike detection
    median_val = np.median(y)
    std_val = np.std(y)
    threshold = median_val / 2

    # Find points above threshold
    above_threshold = y > threshold

    # Find spike boundaries
    spike_starts = []
    spike_ends = []

    in_spike = False
    start_idx = 0

    for i, is_spike in enumerate(above_threshold):
        if is_spike and not in_spike:
            # Start of spike
            start_idx = i
            in_spike = True
        elif not is_spike and in_spike:
            # End of spike
            spike_length = i - start_idx
            if spike_length >= min_spike_width:
                spike_starts.append(start_idx)
                spike_ends.append(i - 1)
            in_spike = False

    # Handle case where data ends while in a spike
    if in_spike:
        spike_length = len(y) - start_idx
        if spike_length >= min_spike_width:
            spike_starts.append(start_idx)
            spike_ends.append(len(y) - 1)

    # Calculate spike lengths
    spike_info = []
    spike_lengths = []

    for start, end in zip(spike_starts, spike_ends):
        length = end - start + 1
        spike_lengths.append(length)
        spike_info.append((start, end, length))

    # Calculate average
    average_length = np.mean(spike_lengths) if spike_lengths else 0

    return average_length, spike_info
