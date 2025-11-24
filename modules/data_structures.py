from _collections_abc import dict_keys
from typing import List, Literal, Optional
from pybaselines import Baseline
from scipy.signal import medfilt
import numpy as np
from dataclasses import dataclass
from modules.math import multi_bi_gaussian, multi_bi_Lorentzian
from typing import Dict, Tuple
from dataclasses import field
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

        for peak in ordered_peaks_list:
            if self.peaks[peak].do_not_fit:
                continue

            if integral:
                packed_params.append(self.peaks[peak].integral)
            else:
                packed_params.append(self.peaks[peak].A_refined)
            packed_params.append(self.peaks[peak].x0_refined)
            packed_params.append(self.peaks[peak].sigma_L)
            packed_params.append(self.peaks[peak].sigma_R)
        return packed_params, ordered_peaks_list

    # @staticmethod
    # def load_from_file(path: str) -> "MSData":
    #    """Load an MSData object from a file using pickle."""
    #    with open(path, 'rb') as f:
    #        data: MSData = pickle.load(f)
    #    return data

    def load_from_file(self, path: str):
        with open(path, "rb") as f:
            new_data: MSData = pickle.load(f)
        self.original_data = new_data.original_data
        self.working_data = new_data.working_data
        self.baseline = new_data.baseline
        self.baseline_corrected = new_data.baseline_corrected
        self.peaks = new_data.peaks
        self.baseline_toggle = new_data.baseline_toggle
        self.baseline_need_update = new_data.baseline_need_update
        self.matching_data = new_data.matching_data
        self.smoothing_window = new_data.smoothing_window
        self.peak_detection_parameters = new_data.peak_detection_parameters
        self.baseline_window = new_data.baseline_window
        try:
            self.block_width = new_data.block_width
        except AttributeError:
            self.block_width = 25
        try:
            self.matching_series = new_data.matching_series
        except AttributeError:
            self.matching_series = 3
        try:
            self.peak_model = new_data.peak_model
        except AttributeError:
            self.peak_model = "gaussian"


ms_data_global_ref = MSData()


def get_global_msdata_ref() -> MSData:
    return ms_data_global_ref


@dataclass
class FitQualityPeakMetrics:
    snr: float
    peak_rmse: float
    relative_error: float
    r_squared: float


@dataclass
class MatchedWith:
    set: int
    charge: int
    mw: float
    ratio: float


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
