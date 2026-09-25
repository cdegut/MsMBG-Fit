from dataclasses import dataclass
from typing import List
from modules.data_structures import MSData, get_global_msdata_ref
import time as time
import dearpygui.dearpygui as dpg
import importlib

from modules.finding_callback import get_smoothing_window


@dataclass
class RenderCallback:
    spectrum: MSData

    def __post_init__(self):

        self.last_baseline_corrected = time.time()
        self.baseline_correct_requested = False
        self.mz_lines = {}
        self.stop_fitting = False
        self.wd_len: List = []
        self.working_peak_list: List[int] = []
        self.iterations_done: int = 0
        self.finishing_delta_theta: float = 0.0

    @property
    def fit_summary(self):
        """FitSummary of the last MBG fit, stored in the spectrum so it is saved with it."""
        return self.spectrum.fit_summary

    @fit_summary.setter
    def fit_summary(self, value):
        self.spectrum.fit_summary = value

    def execute(self):
        now = time.time()
        if self.spectrum.baseline_need_update == True:

            self.wd_len.append(len(self.spectrum.working_data))
            if len(self.wd_len) > 20:
                self.wd_len.pop(0)
                dpg.set_value(
                    "baseline",
                    [
                        self.spectrum.working_data[:, 0].tolist(),
                        [0] * len(self.spectrum.working_data[:, 0].tolist()),
                    ],
                )

            if all(x == self.wd_len[0] for x in self.wd_len):
                self.correct_baseline()
                finding_dpg = importlib.import_module("modules.finding_dpg")
                finding_dpg.move_threshold_callback()
                self.display_2nd_derivative()
                self.last_baseline_corrected = time.time()
                self.baseline_correct_requested = False
                self.spectrum.baseline_need_update = False

    def request_baseline_correction(self):
        self.baseline_correct_requested = True

    def correct_baseline(self):

        window = dpg.get_value("baseline_window")
        self.spectrum.correct_baseline(window)

        dpg.set_value(
            "baseline",
            [
                self.spectrum.baseline[:, 0].tolist(),
                self.spectrum.baseline[:, 1].tolist(),
            ],
        )
        dpg.set_value(
            "corrected_series_plot2",
            [
                self.spectrum.baseline_corrected[:, 0].tolist(),
                self.spectrum.baseline_corrected[:, 1].tolist(),
            ],
        )
        dpg.set_value(
            "corrected_series_plot3",
            [
                self.spectrum.baseline_corrected[:, 0].tolist(),
                self.spectrum.baseline_corrected[:, 1].tolist(),
            ],
        )
        dpg.fit_axis_data("y_axis_plot2")

    def display_2nd_derivative(self):
        if not dpg.get_value("show_smoothed_data_checkbox"):
            return
        window = get_smoothing_window()
        derivative2nd = self.spectrum.get_2nd_derivative(window)
        if derivative2nd is not None and len(derivative2nd) != len(
            self.spectrum.working_data
        ):
            print("2nd derivative length mismatch.")
            return
        dpg.set_value(
            "derivative2nd", [self.spectrum.working_data[:, 0].tolist(), derivative2nd]
        )


spectrum = get_global_msdata_ref()
render_callback_global_ref = RenderCallback(spectrum)


def get_global_render_callback_ref() -> RenderCallback:
    return render_callback_global_ref
