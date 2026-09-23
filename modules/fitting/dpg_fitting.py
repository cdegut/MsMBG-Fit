import dearpygui.dearpygui as dpg
from modules.fitting.dpg_callbacks import *
from modules.fitting.fitting_quality import laplace_covariance_analysis


def fitting_window(render_callback):
    spectrum = render_callback.spectrum
    with dpg.child_window(label="Peak fitting", tag="Peak fitting"):
        # Create a plot for the raw data
        with dpg.plot(
            label="Gaussian Fit", width=1430, height=600, tag="gaussian_fit_plot"
        ) as plot2:
            dpg.add_plot_axis(dpg.mvXAxis, label="m/z", tag="x_axis_plot2")
            dpg.add_plot_axis(dpg.mvYAxis, label="Y Axis", tag="y_axis_plot2")
            dpg.add_line_series(
                [],
                [],
                label="Corrected Data Series",
                parent="y_axis_plot2",
                tag="corrected_series_plot2",
            )
            dpg.add_line_series(
                [],
                [],
                label="Residual",
                parent="y_axis_plot2",
                tag="residual",
                show=False,
            )
            dpg.add_line_series(
                [], [], label="MBG", parent="y_axis_plot2", tag="MBG_plot2", show=False
            )

        with dpg.group(horizontal=True, horizontal_spacing=50):
            with dpg.group(horizontal=False):
                dpg.add_button(
                    label="Start Multi Bi Gaussian Deconvolution",
                    callback=run_fitting_callback,
                    user_data=render_callback,
                    tag="start_fitting_button",
                )
                dpg.add_checkbox(
                    label="<--- Tick to Stop Fitting",
                    user_data=render_callback,
                    tag="stop_fitting_checkbox",
                    show=False,
                )

                with dpg.group(horizontal=False):
                    dpg.add_button(
                        label="Quality of fit Analysis",
                        callback=run_advanced_statistical_analysis_callback,
                        tag="advanced_statistical_analysis_button",
                    )
                    dpg.add_button(
                        label="laplace_covariance_analysis",
                        callback=laplace_covariance_analysis,
                        tag="laplace_covariance_analysis_button",
                    )
                    with dpg.group(horizontal=True):
                        dpg.add_checkbox(
                            label="Use Symmetric Gaussian",
                            default_value=False,
                            tag="use_gaussian",
                        )
                        dpg.add_checkbox(
                            label="use Lorentzian Peak Model",
                            default_value=False,
                            tag="use_lorentzian_checkbox",
                            callback=toggle_lorentzian_peak_model,
                        )
                    dpg.add_checkbox(
                        label="Fit on filtered data",
                        default_value=False,
                        tag="use_filtered",
                    )
                    dpg.add_checkbox(
                        label="Reduce width variance",
                        default_value=True,
                        tag="use_reduced",
                    )
            dpg.add_loading_indicator(
                style=5, radius=3, show=False, tag="Fitting_indicator"
            )
            with dpg.group(horizontal=False, horizontal_spacing=50):
                dpg.add_text("", tag="Fitting_indicator_text")
                dpg.add_text("", tag="Fitting_indicator_sub_text")
        with dpg.group(horizontal=True, horizontal_spacing=50):
            with dpg.group(horizontal=False):
                dpg.add_text("Fitting Iterations:")
                dpg.add_input_int(
                    label="",
                    default_value=1000,
                    min_value=50,
                    max_value=10000,
                    width=200,
                    tag="fitting_iterations",
                )
            with dpg.group(horizontal=False):
                dpg.add_text("OR Parameters changes below (*e-5):")
                dpg.add_input_float(
                    label="",
                    default_value=1,
                    step=1,
                    min_value=0.01,
                    width=200,
                    tag="theta_threshold_selector",
                )
            with dpg.group(horizontal=False):
                dpg.add_text("OR R2 above:")
                dpg.add_input_float(
                    label="",
                    default_value=0.991,
                    min_value=0.8,
                    max_value=1,
                    width=200,
                    tag="fitting_r2",
                    step=0.001,
                )
        dpg.add_checkbox(
            label="Show residual",
            default_value=True,
            tag="show_residual_checkbox",
            callback=show_residual_callback,
            user_data=spectrum,
        )
        dpg.add_checkbox(
            label="Show baseline projections",
            callback=draw_base_projection,
            tag="show_projection_checkbox",
            default_value=False,
        )
        # dpg.add_button(
        #     label="Draw Initial Peaks",
        #     callback=draw_initial_peaks_callback,
        #     user_data=spectrum,
        # )
        dpg.add_button(label="Redraw Peaks", callback=draw_fitted_peaks_callback)

        with dpg.child_window(height=100, width=1200, tag="peak_table_window"):
            with dpg.table(header_row=True, tag="peak_table"):
                dpg.add_table_column(label="Peak Label")
                dpg.add_table_column(label="Peak Start")
                dpg.add_table_column(label="Peak Apex")
                dpg.add_table_column(label="Peak Integral")
                dpg.add_table_column(label="Sigma Left")
                dpg.add_table_column(label="Sigma Right")
                dpg.add_table_column(label="Relative Error")
                dpg.add_table_column(label="R²")

    dpg.bind_item_theme("corrected_series_plot2", "data_theme")
    dpg.bind_item_theme("residual", "residual_theme")
    dpg.bind_item_theme("MBG_plot2", "fitting_MBG_theme")
