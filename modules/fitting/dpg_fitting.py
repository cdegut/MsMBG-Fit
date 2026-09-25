import dearpygui.dearpygui as dpg
from modules.fitting.dpg_callbacks import *


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

        # Display options, on one line under the plot
        with dpg.group(horizontal=True, horizontal_spacing=30):
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
            dpg.add_button(label="Redraw Peaks", callback=draw_fitted_peaks_callback)
            # Progress and result of the last fit / analysis: right under the plot,
            # so it stays visible while fitting
            dpg.add_loading_indicator(
                style=5, radius=3, show=False, tag="Fitting_indicator"
            )
            with dpg.group(horizontal=False):
                dpg.add_text("", tag="Fitting_indicator_text")
                dpg.add_text("", tag="Fitting_indicator_sub_text")
        dpg.add_spacer(height=4)

        # Row 1: start / stop, and the three stopping criteria
        # (the one that ended the last fit gets highlighted)
        with dpg.group(horizontal=True, horizontal_spacing=40):
            with dpg.group(horizontal=False):
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Start Multi Bi Gaussian Deconvolution",
                        callback=run_fitting_callback,
                        user_data=render_callback,
                        tag="start_fitting_button",
                    )
                    dpg.add_button(label="Fit options...", tag="fit_options_button")
                # Model options, in a popup opened by the button above
                with dpg.popup(
                    "fit_options_button",
                    mousebutton=dpg.mvMouseButton_Left,
                    tag="fit_options_popup",
                ):
                    dpg.add_text("Peak model")
                    dpg.bind_item_theme(dpg.last_item(), "text_muted_theme")
                    dpg.add_checkbox(
                        label="Use Symmetric Gaussian",
                        default_value=False,
                        tag="use_gaussian",
                    )
                    dpg.add_checkbox(
                        label="Use Lorentzian Peak Model",
                        default_value=False,
                        tag="use_lorentzian_checkbox",
                        callback=toggle_lorentzian_peak_model,
                    )
                    dpg.add_separator()
                    dpg.add_text("Fitting")
                    dpg.bind_item_theme(dpg.last_item(), "text_muted_theme")
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
                # A selectable rather than a button: its value changes in the render
                # thread immediately, while a button callback would be queued until
                # the running fit callback returns.
                with dpg.child_window(
                    width=390,
                    height=26,
                    no_scrollbar=True,
                    border=False,
                    show=False,
                    tag="stop_fitting_frame",
                ):
                    dpg.add_selectable(
                        label="Stop fitting",
                        width=390,
                        height=26,
                        tag="stop_fitting_button",
                    )
                dpg.bind_item_theme("stop_fitting_frame", "stop_button_theme")

            # Stopping criteria panel
            with dpg.child_window(
                border=True,
                auto_resize_x=True,
                auto_resize_y=True,
                no_scrollbar=True,
                tag="stop_criteria_panel",
            ):
                # Collapsible; the label summarises the settings and the last stop reason
                with dpg.tree_node(
                    label="Stop at", default_open=False, tag="stop_criteria_node"
                ):
                    with dpg.group(horizontal=True, horizontal_spacing=40):
                        with dpg.group(horizontal=False):
                            dpg.add_text("Max iterations:", tag="stop_iter_label")
                            dpg.add_input_int(
                                label="",
                                default_value=1000,
                                min_value=50,
                                max_value=5000,
                                width=150,
                                tag="fitting_iterations",
                                callback=update_stop_criteria_label,
                            )
                            dpg.add_text("", tag="stop_iter_status")
                        with dpg.group(horizontal=False):
                            dpg.add_text(
                                "OR moves below (error bars / 50 it.):", tag="stop_theta_label"
                            )
                            with dpg.tooltip("stop_theta_label"):
                                dpg.add_text(
                                    "Converged when the parameters averaged over the last 50 "
                                    "iterations differ from the average over the 50 before by "
                                    "less than this fraction of their Laplace error bar, for every "
                                    "peak's integral, apex and widths.",
                                    wrap=380,
                                )
                            dpg.add_input_float(
                                label="",
                                default_value=0.3,
                                step=0.05,
                                min_value=0.01,
                                min_clamped=True,
                                format="%.2f",
                                width=150,
                                tag="theta_threshold_selector",
                                callback=update_stop_criteria_label,
                            )
                            dpg.add_text("", tag="stop_theta_status")
                        with dpg.group(horizontal=False):
                            dpg.add_text("OR R² above:", tag="stop_r2_label")
                            dpg.add_input_float(
                                label="",
                                default_value=0.991,
                                min_value=0.8,
                                max_value=1,
                                width=150,
                                tag="fitting_r2",
                                callback=update_stop_criteria_label,
                                step=0.001,
                            )
                            dpg.add_text("", tag="stop_r2_status")

        # Row 2: final error analysis, its status on the right
        with dpg.group(horizontal=True, horizontal_spacing=20):
            dpg.add_button(
                label="Final error analysis (bootstrap)",
                callback=run_advanced_statistical_analysis_callback,
                tag="advanced_statistical_analysis_button",
                width=300,
                height=30,
            )
            dpg.bind_item_theme(dpg.last_item(), "primary_button_theme")
            with dpg.tooltip("advanced_statistical_analysis_button"):
                dpg.add_text(
                    "Parametric bootstrap (128 resamples, about 1-2 minutes) and random "
                    "restarts (16 refits from randomised starts, several minutes). Every "
                    "refit stops with the same convergence test as the fit. "
                    "Gives the final ± on integrals and positions: "
                    "the largest of bootstrap, Laplace and restarts.",
                    wrap=400,
                )
            dpg.add_text("", tag="error_analysis_status", wrap=800)
            # dpg.add_button(
            #     label="Laplace error analysis",
            #     callback=run_laplace_analysis_callback,
            #     tag="laplace_covariance_analysis_button",
            # )

        # dpg.add_button(
        #     label="Draw Initial Peaks",
        #     callback=draw_initial_peaks_callback,
        #     user_data=spectrum,
        # )

        dpg.add_separator()
        # Global fit statistics, filled by update_peak_table()
        dpg.add_group(horizontal=True, horizontal_spacing=30, tag="fit_summary_group")

        with dpg.colormap_registry():
            dpg.add_colormap(
                [[int(c * 255) for c in col] + [255] for col in PEAK_COLORS],
                qualitative=True,
                tag="peak_error_colormap",
            )

        with dpg.group(horizontal=True):
            with dpg.table(
                header_row=True,
                tag="peak_table",
                width=1330,
                height=320,
                scrollY=True,
                freeze_rows=1,
                resizable=True,
                sortable=True,
                callback=sort_peak_table,
                row_background=True,
                borders_innerV=True,
                borders_outerV=True,
                borders_outerH=True,
                policy=dpg.mvTable_SizingStretchProp,
            ):
                for label, key, weight in PEAK_TABLE_COLUMNS:
                    dpg.add_table_column(
                        label=label,
                        user_data=key,
                        no_sort=key is None,
                        init_width_or_weight=weight,
                        default_sort=key == "apex",
                    )
            dpg.add_colormap_scale(
                label="Rel. error",
                colormap="peak_error_colormap",
                min_scale=0.0,
                max_scale=ERROR_COLOR_MAX,
                height=320,
                width=80,
                format="%.2f",
            )
        dpg.add_text(
            "Peak colour = relative error (same as the plot). Green/orange/red = good/check/poor. "
            "Click a peak to zoom on it, double-click the plot to reset. "
            "± = Laplace error right after a fit (noise only, lower bound), final error after the bootstrap analysis. Area corr. L / R = integral correlation with the left / right neighbour (near -1: area split undetermined). Hover cells and Flags for details.",
            wrap=1400,
        )
        dpg.bind_item_theme(dpg.last_item(), "text_muted_theme")

    update_stop_criteria_label()
    dpg.bind_item_theme("corrected_series_plot2", "data_theme")
    dpg.bind_item_theme("residual", "residual_theme")
    dpg.bind_item_theme("MBG_plot2", "fitting_MBG_theme")
