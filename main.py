import dearpygui.dearpygui as dpg
from modules.data_structures import MSData
from modules.fitting.dpg_fitting import fitting_window
from modules.matching_dpg import matching_window
from modules.finding_dpg import finding_window

# from modules.intialise import initialise_windows
from modules.intialise import (
    APP_TITLE,
    file_dialog,
    file_dialog_save_data,
    file_dialog_load_saved_data,
)
from modules.rendercallback import RenderCallback, get_global_render_callback_ref
from modules.dpg_style import create_styles
from modules.data_structures import get_global_msdata_ref


# path = fr"D:\MassSpec\Um_2-1_1x.csv"
# path = fr"D:\MassSpec\Um_data.csv"
# path = fr"D:\MassSpec\CS_RBC_alone.csv"
# path = fr"D:\MassSpec\CR_aloneCID.csv"


def main():
    dpg.create_context()
    render_callback = get_global_render_callback_ref()
    create_styles()
    file_dialog(render_callback)
    file_dialog_load_saved_data(render_callback)
    file_dialog_save_data(render_callback)

    with dpg.window(
        label="Control",
        width=1430,
        height=-1,
        no_close=True,
        no_collapse=True,
        no_move=True,
        no_resize=True,
        no_title_bar=True,
        tag="Control",
    ):
        dpg.set_primary_window("Control", True)
        # Add a slider to adjust the window length
        with dpg.child_window(height=40):
            dpg.add_text(tracked=True, track_offset=1.0, tag="message_box")
        with dpg.tab_bar(label="Test", tag="tab_bar", pos=[500, 0]):  # <- Not working
            with dpg.tab(label="Home"):
                dpg.add_button(
                    label="Open data", callback=lambda: dpg.show_item("file_dialog_id")
                )
                dpg.add_button(
                    label="Open processed data",
                    callback=lambda: dpg.show_item("file_dialog_id_saved_data"),
                )
                dpg.add_button(
                    label="Save processed data",
                    callback=lambda: dpg.show_item("file_dialog_id_save_data"),
                )
                dpg.add_loading_indicator(
                    label="Loading", tag="file_loading_indicator", show=False
                )

            with dpg.tab(label="Peak Finding"):
                finding_window(render_callback)
            with dpg.tab(label="Fitting and deconvolution"):
                fitting_window(render_callback)
                pass
            with dpg.tab(label="Matching"):
                matching_window(render_callback)
                pass

    dpg.bind_theme("light")

    ### Auto start ### delete for normal use
    """
    dpg.set_value("L_data_clipping", 9500)
    dpg.set_value("R_data_clipping", 11700)
    dpg.set_value("peak_detection_threshold", 50)
    dpg.set_value("peak_detection_width", 8)
    dpg.set_value("peak_detection_distance", 30)
    dpg.set_value("baseline_window", 1000)
    from modules.finding_dpg import data_clipper, toggle_baseline
    toggle_baseline(None,None, spectrum)
    import time
    time.sleep(1)
    data_clipper(None, None, spectrum)
    spectrum.baseline_need_update = True
    #peaks_finder_callback(None, None, spectrum)
    #run_fitting(None, None, spectrum)
    #update_peak_starting_points(None, None, render_callback)
    #multi_bigaussian_fit(None, None,spectrum)
    # Create a viewport and show the plot
    dpg.create_viewport(title='Multi Bi Gaussian Fit', width=1450, height=1000, x_pos=0, y_pos=0)
    #dpg.focus_item("Data Filtering")
    #dpg.focus_item("Peak fitting")
    data_clipper(None, None, spectrum)
    #dpg.focus_item("Peak matching")
    dpg.show_style_editor()
    #dpg.show_metrics()
    """
    #####
    # dpg.show_style_editor()
    dpg.create_viewport(title=APP_TITLE, width=1450, height=1000, x_pos=0, y_pos=0)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    # spectrum.import_csv(path)
    # initialise_windows(render_callback)

    while dpg.is_dearpygui_running():

        render_callback.execute()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
