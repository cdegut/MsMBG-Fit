import dearpygui.dearpygui as dpg

# Good / bad colours, shared by the status texts and the peak labels
GOOD_COLOR = (20, 130, 60)
BAD_COLOR = (200, 30, 30)


def peak_label_color(is_bad: bool) -> tuple[int, int, int]:
    """Colour of a peak's label (plots): red for a peak marked bad, green otherwise."""
    return BAD_COLOR if is_bad else GOOD_COLOR


def peak_label_theme(is_bad: bool) -> str:
    """Text theme of a peak's label (tables), same colours as peak_label_color."""
    return "text_bad_theme" if is_bad else "text_good_theme"


def create_styles():
    with dpg.theme(tag="general_theme"):
        with dpg.theme_component(dpg.mvAll):
            # dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (131, 184, 198), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(
                dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_style(
                dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots
            )

    with dpg.theme(tag="derivative2nd_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(
                dpg.mvPlotStyleVar_LineWeight, 1, category=dpg.mvThemeCat_Plots
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_Line, (48, 70, 116), category=dpg.mvThemeCat_Plots
            )

    with dpg.theme(tag="data_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(
                dpg.mvPlotStyleVar_LineWeight, 1, category=dpg.mvThemeCat_Plots
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_Line, (100, 100, 100), category=dpg.mvThemeCat_Plots
            )

    with dpg.theme(tag="filtered_data_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(
                dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_Line, (250, 120, 14), category=dpg.mvThemeCat_Plots
            )

    with dpg.theme(tag="baseline_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(
                dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_Line, (44, 160, 44), category=dpg.mvThemeCat_Plots
            )

    with dpg.theme(tag="matching_MBG_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(
                dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_Line, (216, 82, 75), category=dpg.mvThemeCat_Plots
            )

    with dpg.theme(tag="fitting_MBG_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(
                dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_Line, (216, 82, 75), category=dpg.mvThemeCat_Plots
            )

    with dpg.theme(tag="table_row_bg_grey"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(
                dpg.mvThemeCol_TableRowBg, (100, 100, 100), category=dpg.mvThemeCat_Core
            )

    # Status text colours (readable on the light background)
    for name, color in (
        ("text_good_theme", GOOD_COLOR),
        ("text_warn_theme", (200, 110, 0)),
        ("text_bad_theme", BAD_COLOR),
        ("text_muted_theme", (130, 130, 130)),
    ):
        with dpg.theme(tag=name):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Text, color, category=dpg.mvThemeCat_Core
                )

    # Prominent filled button (final error analysis)
    with dpg.theme(tag="primary_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(
                dpg.mvThemeCol_Button, (30, 110, 200), category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonHovered, (20, 90, 175), category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonActive, (15, 70, 145), category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Text, (255, 255, 255), category=dpg.mvThemeCat_Core
            )

    # Red "Stop fitting" button (a selectable inside a red frame, see dpg_fitting)
    with dpg.theme(tag="stop_button_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(
                dpg.mvThemeCol_ChildBg, (205, 55, 55), category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_HeaderHovered, (175, 35, 35), category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_HeaderActive, (130, 20, 20), category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Header, (130, 20, 20), category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Text, (255, 255, 255), category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_ChildRounding, 5, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_SelectableTextAlign, 0.5, 0.5, category=dpg.mvThemeCat_Core
            )

    with dpg.theme(tag="residual_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(
                dpg.mvPlotStyleVar_LineWeight, 1, category=dpg.mvThemeCat_Plots
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_Line, (98, 109, 126), category=dpg.mvThemeCat_Plots
            )

    with dpg.theme(tag="residual_theme_plot1"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(
                dpg.mvPlotStyleVar_LineWeight, 1, category=dpg.mvThemeCat_Plots
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_Line, (148, 103, 189), category=dpg.mvThemeCat_Plots
            )

    with dpg.theme(tag="light"):
        with dpg.theme_component(0):
            dpg.add_theme_style(
                dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_style(
                dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots
            )
            dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0, 255))
            dpg.add_theme_color(
                dpg.mvThemeCol_TextDisabled,
                (153, 153, 153, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_WindowBg,
                (240, 240, 240, 255),
            )
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (0, 0, 0, 0))
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (255, 255, 255, 250))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 77))
            dpg.add_theme_color(
                dpg.mvThemeCol_BorderShadow,
                (0, 0, 0, 0),
            )
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (255, 255, 255, 255))
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgHovered,
                (66, 150, 250, 102),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgActive,
                (66, 150, 250, 171),
            )
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (245, 245, 245, 255))
            dpg.add_theme_color(
                dpg.mvThemeCol_TitleBgActive,
                (209, 209, 209, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TitleBgCollapsed,
                (255, 255, 255, 130),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_MenuBarBg,
                (219, 219, 219, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarBg,
                (250, 250, 250, 135),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarGrab,
                (176, 176, 176, 204),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarGrabHovered,
                (125, 125, 125, 204),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarGrabActive,
                (125, 125, 125, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_CheckMark,
                (66, 150, 250, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_SliderGrab,
                (66, 150, 250, 199),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_SliderGrabActive,
                (117, 138, 204, 153),
            )
            dpg.add_theme_color(dpg.mvThemeCol_Button, (66, 150, 250, 102))
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonHovered,
                (66, 150, 250, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonActive,
                (15, 135, 250, 255),
            )
            dpg.add_theme_color(dpg.mvThemeCol_Header, (66, 150, 250, 79))
            dpg.add_theme_color(
                dpg.mvThemeCol_HeaderHovered,
                (66, 150, 250, 204),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_HeaderActive,
                (66, 150, 250, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Separator,
                (99, 99, 99, 158),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_SeparatorHovered,
                (36, 112, 204, 199),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_SeparatorActive,
                (36, 112, 204, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ResizeGrip,
                (89, 89, 89, 43),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ResizeGripHovered,
                (66, 150, 250, 171),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ResizeGripActive,
                (66, 150, 250, 242),
            )
            dpg.add_theme_color(dpg.mvThemeCol_Tab, (194, 204, 214, 237))
            dpg.add_theme_color(
                dpg.mvThemeCol_TabHovered,
                (66, 150, 250, 204),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabActive,
                (153, 186, 224, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabUnfocused,
                (235, 237, 240, 252),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabUnfocusedActive,
                (189, 209, 232, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_DockingPreview,
                (66, 150, 250, 56),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_DockingEmptyBg,
                (51, 51, 51, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_PlotLines,
                (99, 99, 99, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_PlotLinesHovered,
                (255, 110, 89, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_PlotHistogram,
                (230, 179, 0, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_PlotHistogramHovered,
                (255, 115, 0, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableHeaderBg,
                (199, 222, 250, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableBorderStrong,
                (145, 145, 163, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableBorderLight,
                (173, 173, 189, 255),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableRowBg,
                (0, 0, 0, 0),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableRowBgAlt,
                (77, 77, 77, 23),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TextSelectedBg,
                (66, 150, 250, 89),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_DragDropTarget,
                (66, 150, 250, 242),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_NavHighlight,
                (66, 150, 250, 204),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_NavWindowingHighlight,
                (179, 179, 179, 179),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_NavWindowingDimBg,
                (51, 51, 51, 51),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ModalWindowDimBg,
                (51, 51, 51, 89),
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_FrameBg,
                (255, 255, 255, 255),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_PlotBg,
                (255, 255, 255, 255),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_PlotBorder,
                (0, 0, 0, 0),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_LegendBg,
                (255, 255, 255, 250),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_LegendBorder,
                (209, 209, 209, 204),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_LegendText,
                (0, 0, 0, 255),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_TitleText,
                (0, 0, 0, 255),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_InlayText,
                (0, 0, 0, 255),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_AxisBg,
                (0, 0, 0, 0),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_AxisBgActive,
                (15, 135, 250, 255),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_AxisBgHovered,
                (66, 150, 250, 255),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_AxisGrid,
                (0, 0, 0, 255),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_AxisText,
                (0, 0, 0, 255),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_Selection,
                (209, 163, 8, 255),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_Crosshairs,
                (0, 0, 0, 128),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_NodeBackground,
                (240, 240, 240, 255),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_NodeBackgroundHovered,
                (240, 240, 240, 255),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_NodeBackgroundSelected,
                (240, 240, 240, 255),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_NodeOutline,
                (100, 100, 100, 255),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_TitleBar,
                (248, 248, 248, 255),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_TitleBarHovered,
                (209, 209, 209, 255),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_TitleBarSelected,
                (209, 209, 209, 255),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_Link, (66, 150, 250, 100), category=dpg.mvThemeCat_Nodes
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_LinkHovered,
                (66, 150, 250, 242),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_LinkSelected,
                (66, 150, 250, 242),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_Pin, (66, 150, 250, 160), category=dpg.mvThemeCat_Nodes
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_PinHovered,
                (66, 150, 250, 255),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_BoxSelector,
                (90, 170, 250, 30),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_BoxSelectorOutline,
                (90, 170, 250, 150),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_GridBackground,
                (225, 225, 225, 255),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_GridLine,
                (180, 180, 180, 100),
                category=dpg.mvThemeCat_Nodes,
            )
