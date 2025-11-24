# MsMBG-Fit

Multi Bi-Gaussian Fitting tool for native Mass Spectrometry data analysis and deconvolution.

## Overview

MsMBG-Fit is a Python application designed for analyzing mass spectrometry data through multi bi-gaussian peak fitting and deconvolution. The tool provides an interactive GUI built with DearPyGUI for peak detection, fitting, and matching workflows.

## Features

-   **Interactive GUI**: User-friendly interface for data visualization and analysis
-   **Peak Detection**: Automated peak finding with customizable thresholds and parameters
-   **Multi Bi-Gaussian Fitting**: Advanced fitting algorithm using asymmetric bi-gaussian functions
-   **Baseline Correction**: Automatic baseline detection and correction
-   **Statistical Analysis**:
    -   Bootstrap methods for error estimation
    -   Quality metrics (R², χ², RMSE)
    -   Signal-to-noise ratio calculations
    -   Standard error estimation
-   **Peak Matching**: Match identified peaks across datasets
-   **Data Import/Export**: Support for CSV data files and processed data persistence

## Installation

### Requirements

-   Python 3.13 (3.14 not supported yet by dearpygui)
-   Required packages:
    -   dearpygui
    -   numpy
    -   scipy
    -   scikit-learn
    -   pandas (for data handling)
    -   pybaselines (for baseline correction)
    -   whittaker_eilers (data smoother)
    -   matplotlib
    -   seaborn

### Setup

```powershell
# Clone the repository
git clone https://github.com/cdegut/MsMBG-Fit.git
cd MsMBG-Fit

# Install dependencies
pip install dearpygui numpy scipy scikit-learn pandas pybaselines whittaker_eilers matplotlib seaborn
```

## Usage

### Starting the Application

```powershell
python main.py
```

### Workflow

1. **Load Data**

    - Click "Open data" to load a CSV file
    - Or "Open processed data" to load previously saved analysis

2. **Peak Finding**

    - Navigate to the "Peak Finding" tab
    - Adjust data clipping ranges
    - Tick Show Smoothed Data and Show 2nd order Derivative
    - Move the lambda smother so the 2nd derivative shows clear maximum at peaks
    - Add baseline correction
        - Change basiline window so the baseline doesn't hug the data too much
    - Configure peak detection parameters:
        - Threshold (click show to see the effect)
        - Width
        - Distance
    - Run peak detection
    - Review detected peaks on the plot
    - Optionally add or remove peaks manually

3. **Fitting and Deconvolution**

    - Switch to "Fitting and deconvolution" tab
    - Run multi bi-gaussian fit
    - Review fit quality metrics
    - Optionally run advanced statistical analysis for error estimation (always do this before reporting results)

4. **Matching**

    - Use the "Matching" tab to compare and match peaks across datasets
    - You can use up to 10 peak series for matching
    - Individually change the expected MW, Maximum charges, and number of peaks.
    - You can use refine to try to improve the matching based on the current peak series
    - Look at the matching score and the matching results
        - series should be unimodals (singular maximum and monotonic decrease on each side)
        - unique series should not have huge width std error
    - Print to terminal the matching results an then be copied easily

5. **Save Results**

    - Click "Save processed data" to export your analysis

## Data Format

The application expects CSV files with mass spectrometry data containing:

-   Column 1: m/z values (mass-to-charge ratio)
-   Column 2: Intensity values

Example data files are provided:

-   `ExempleData1.csv`
-   `ExempleData2.csv`
-   `ExempleData3.csv`
-   `synthetic_bi_gaussian.csv`
-   `synthetic_multi_bi_gaussian.csv`

## Project Structure

```
MsMBG-Fit/
├── main.py                          # Application entry point
├── modules/
│   ├── data_structures.py           # Core data structures
│   ├── finding.py                   # Peak finding algorithms
│   ├── finding_dpg.py               # Peak finding GUI
│   ├── matching.py                  # Peak matching logic
│   ├── matching_dpg.py              # Peak matching GUI
│   ├── dpg_draw.py                  # Drawing utilities
│   ├── dpg_style.py                 # GUI styling
│   ├── intialise.py                 # Initialization routines
│   ├── read_excel.py                # Data import
│   ├── rendercallback.py            # Rendering callbacks
│   ├── math.py                      # Mathematical functions
│   ├── utils.py                     # Utility functions
│   ├── var.py                       # Global variables
│   └── fitting/
│       ├── MBGfit.py                # Main fitting algorithm
│       ├── fitting_quality.py       # Quality metrics calculation
│       ├── peak_starting_points.py  # Initial parameter estimation
│       ├── refiner.py               # Parameter refinement
│       ├── draw_MBG.py              # Fit visualization
│       └── dpg_fitting.py           # Fitting GUI
```

## Algorithm Details

### Bi-Gaussian Model

The tool uses an asymmetric bi-gaussian function for peak modeling:

$$
g(x) = A  \cdot \begin{cases}
\exp\left(-\frac{(x - x_0)^2}{2\sigma^2_L}\right),&\text{if \(x < x_0\)}
\\\\
\exp\left(-\frac{(x - x_0)^2}{2\sigma^2_R}\right),&\text{if \(x > x_0\)}
\end{cases}


$$

where:

-   $A$ = peak amplitude
-   $x_0$ = peak center
-   $\sigma_L$ = left-side standard deviation
-   $\sigma_R$ = right-side standard deviation

### Fitting Process

1. **Initial Parameter Estimation**: Automatic detection of peak positions, heights, and widths
2. **Iterative Refinement**: Sequential optimization of individual peak parameters
3. **Convergence Checking**: Multiple criteria including R², RMSE, and parameter gradients
4. **Quality Assessment**: Comprehensive statistical analysis of fit quality

### Statistical Analysis

-   **Parametric Bootstrap**: Resampling with added Gaussian noise
-   **Residual Bootstrap**: Resampling of fit residuals
-   **Random Start Analysis**: Multiple fits with perturbed initial conditions
-   **Standard Error Calculation**: Robust uncertainty quantification

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is maintained by cdegut. Please check the repository for license information.

## Acknowledgments

Built with:

-   [DearPyGUI](https://github.com/hoffstadt/DearPyGui) for the GUI framework
-   NumPy and SciPy for numerical computations
-   scikit-learn for statistical utilities

## Contact

For questions or support, please open an issue on the GitHub repository.
