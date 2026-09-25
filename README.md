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

-   Python 3.13 (3.14 not supported yet by dearpygui). You do not need to install it yourself: uv downloads it automatically.
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

### Quick start (launcher scripts)

Download the code (see [Get the code](#2-get-the-code)), then use the launcher for your system. On first launch it installs uv if needed, downloads Python 3.13 and the dependencies, then starts the application. Every launch also runs `git pull` to update to the latest version when the code was cloned with git. If you're offline or have local changes, the update is skipped and the current version starts.

| System  | Double-click          | Or from a terminal in the project folder |
| ------- | --------------------- | ---------------------------------------- |
| Windows | `run.bat`             | `.\run.bat`                              |
| macOS   | `run.command`         | `sh run.sh`                              |
| Linux   | —                     | `sh run.sh`                              |

-   **macOS**: if Finder refuses to open `run.command` because it is from an "unidentified developer", right-click it and choose **Open**. If it says you do not have permission, run `chmod +x run.command run.sh` once in a terminal.
-   **Windows**: if SmartScreen shows "Windows protected your PC", click **More info → Run anyway**.

If you'd rather install things yourself, follow the manual steps below.

### Setup with uv

The recommended way to run MsMBG-Fit is with [uv](https://docs.astral.sh/uv/). uv is a fast Python package manager that installs the correct Python version and the exact dependency versions recorded in `uv.lock`, in an environment kept separate from the rest of your system.

#### 1. Install uv

**Windows** (PowerShell):

```powershell
winget install --id=astral-sh.uv -e
```

or, without winget:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS**:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or, with Homebrew:

```shell
brew install uv
```

**Linux**:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(If `curl` is not available, use `wget -qO- https://astral.sh/uv/install.sh | sh`.)

After installing, **close and reopen your terminal**, then check that uv works:

```shell
uv --version
```

#### 2. Get the code

With git:

```shell
git clone https://github.com/cdegut/MsMBG-Fit.git
cd MsMBG-Fit
```

Without git: on the GitHub page, click **Code → Download ZIP**, extract it, and open a terminal in the extracted folder.

#### 3. Install the dependencies

From inside the project folder:

```shell
uv sync
```

This creates a `.venv` folder containing Python 3.13 and all the required packages. You only need to do this once, and again after updating the code.

#### Troubleshooting

-   **`uv` is not recognized / `command not found`**: your terminal hasn't picked up the updated PATH yet. Close and reopen it. In VS Code, close the whole application and reopen it, because a new terminal tab isn't enough. On Windows you can also reload the PATH in the current PowerShell window:

    ```powershell
    $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")
    ```

    On macOS/Linux, run `source $HOME/.local/bin/env` (or open a new terminal).

-   **Linux: the window does not open / OpenGL errors**: DearPyGUI needs OpenGL drivers and a graphical session. On Debian/Ubuntu, `sudo apt install libgl1 libglib2.0-0` usually fixes this.
-   **Windows: "running scripts is disabled on this system"**: this only happens if you activate `.venv` manually. Use `uv run` instead (see below), which does not need activation.

## Usage

### Starting the Application

From the project folder, on any operating system:

```shell
uv run python main.py
```

`uv run` automatically uses the project environment, so there is no need to activate `.venv` first. If the dependencies have changed, it syncs them before starting.

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
