import numpy as np
from scipy.integrate import quad


# Define multi-peak bi-Gaussian function
def multi_bi_gaussian(x, *params):
    n_peaks = len(params) // 4
    y = np.zeros_like(x, dtype=float)
    for i in range(n_peaks):
        A, x0, sigma_L, sigma_R = params[i * 4 : (i + 1) * 4]
        y += np.where(
            x < x0,
            A * np.exp(-((x - x0) ** 2) / (2 * sigma_L**2)),
            A * np.exp(-((x - x0) ** 2) / (2 * sigma_R**2)),
        )
    return y


def multi_bi_gaussian_using_I(x, *params):
    n_peaks = len(params) // 4
    y = np.zeros_like(x, dtype=float)
    for i in range(n_peaks):
        I, x0, sigma_L, sigma_R = params[i * 4 : (i + 1) * 4]
        A = I / (np.sqrt(2 * np.pi) * (sigma_L + sigma_R) / 2)
        y += np.where(
            x < x0,
            A * np.exp(-((x - x0) ** 2) / (2 * sigma_L**2)),
            A * np.exp(-((x - x0) ** 2) / (2 * sigma_R**2)),
        )
    return y


# Define bi-Gaussian function
def bi_gaussian(x, A, x0, sigma_L, sigma_R):
    return np.where(
        x < x0,
        A * np.exp(-((x - x0) ** 2) / (2 * sigma_L**2)),
        A * np.exp(-((x - x0) ** 2) / (2 * sigma_R**2)),
    )


def bi_gaussian_integral(A, sigmaL, sigmaR):
    """
    Compute total integral (area under curve) of an asymmetric bi-Gaussian.

    f(x) = A * exp(-((x - mu)^2)/(2*sigma_L^2)) for x < mu
         = A * exp(-((x - mu)^2)/(2*sigma_R^2)) for x >= mu

    Parameters
    ----------
    A : float
        Amplitude at x = mu
    sigmaL, sigmaR : float
        Left and right standard deviations

    Returns
    -------
    float
        Total integral over all x
    """
    return A * np.sqrt(np.pi / 2) * (sigmaL + sigmaR)


def bi_Lorentzian(x, amplitude, center, width_left, width_right):
    """
    Asymmetric Lorentzian function with different widths on each side

    Parameters:
    x: array of x values
    amplitude: peak height
    center: center position (x0)
    width_left: half-width at half-maximum for x < center (left side)
    width_right: half-width at half-maximum for x >= center (right side)
    """
    y = np.zeros_like(x)
    y += np.where(
        x < center,
        amplitude * (width_left**2) / ((x - center) ** 2 + width_left**2),
        amplitude * (width_right**2) / ((x - center) ** 2 + width_right**2),
    )

    return y


def multi_bi_Lorentzian(x, *params):
    """
    Multi-peak asymmetric Lorentzian function.

    Parameters
    ----------
    x : array-like
        Independent variable
    *params : tuple
        Flattened parameters: (A1, x0_1, width_L1, width_R1, A2, x0_2, width_L2, width_R2, ...)
        Each peak requires 4 parameters: amplitude, center, left width, right width

    Returns
    -------
    array-like
        Sum of all asymmetric Lorentzian peaks
    """
    n_peaks = len(params) // 4
    y = np.zeros_like(x, dtype=float)
    for i in range(n_peaks):
        A, x0, width_L, width_R = params[i * 4 : (i + 1) * 4]
        y += bi_Lorentzian(x, A, x0, width_L, width_R)
    return y


def bi_Lorentzian_integral(A, width_L, width_R):
    return A * np.pi * (width_L + width_R)


def bi_Lorentzian_integral_numerical(A, width_L, width_R, start, end):
    """
    Numerically integrate the bi-Lorentzian function from start to end.

    Parameters
    ----------
    A : float
        Amplitude at the center
    width_L : float
        Left width (HWHM for x < center)
    width_R : float
        Right width (HWHM for x >= center)
    start : float
        Lower integration limit
    end : float
        Upper integration limit

    Returns
    -------
    float
        Numerical integral of the bi-Lorentzian over [start, end]
    """

    # Assume center is at x=0 for integration
    center = 0.0

    def integrand(x):
        return bi_Lorentzian(x, A, center, width_L, width_R)

    result, _ = quad(integrand, start, end)
    return result


def combine_errors(noise: list[float], restart: float) -> float:
    """
    Final standard error: the largest of the available estimates (bootstrap,
    Laplace, random restarts). They are not independent (bootstrap refits include
    optimiser spread, restarts spread along the same flat directions Laplace
    measures), so adding them would double count. Values <= 0 are "not computed".
    """
    return float(max([0.0] + [v for v in [*noise, restart] if v > 0]))


def bootstrap_std(values: list[float]) -> float:
    """
    Standard error of a parameter estimated by resampling (bootstrap, random
    restarts): the standard deviation of the resampled values.
    """
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def standard_error(values: list[float]) -> float:
    """
    Calculate the standard error of the mean for a list of values.

    Standard Error (SE) = Standard Deviation (SD) / sqrt(n)

    where n is the number of samples.

    Parameters
    ----------
    values : list of float
        The list of values to calculate the standard error for.

    Returns
    -------
    float
        The standard error of the mean.
    """
    n = len(values)
    if n == 0:
        return float("nan")
    sd = np.std(values, ddof=1)  # Sample standard deviation
    se = sd / np.sqrt(n)
    return se


def check_theta_convergence(theta_old, theta_new, tol_theta=1e-4, eps=1e-12):
    """
    Determine whether local optimization has converged.

    Parameters
    ----------
    theta_old : np.ndarray
        Previous parameter vector (flattened).
    theta_new : np.ndarray
        New parameter vector after an iteration.
    tol_theta : float
        Relative tolerance on parameter change.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    converged : bool
        True if both criteria are satisfied.
    delta_obj : float
        Relative change in objective (for logging/debug).
    delta_theta : float
        Relative change in parameters (for logging/debug).
    """

    # Parameter change (normalized L2 norm)
    # Convert to numpy arrays if needed
    theta_old = np.asarray(theta_old)
    theta_new = np.asarray(theta_new)
    diff = np.linalg.norm(theta_new - theta_old)
    norm = np.linalg.norm(theta_old) + eps
    delta_theta = diff / norm

    converged = delta_theta < tol_theta
    return bool(converged), delta_theta
