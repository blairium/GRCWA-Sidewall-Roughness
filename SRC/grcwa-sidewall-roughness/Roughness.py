import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.ndimage import gaussian_filter
from scipy.stats import norm


def correlation_function() -> None:
    return None


def sidewall_roughness(sigma: float, r1: np.ndarray, r2: np.ndarray, taux: float, tauz: float) -> np.ndarray:
    """
    Calculates sidewall roughness based on Ban 2025.

    Args:
        sigma (float): RMS roughness.
        r1 (np.ndarray): Position vector 1.
        r2 (np.ndarray): Position vector 2.
        taux (float): Correlation length in x.
        tauz (float): Correlation length in z.

    Returns
    -------
        np.ndarray: Roughness value.
    """
    return np.square(sigma) * np.exp(
        -np.abs(r1[0] - np.square(r2[0])) / np.square(taux)
        + np.abs(r1[1] - np.square(r2[1])) / np.square(tauz)
    )


def generate_correlated_noise(
    n_points: int, rms: float, correlation_length: float, pixel_size: float
) -> np.ndarray:
    """
    Generates correlated noise using a Gaussian filter.

    Args:
        n_points (int): Number of points in the noise array.
        rms (float): The root mean square roughness amplitude.
        correlation_length (float): The correlation length of the roughness.
        pixel_size (float): The physical size of each pixel/grid point.

    Returns
    -------
        numpy.ndarray: 1D array of correlated noise with the specified RMS amplitude.
    """
    if correlation_length <= 0 or pixel_size <= 0:
        return np.random.normal(0, rms, n_points)

    rng = np.random.default_rng()
    x = np.arange(-correlation_length, correlation_length)
    dist = np.sqrt(x**2)

    # Avoid division by zero if correlation_length is extremely small,
    # though the check above handles <= 0.
    filter_kernel = np.exp(-dist**2 / (2 * correlation_length))

    noise = rng.standard_normal(n_points)

    noise = scipy.signal.fftconvolve(noise, filter_kernel, mode="same")

    # Rescale to match desired RMS
    current_std = np.std(noise)
    if current_std == 0:
        return np.zeros(n_points)

    displacement = noise * (rms / current_std)
    return displacement


def apply_roughness(
    epgrid: np.ndarray,
    sidewall_roughness: float,
    period: float,
    height: float,
    surface_roughness: float = 0.0,
    correlation_length: float = 100.0,
) -> np.ndarray:
    """
    Applies surface and sidewall roughness to the grating structure using correlated noise.

    Args:
        epgrid (numpy.ndarray): The permittivity grid of the grating.
        sidewall_roughness (float): The RMS roughness of the sidewall in nm.
        period (float): The grating period in nm.
        height (float): The grating height in nm.
        surface_roughness (float, optional): The RMS roughness of the surface in nm. Not currently implemented for single-layer structures. Defaults to 0.0.
        correlation_length (float, optional): The correlation length of the roughness in nm. Defaults to 100.0.

    Returns
    -------
        numpy.ndarray: The modified permittivity grid with roughness.
    """
    Nx, Ny = epgrid.shape
    ni_sq_val = epgrid[Nx // 2, Ny // 2]
    vac_sq_val = epgrid[Nx - 1, Ny - 1]

    rough_epgrid = np.copy(epgrid)

    # Sidewall Roughness
    if sidewall_roughness > 0:
        grid_res_y = period / Ny
        # Assuming square unit cell or period in y matches correlation length scale
        pixel_size_y = period / Ny

        # Determine boundaries
        # Assuming the structure is roughly centered or symmetric as in the original code
        original_boundary_x1, original_boundary_x2 = Nx // 4, 3 * Nx // 4

        # Generate correlated noise for displacements
        # The scale factor for RMS depends on pixel size in X because the shift is in integer pixels.
        # shift * period/Nx = physical_shift. We want physical_shift RMS to be sidewall_roughness.
        # So shift RMS should be sidewall_roughness / (period/Nx).

        pixel_size_x = period / Nx
        rms_pixels = sidewall_roughness / pixel_size_x

        displacement1 = generate_correlated_noise(
            Ny, rms_pixels, correlation_length, pixel_size_y
        )
        displacement2 = generate_correlated_noise(
            Ny, rms_pixels, correlation_length, pixel_size_y
        )

        # Apply displacements
        for y in range(Ny):
            shift1 = int(displacement1[y])
            shift2 = int(displacement2[y])

            new_boundary_x1 = np.clip(original_boundary_x1 + shift1, 0, Nx - 1)
            new_boundary_x2 = np.clip(original_boundary_x2 + shift2, 0, Nx - 1)

            # Fill the middle with material
            rough_epgrid[new_boundary_x1:new_boundary_x2, y] = ni_sq_val
            # Fill outside with vacuum
            rough_epgrid[new_boundary_x2:, y] = vac_sq_val
            rough_epgrid[:new_boundary_x1, y] = vac_sq_val

    if surface_roughness > 0:
        print(
            "Warning: Surface roughness > 0 requested but not implemented for single-layer grid modification."
        )

    return rough_epgrid


def ban(
    rms: float,
    x1: np.ndarray,
    x2: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    corr_x: float,
    corr_y: float,
) -> np.ndarray:
    return rms**2 * np.exp(
        -((np.abs(x1 - x2) ** 2) / (corr_x**2)) + ((np.abs(y1 - y2) ** 2) / corr_y**2)
    )


def autocorrelation(
    rms: float, z_dist: float, corr_len: float, roughness_index: float
) -> float:
    """
    #1 in https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-22-40413
    """
    return rms**2 * np.exp(-(z_dist / corr_len) ** (2 * roughness_index))


def PSD(rms: float, corr_len: float, grating_thickness: float, j: int) -> float:
    """
    #2 in https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-22-40413
    """
    return (
        np.sqrt(np.pi)
        * rms**2
        * corr_len
        * np.exp(-(np.pi * (j / grating_thickness) * corr_len) ** 2)
    )


def wu_sidewall_roughness_model(H: float, N: int, fj: float) -> None:
    """
    https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-22-40413
    """
