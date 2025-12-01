from scipy.ndimage import gaussian_filter
from scipy.stats import norm
import numpy as np


def correlation_function():
    return None


def sidewall_roughness():
    return None


def apply_roughness(epgrid, sidewall_roughness, period, height):
    """
    Applies surface and sidewall roughness to the grating structure using correlated noise.

    Args:
        epgrid (numpy.ndarray): The permittivity grid of the grating.
        sidewall_roughness (float): The RMS roughness of the sidewall in nm.
        surface_roughness (float): The RMS roughness of the surface in nm.
        period (float): The grating period in nm.
        height (float): The grating height in nm.

    Returns:
        numpy.ndarray: The modified permittivity grid with roughness.
    """
    Nx, Ny = epgrid.shape
    ni_sq_val = epgrid[Nx // 2, Ny // 2]
    vac_sq_val = epgrid[Nx - 1, Ny - 1]

    rough_epgrid = np.copy(epgrid)

    # Sidewall Roughness
    if sidewall_roughness > 0:
        grid_res_x = period / Nx
        grid_res_y = period / Ny  # Assuming square unit cell for correlation length
        corr_len_y = 2.0  # nm, assumed correlation length

        sidewall_rms_pixels = sidewall_roughness / grid_res_x
        corr_pixels_y = corr_len_y / grid_res_y

        # Generate correlated noise for displacement
        def gen_noise(Ny):
            noise = np.random.randn(Ny)
            smooth_noise = gaussian_filter(noise, sigma=corr_pixels_y)
            displacement = smooth_noise * (sidewall_rms_pixels / np.std(smooth_noise))
            return displacement

        original_boundary_x1, original_boundary_x2 = Nx // 4, 3 * Nx // 4
        displacement1 = gen_noise(Ny)
        displacement2 = gen_noise(Ny)
        for y in range(Ny):
            shift1 = int(displacement1[y])
            shift2 = int(displacement2[y])
            new_boundary_x1 = np.clip(original_boundary_x1 + shift1, 0, Nx - 1)
            new_boundary_x2 = np.clip(original_boundary_x2 + shift2, 0, Nx - 1)
            rough_epgrid[new_boundary_x1:new_boundary_x2, y] = ni_sq_val
            rough_epgrid[new_boundary_x2:, y] = vac_sq_val
            rough_epgrid[:new_boundary_x1, y] = vac_sq_val

    return rough_epgrid
