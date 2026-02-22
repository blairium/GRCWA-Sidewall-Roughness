from scipy.ndimage import gaussian_filter
from scipy.stats import norm
import numpy as np
import sympy
import matplotlib.pyplot as plt


def correlation_function():
    return None


def sidewall_roughness(sigma, r1, r2, taux, tauz):
    """
    From Ban 2025
    """

    return np.square(sigma) * np.exp(
        -np.abs(r1[0] - np.square(r2[0])) / np.square(taux)
        + np.abs(r1[1] - np.square(r2[1])) / np.square(tauz)
    )


print(sidewall_roughness(0.4, np.array([10, 10]), np.array([11, 14]), 3, 4))


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


# 1. Define Parameters
num_points = 1000
# Generate white noise (Gaussian distribution)
np.random.seed(42)  # For reproducibility
roughness = np.random.normal(0, 1, num_points)

# 2. Simulate Roughness
# Option A: White Noise Profile (very rough)
profile_white = roughness

# Option B: Cumulative Sum (Brownian Noise / Random Walk - smoother)
profile_walk = np.cumsum(roughness)

# 3. Plotting
plt.figure(figsize=(10, 5))
plt.plot(profile_walk, label="1D Rough Profile (Random Walk)", color="blue")
plt.title("Simulated 1D Rough Line")
plt.xlabel("Position")
plt.ylabel("Elevation/Height")
plt.grid(True)
plt.legend()
plt.show()
