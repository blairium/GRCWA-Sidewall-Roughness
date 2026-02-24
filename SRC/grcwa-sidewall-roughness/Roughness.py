from scipy.ndimage import gaussian_filter
from scipy.stats import norm
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def correlation_function():
    return None


def sidewall_roughness(sigma, r1, r2, taux, tauz):
    """python
    From Ban 2025
    """

    return np.square(sigma) * np.exp(
        -np.abs(r1[0] - np.square(r2[0])) / np.square(taux)
        + np.abs(r1[1] - np.square(r2[1])) / np.square(tauz)
    )


# print(sidewall_roughness(0.4, np.array([10, 10]), np.array([11, 14]), 3, 4))
def fast_generate(correlation_length,):
    #https://stackoverflow.com/questions/63816481/faster-method-for-creating-spatially-correlated-noise?rq=3
    x = np.arange(-correlation_length, correlation_length)
    dist = np.sqrt(x**2)
    filter_kernel = np.exp(-dist**2/(2*correlation_length))

    # Generate n-by-n grid of spatially correlated noise
    n = 50
    noise = np.random.randn(n)
    return scipy.signal.fftconvolve(noise, filter_kernel, mode='same')

def generate_correlated_noise(n_points, rms, correlation_length, pixel_size):
    """
    Generates correlated noise using a Gaussian filter.

    Args:
        n_points (int): Number of points in the noise array.
        rms (float): The root mean square roughness amplitude.
        correlation_length (float): The correlation length of the roughness.
        pixel_size (float): The physical size of each pixel/grid point.

    Returns:
        numpy.ndarray: 1D array of correlated noise with the specified RMS amplitude.
    """
    if correlation_length <= 0 or pixel_size <= 0:
        return np.random.normal(0, rms, n_points)
    rng = np.random.default_rng()
    x = np.arange(-correlation_length, correlation_length)
    dist = np.sqrt(x**2)
    filter_kernel = np.exp(-dist**2/(2*correlation_length))

    noise = rng.standard_normal(n_points)

    noise =  scipy.signal.fftconvolve(noise, filter_kernel, mode='same')



    # sigma_pixels = correlation_length / pixel_size
    # noise = np.random.randn(n_points)
    # smooth_noise = gaussian_filter(noise, sigma=sigma_pixels)

    # Rescale to match desired RMS
    current_std = np.std(noise)
    if current_std == 0:
        return np.zeros(n_points)

    displacement = noise * (rms / current_std) 
    return displacement


def apply_roughness(epgrid, sidewall_roughness, period, height, surface_roughness=0.0, correlation_length=100.0):
    """
    Applies surface and sidewall roughness to the grating structure using correlated noise.

    Args:
        epgrid (numpy.ndarray): The permittivity grid of the grating.
        sidewall_roughness (float): The RMS roughness of the sidewall in nm.
        period (float): The grating period in nm.
        height (float): The grating height in nm.
        surface_roughness (float): The RMS roughness of the surface in nm. Not currently implemented for single-layer structures.
        correlation_length (float): The correlation length of the roughness in nm. Defaults to 2.0.

    Returns:
        numpy.ndarray: The modified permittivity grid with roughness.
    """
    Nx, Ny = epgrid.shape
    ni_sq_val = epgrid[Nx // 2, Ny // 2]
    vac_sq_val = epgrid[Nx - 1, Ny - 1]

    rough_epgrid = np.copy(epgrid)

    # Sidewall Roughness
    if sidewall_roughness > 0:
        grid_res_y = period / Ny  # Assuming square unit cell or period in y matches correlation length scale
        # Wait, if period is in x? No, L1=[period, 0], L2=[0, period] usually means square lattice.
        # But if the grating is 1D line along y, it is periodic in x.
        # The correlation length applies along the line (y-direction).
        # So resolution along y is period_y / Ny.
        # But period is passed as a single value. Assuming square unit cell:
        pixel_size_y = period / Ny

        # Determine boundaries
        # Assuming the structure is roughly centered or symmetric as in the original code
        original_boundary_x1, original_boundary_x2 = Nx // 4, 3 * Nx // 4

        # Generate correlated noise for displacements
        displacement1 = generate_correlated_noise(Ny, sidewall_roughness / (period / Nx), correlation_length, pixel_size_y)
        displacement2 = generate_correlated_noise(Ny, sidewall_roughness / (period / Nx), correlation_length, pixel_size_y)

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
        print("Warning: Surface roughness > 0 requested but not implemented for single-layer grid modification.")

    return rough_epgrid

def ban(rms,x1,x2,y1,y2,corr_x,corr_y):
    return rms**2 * np.exp(-((np.abs(x1-x2)**2)/(corr_x**2))+((np.abs(y1-y2)**2)/corr_y**2))


def autocorrelation(rms,z_dist,corr_len,roughness_index):
    '''
    #1 in https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-22-40413
    '''
    return rms**2 * np.exp(-(z_dist/corr_len)**(2*roughness_index))

def PSD(rms,corr_len,grating_thickness,j):
    '''
    #2 in https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-22-40413
    '''
    return np.sqrt(np.pi)*rms**2 * corr_len * np.exp(-(np.pi*(j/grating_thickness)*corr_len)**2)


def wu_sidewall_roughness_model(H,N,fj):
    '''
    https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-22-40413
    '''






if __name__ == "__main__":
    fast_generate()
# 1. Define Parameters
# num_points = 1000
# # Generate white noise (Gaussian distribution)
# np.random.seed(42)  # For reproducibility
# roughness = np.random.normal(0, 1, num_points)

# # 2. Simulate Roughness
# # Option A: White Noise Profile (very rough)
# profile_white = roughness

# # Option B: Cumulative Sum (Brownian Noise / Random Walk - smoother)
# profile_walk = np.cumsum(roughness)

# # 3. Plotting
# plt.figure(figsize=(10, 5))
# plt.plot(profile_walk, label="1D Rough Profile (Random Walk)", color="blue")
# plt.title("Simulated 1D Rough Line")
# plt.xlabel("Position")
# plt.ylabel("Elevation/Height")
# plt.grid(True)
# plt.legend()
# plt.show()
