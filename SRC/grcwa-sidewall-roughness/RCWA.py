import grcwa
import numpy as np
from data import get_optical_constants
from plotting import plot_grating_schematic
from Roughness import apply_roughness


def calculate_first_order_transmission(
    height: float,
    mat: str = "Ni",
    wavelength: float = 6.7,
    period: float = 100.0,
    sidewall_roughness: float = 0.0,
    nG: int = 51,
    theta: float = 0.0,
    phi: float = 0.0,
    Nx: int = 1000,
    Ny: int = 1000,
    correlation_length: float = 100.0,
) -> float:
    """
    Calculates the first-order transmitted diffraction intensity for a binary grating.

    Args:
        height (float): The height of the grating in nm.
        mat (str, optional): Material name for optical constants. Defaults to "Ni".
        wavelength (float, optional): Wavelength in nm. Defaults to 6.7.
        period (float, optional): Grating period in nm. Defaults to 100.0.
        sidewall_roughness (float, optional): RMS roughness of the sidewall in nm. Defaults to 0.0.
        nG (int, optional): Number of Fourier harmonics. Defaults to 51.
        theta (float, optional): Incident angle theta. Defaults to 0.0.
        phi (float, optional): Incident angle phi. Defaults to 0.0.
        Nx (int, optional): Grid size in x. Defaults to 1000.
        Ny (int, optional): Grid size in y. Defaults to 1000.

    Returns
    -------
        float: The first-order transmitted intensity.
    """
    grcwa.set_backend("numpy")

    # Refractive index for material at specified wavelength.
    refractive_index_ni = get_optical_constants(mat, wavelength)
    refractive_index_vacuum = 1.0

    # GRCWA setup
    L1 = [period, 0]
    L2 = [0, period]

    freq = 1.0 / wavelength

    obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)

    # Layer definition
    obj.Add_LayerUniform(0, refractive_index_vacuum**2)
    obj.Add_LayerGrid(height, Nx, Ny)
    obj.Add_LayerUniform(0, refractive_index_vacuum**2)
    obj.Init_Setup()

    # Grating pattern (binary with 50% fill factor)
    epgrid = np.ones((Nx, Ny), dtype=complex) * refractive_index_vacuum**2
    epgrid[Nx // 4 : 3 * Nx // 4, :] = refractive_index_ni**2

    # Apply roughness
    epgrid = apply_roughness(epgrid, sidewall_roughness, period, height, 0, correlation_length)

    obj.GridLayer_geteps(epgrid.flatten())

    # Excitation
    planewave = {"p_amp": 0, "s_amp": 1, "p_phase": 0, "s_phase": 0}
    obj.MakeExcitationPlanewave(
        planewave["p_amp"], planewave["p_phase"], planewave["s_amp"], planewave["s_phase"], order=0
    )

    # Solve for transmission
    R_by_order, T_by_order = obj.RT_Solve(byorder=1)

    orders = obj.G

    # Find the index for the (1,0) order
    order_index = np.where(np.all(orders == [1, 0], axis=1))[0][0]

    return float(T_by_order[order_index])


if __name__ == "__main__":
    # Generate and plot a schematic of a rough grating for visualization
    # We will use the parameters from the middle of the height range
    example_height = 50.0
    Nx, Ny = 1000, 1000
    period = 100.0
    refractive_index_ni = get_optical_constants("Ni", 6.7)
    refractive_index_vacuum = 1.0

    # Create the ideal grating grid
    epgrid_ideal = np.ones((Nx, Ny), dtype=complex) * refractive_index_vacuum**2
    epgrid_ideal[Nx // 4 : 3 * Nx // 4, :] = refractive_index_ni**2

    # Apply roughness
    epgrid_rough = apply_roughness(epgrid_ideal, 3, period, example_height)

    # Plot the schematic
    plot_grating_schematic(epgrid_rough, period, filename=r"grating_schematic.png")
