import grcwa
import numpy as np
from data import get_optical_constants
from Roughness import apply_roughness


def calculate_first_order_transmission(
    height,
    mat="Ni",
    wavelength=6.7,
    period=100.0,
    sidewall_roughness=0.0,
    nG=51,
    theta=0.0,
    phi=0.0,
    Nx=1000,
    Ny=1000,
):
    """
    Calculates the first-order transmitted diffraction intensity for a nickel binary grating.

    Args:
        height (float): The height of the grating in nm.

    Returns:
        float: The first-order transmitted intensity.
    """
    grcwa.set_backend("numpy")
    # Grating parameters
    wavelength = 6.7  # nm
    period = 100.0  # nm
    # Refractive index for nickel at 6.7 nm.
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
    epgrid = apply_roughness(epgrid, sidewall_roughness, period, height)

    obj.GridLayer_geteps(epgrid.flatten())

    # Excitation
    planewave = {"p_amp": 0, "s_amp": 1, "p_phase": 0, "s_phase": 0}
    obj.MakeExcitationPlanewave(
        planewave["p_amp"],
        planewave["p_phase"],
        planewave["s_amp"],
        planewave["s_phase"],
        order=0,
    )

    # Solve for transmission
    R_by_order, T_by_order = obj.RT_Solve(byorder=1)

    orders = obj.G

    # Find the index for the (1,0) order
    order_index = np.where(np.all(orders == [1, 0], axis=1))[0][0]

    return T_by_order[order_index]


# if __name__ == "__main__":
# heights = np.linspace(5, 250, 100)
# num_runs = 10
# roughness = np.arange(1, 6, 1)  # nm
# results = {}
# task3 = progress_custom.add_task(
#     "Playing hard...", total=len(heights) * num_runs * len(roughness)
# )
# with progress_custom:
#     for s in roughness:
#         first_order_intensities_ideal, first_order_intensities_rough = run(
#             heights,
#             s,
#             num_runs=num_runs,
#             progress=progress_custom.update(task3, advance=1),
#         )
#         results[s] = [first_order_intensities_ideal, first_order_intensities_rough]

#     # print(
#     #     f"Height: {height:.2f} nm, Ideal Intensity: {ideal_intensity:.4f}, Rough Intensity: {rough_intensity:.4f}"
#     # )

# # Plot the results
# fig, ax = plt.subplots()
# ax.plot(heights, results[1][0], label="Ideal Grating")
# for roughness, data in results.items():
#     ax.scatter(heights, data[1], label=f"{roughness} nm roughness")
# ax.set_xlabel("Grating Height (nm)")
# ax.set_ylabel("First-Order Intensity")
# ax.set_title("First-Order Intensity vs. Grating Height")
# ax.grid(True)
# ax.legend()
# plt.savefig("first_order_intensity.png")
# print("Plot saved to first_order_intensity.png")

# # Generate and plot a schematic of a rough grating for visualization
# # We will use the parameters from the middle of the height range
# example_height = 50.0
# Nx, Ny = 1000, 1000
# period = 100.0
# refractive_index_ni = get_optical_constants("Ni", 6.7)
# refractive_index_vacuum = 1.0

# # Create the ideal grating grid
# epgrid_ideal = np.ones((Nx, Ny), dtype=complex) * refractive_index_vacuum**2
# epgrid_ideal[Nx // 4 : 3 * Nx // 4, :] = refractive_index_ni**2

# # Apply roughness
# epgrid_rough = apply_roughness(epgrid_ideal, s, period, example_height)

# # Plot the schematic
# plot_grating_schematic(epgrid_rough, period, filename=r"grating_schematic.png")
