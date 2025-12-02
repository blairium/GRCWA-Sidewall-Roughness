import matplotlib.pyplot as plt
import numpy as np


def plot_grating_schematic(epgrid, period, filename="grating_schematic.png"):
    """
    Plots a schematic of the grating structure with axes in nanometers.

    Args:
        epgrid (numpy.ndarray): The permittivity grid of the grating.
        period (float): The grating period in nm.
        filename (str): The name of the file to save the plot to.
    """
    plt.figure(figsize=(8, 8))
    # We plot the real part of the permittivity grid.
    plt.imshow(
        np.real(epgrid.T), origin="lower", cmap="viridis", extent=[0, period, 0, period]
    )
    plt.colorbar(label="Real part of Permittivity")
    plt.xlabel("x (nm)")
    plt.ylabel("y (nm)")
    plt.title("Grating Structure Schematic (Unit Cell)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(filename)
    plt.close()
