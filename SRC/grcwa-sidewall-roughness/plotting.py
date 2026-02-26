from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plot_grating_schematic(
    epgrid: np.ndarray, period: float, filename: str = "grating_schematic.png"
) -> None:
    """
    Plots a schematic of the grating structure with axes in nanometers.

    Args:
        epgrid (numpy.ndarray): The permittivity grid of the grating.
        period (float): The grating period in nm.
        filename (str, optional): The name of the file to save the plot to. Defaults to "grating_schematic.png".
    """
    plt.figure(figsize=(8, 8))
    # We plot the real part of the permittivity grid.
    plt.imshow(np.real(epgrid.T), origin="lower", cmap="viridis", extent=[0, period, 0, period])
    plt.colorbar(label="Real part of Permittivity")
    plt.xlabel("x (nm)")
    plt.ylabel("y (nm)")
    plt.title("Grating Structure Schematic (Unit Cell)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(filename)
    plt.close()


def plot_simulation_results(csv_path: str | Path) -> None:
    """
    Plots the simulation results from the CSV file.

    Args:
        csv_path (str | Path): Path to the results CSV file.
    """
    df = pl.read_csv(csv_path)

    # Check if we have multiple wavelengths/periods to decide how to plot
    wavelengths = df["wavelength"].unique().to_list()
    periods = df["period"].unique().to_list()
    roughnesses = df["roughness"].unique().to_list()

    # Simple plot: Intensity vs Height for different roughnesses (for first wavelength/period)
    # This is just a demonstration plot

    w = wavelengths[0]
    p = periods[0]

    subset = df.filter((pl.col("wavelength") == w) & (pl.col("period") == p))

    plt.figure(figsize=(10, 6))

    for r in roughnesses:
        data = subset.filter(pl.col("roughness") == r).sort("height")
        plt.plot(data["height"], data["intensity"], label=f"Roughness {r} nm", marker="o")

    plt.xlabel("Height (nm)")
    plt.ylabel("Intensity")
    plt.title(f"Diffraction Efficiency vs Height\n(Wavelength={w} nm, Period={p} nm)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_simulation_results("./data/simulation_results.csv")
