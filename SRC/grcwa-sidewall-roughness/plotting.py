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

    if not wavelengths or not periods:
        print("No data found.")
        return

    w = wavelengths[0]
    p = periods[0]

    # Filter for specific wavelength and period
    subset = df.filter((pl.col("wavelength") == w) & (pl.col("period") == p))

    # Group by height and roughness to calculate mean and std dev
    # We group by 'height' and 'roughness' (and keep 'wavelength' and 'period' which are constant here)
    grouped = subset.group_by(["height", "roughness"]).agg(
        [
            pl.col("intensity").mean().alias("mean_intensity"),
            pl.col("intensity").std().alias("std_intensity"),
        ]
    )

    plt.figure(figsize=(10, 6))

    for r in roughnesses:
        data = grouped.filter(pl.col("roughness") == r).sort("height")

        # If std_intensity is null (e.g. only 1 run), fill with 0
        mean_intensity = data["mean_intensity"]
        std_intensity = data["std_intensity"].fill_null(0.0)

        plt.errorbar(
            data["height"],
            mean_intensity,
            yerr=std_intensity,
            label=f"Roughness {r} nm",
            marker="o",
            capsize=5,
            linestyle="-",
        )

    plt.xlabel("Height (nm)")
    plt.ylabel("Mean Intensity")
    plt.title(f"Diffraction Efficiency vs Height\n(Wavelength={w} nm, Period={p} nm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
