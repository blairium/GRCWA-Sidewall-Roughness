from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from thesistools.plotting import panel_labeller, add_subfig_label

from colormaps.cmaps import Cmaps

plt.style.use("thesis")


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
            linestyle="none",
        )

    plt.xlabel("Height (nm)")
    plt.ylabel("Mean Intensity")
    plt.title(f"Diffraction Efficiency vs Height\n(Wavelength={w} nm, Period={p} nm)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_simulation_results_all(csv_path: str | Path) -> None:
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

    labeller = panel_labeller()
    fig, axes = plt.subplots(
        len(wavelengths), len(periods), figsize=(12, 8), sharey=True, layout="constrained"
    )

    for i, w in enumerate(wavelengths):
        for j, p in enumerate(periods):
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

            for r in roughnesses:
                data = grouped.filter(pl.col("roughness") == r).sort("height")
                ideal = grouped.filter(pl.col("roughness") == 0).sort("height")

                # If std_intensity is null (e.g. only 1 run), fill with 0
                mean_intensity = data["mean_intensity"]
                std_intensity = data["std_intensity"].fill_null(0.0)

                axes[i][j].errorbar(
                    data["height"],
                    (mean_intensity),
                    yerr=(std_intensity),
                    label=f"Roughness {r} nm",
                    marker="o",
                    capsize=5,
                    linestyle="none",
                )
                axes[i][j].set_title(
                    f"{r'$\eta$'} vs {r'$G_H$'}\n({r'$\lambda$'}={w} nm, P={p} nm)", fontsize=8
                )
    for ax in axes.flatten():
        add_subfig_label(ax, labeller.next())
        ax.set_xlabel("Height (nm)")
        ax.set_ylabel("Intensity")
        ax.set_ylim(0, 0.18)
        ax.legend()

    plt.show()


def plot_simulation_results_slice(csv_path: str | Path, gh: np.ndarray | list) -> None:
    """plot_simulation_results_slice _summary_

    Parameters
    ----------
    csv_path : str | Path
        _description_
    gh : np.ndarray | list
        _description_
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

    labeller = panel_labeller()
    fig, axes = plt.subplots(
        len(wavelengths), len(periods), figsize=(12, 8), sharey=True, layout="constrained"
    )
    colours = Cmaps().glasgow.discrete(len(gh))
    # cmap = mpl.colormaps["plasma"]

    for i, w in enumerate(wavelengths):
        for j, p in enumerate(periods):
            # Filter for specific wavelength and period
            for k, h in enumerate(gh):
                subset = df.filter(
                    (pl.col("wavelength") == w) & (pl.col("period") == p) & (pl.col("height") == h)
                )

                # Group by height and roughness to calculate mean and std dev
                # We group by 'height' and 'roughness' (and keep 'wavelength' and 'period' which are constant here)
                grouped = subset.group_by(["height", "roughness"]).agg(
                    [
                        pl.col("intensity").mean().alias("mean_intensity"),
                        pl.col("intensity").std().alias("std_intensity"),
                    ]
                )

                intensity = []
                intensity_err = []
                for r in roughnesses:
                    data = grouped.filter(pl.col("roughness") == r).sort("height")
                    ideal = grouped.filter(pl.col("roughness") == 0).sort("height")

                    # If std_intensity is null (e.g. only 1 run), fill with 0
                    mean_intensity = data["mean_intensity"]
                    std_intensity = data["std_intensity"].fill_null(0.0)

                    intensity.append(np.array(mean_intensity.to_numpy())[0])

                    intensity_err.append(np.array(std_intensity.to_numpy())[0])

                axes[i][j].errorbar(
                    roughnesses,
                    intensity,
                    yerr=intensity_err,
                    label=f"{r'$g_h$'} = {h} nm",
                    color=colours(k),
                    marker="o",
                    alpha=0.75,
                    capsize=5,
                    linestyle="none",
                )
                axes[i][j].set_title(
                    f"{r'$\eta$'} vs {r'$G_H$'}\n({r'$\lambda$'}={w} nm, P={p} nm)", fontsize=8
                )

    for ax in axes.flatten():
        add_subfig_label(ax, labeller.next())
        ax.set_xlabel("Sidewall Roughness (nm)")
        ax.set_ylabel("First-order Diffracted Flux Intensity")
        # ax.set_ylim(0, 0.18)
        ax.legend()
    # fig.suptitle(f"{ghv} nm grating height")
    plt.show()


if __name__ == "__main__":
    # df = pl.read_csv("./data/2026-02-26-simulation_results.csv")

    # wavelengths = df["wavelength"].unique().to_list()
    # periods = df["period"].unique().to_list()
    # roughnesses = df["roughness"].unique().to_list()
    # w = wavelengths[-1]
    # p = periods[-1]
    # h = df["height"].unique().to_list()[-1]
    # # Filter for specific wavelength and period
    # subset = df.filter(
    #     (pl.col("wavelength") == w) & (pl.col("period") == p) & (pl.col("height") == h)
    # )
    # print(subset)
    # subset = df.filter((pl.col("wavelength") == w) & (pl.col("period") == p))
    # grouped = subset.group_by(["height", "roughness"]).agg(
    #     [
    #         pl.col("intensity").mean().alias("mean_intensity"),
    #         pl.col("intensity").std().alias("std_intensity"),
    #     ]
    # )
    # print(grouped)
    gharray = [20, 40, 60, 80]
    plot_simulation_results_slice(
        "./data/2026-02-26-simulation_results_250nm_corr_len.csv", gharray
    )
