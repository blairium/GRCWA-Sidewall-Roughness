import itertools
import multiprocessing
import pathlib
import time
from typing import NamedTuple
from datetime import datetime

import numpy as np
import polars as pl
from RCWA import calculate_first_order_transmission
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.pretty import pprint

# Define the root directory
dirpath: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent.parent


class SimulationParams(NamedTuple):
    """Parameters for a single simulation run."""

    height: float
    roughness: float
    wavelength: float
    period: float


def simulation_task(params: SimulationParams) -> tuple[float, float, float, float, float]:
    """
    Worker function to run a single simulation.

    Args:
        params (SimulationParams): The parameters for the simulation.

    Returns
    -------
        tuple: (height, roughness, wavelength, period, intensity)
    """
    intensity = calculate_first_order_transmission(
        height=params.height,
        sidewall_roughness=params.roughness,
        wavelength=params.wavelength,
        period=params.period,
        # Default values for other parameters are used here.
        # If needed, these could be added to SimulationParams.
    )
    return (params.height, params.roughness, params.wavelength, params.period, intensity)


def run_simulations(
    heights: np.ndarray,
    roughness_values: np.ndarray,
    wavelengths: np.ndarray,
    periods: np.ndarray,
    num_processes: int = 4,
) -> pl.DataFrame:
    """
    Run RCWA simulations for all combinations of the provided parameters.

    Args:
        heights (np.ndarray): Array of grating heights (nm).
        roughness_values (np.ndarray): Array of sidewall roughness values (nm RMS).
        wavelengths (np.ndarray): Array of wavelengths (nm).
        periods (np.ndarray): Array of grating periods (nm).
        num_processes (int, optional): Number of parallel processes to use. Defaults to 4.

    Returns
    -------
        pl.DataFrame: DataFrame containing the results with columns:
                      'height', 'roughness', 'wavelength', 'period', 'intensity'.
    """
    # specific progress bar columns
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        SpinnerColumn(spinner_name="monkey", finished_text="Calculations Complete 🚀"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        expand=True,
    )

    # Generate all combinations of parameters
    # itertools.product returns an iterator, but we need the total length for the progress bar
    # so we convert to a list or calculate the length.
    # Calculating length is cheaper than storing a huge list if it's very large,
    # but for typical simulation parameters, a list is fine.
    combinations = [
        SimulationParams(*args)
        for args in itertools.product(heights, roughness_values, wavelengths, periods)
    ]

    total_sims = len(combinations)
    pprint(f"Starting {total_sims} simulations...")

    results: list[tuple[float, float, float, float, float]] = []

    with progress:
        task_id = progress.add_task("[green]Processing Simulations...", total=total_sims)

        # Use multiprocessing Pool
        with multiprocessing.Pool(processes=num_processes) as pool:
            # imap allows us to process results as they complete and update the progress bar
            # chunksize can be tuned for performance
            for result in pool.imap(simulation_task, combinations, chunksize=10):
                results.append(result)
                progress.advance(task_id)

    # Create DataFrame
    df = pl.DataFrame(
        results, schema=["height", "roughness", "wavelength", "period", "intensity"], orient="row"
    )

    return df


if __name__ == "__main__":
    start_time = time.time()
    today = datetime.today().strftime("%Y-%m-%d")

    # Define parameter ranges for the simulation
    heights = np.arange(5, 100, 2.5)
    roughness_values = np.arange(0, 11, 2)
    wavelengths = np.array([4.23, 6.7, 13.5])
    periods = np.arange(40, 120, 20)

    pprint(f"{multiprocessing.cpu_count()} processors available")
    pprint("Running simulations with:")
    pprint(f"Heights: {heights}")
    pprint(f"Roughness: {roughness_values}")
    pprint(f"Wavelengths: {wavelengths}")
    pprint(f"Periods: {periods}")

    df = run_simulations(
        heights=heights,
        roughness_values=roughness_values,
        wavelengths=wavelengths,
        periods=periods,
        num_processes=4,  # multiprocessing.cpu_count(),
    )

    # Ensure data directory exists
    data_dir = dirpath / "data"
    data_dir.mkdir(exist_ok=True)

    output_path = data_dir / f"{today}_simulation_results.csv"
    df.write_csv(output_path)
    pprint(f"Results saved to {output_path}")
    pprint(f"Total execution time: {time.time() - start_time:.2f} seconds")
