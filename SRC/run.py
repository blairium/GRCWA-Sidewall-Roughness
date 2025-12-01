from multiprocessing import Pool
from functools import partial
import time
import numpy as np
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)
from RCWA import calculate_first_order_transmission
import polars as pl

progress_custom = Progress(
    TextColumn("[progress.description]{task.description}"),
    SpinnerColumn(spinner_name="monkey", finished_text="Calculations Complete 🚀"),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
    expand=True,
)


def pool_func(height, sidewall_roughness):
    return calculate_first_order_transmission(
        height, sidewall_roughness=sidewall_roughness
    )


def run(
    heights: np.ndarray,
    sidewall_roughness,
    num_runs=1,
    mat="Ni",
    wavelength=6.7,
    period=100.0,
    nG=51,
    theta=0.0,
    phi=0.0,
    Nx=1000,
    Ny=1000,
    progress=None,
):
    first_order_intensities_ideal = np.zeros(len(heights))

    def multi_pool(heights, SR):
        with Pool(4) as p:
            data = p.map(partial(pool_func, sidewall_roughness=SR), heights)

        return data

    first_order_intensities_rough = np.zeros(len(heights))
    df = pl.DataFrame({"grating_height": heights})
    for i, height in enumerate(heights):
        # Ideal grating
        first_order_intensities_ideal[i] += calculate_first_order_transmission(height)
    df.insert_column(1, pl.Series("Ideal", first_order_intensities_ideal))
    for n in range(num_runs):
        # Rough grating
        s = multi_pool(heights, sidewall_roughness)
        df.insert_column(len(df.columns) - 1, pl.Series(f"Run {n}", s))

    return df


if __name__ == "__main__":
    start_time = time.time()
    df = run(
        np.arange(5, 50, 10),
        2,
        num_runs=32,
    )
    print(df)
    print("--- %s seconds ---" % (time.time() - start_time))
