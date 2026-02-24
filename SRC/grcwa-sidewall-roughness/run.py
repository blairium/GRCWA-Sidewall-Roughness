from multiprocessing import Pool
import pathlib
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
import matplotlib.pyplot as plt
from colormaps.cmaps import Cmaps

dirpath: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent.parent 



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
    roughness_values,
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
    print('Starting run:')
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
    # with progress_custom:
    #     task = progress_custom.add_task("[green]Processing...", total=len(roughness_values))
    for roughness in roughness_values:
        # Rough grating
        s = multi_pool(heights, roughness)
        df.insert_column(len(df.columns) - 1, pl.Series(f"{roughness} nm RMS", s))
            # 

    return df


if __name__ == "__main__":
    start_time = time.time()
    with progress_custom:
        task = progress_custom.add_task("[green]Processing...", total=20)
        for nrun in range(20):
            rough_array = np.arange(1,11,.5)
            
            df = run(
                np.arange(5, 150, 2),
                rough_array,
            )
            # print(df)
            
            # colours =Cmaps().glasgow.discrete(len(rough_array)+1)
            # fig,ax = plt.subplots()
            # ax.plot(df[:,0],df[:,-1],label='Ideal')
            # for i,col in enumerate(df.columns[1:-1]):
            #     ax.scatter(df[:,0],df[col],label=col,edgecolors=colours(i),facecolors='none')
            # ax.legend()
            # ax.set_xlabel('Grating Height [nm]')
            # ax.set_ylabel('Diffraction Efficency')
            # figpath: pathlib.Path = dirpath / 'figures' / 'Roughness-de.png'
            # plt.savefig(figpath,dpi=500)
            # plt.show()
            filepath: pathlib.Path = dirpath / "data" / f'roughness-de-data_run_{nrun}.csv'
            df.write_csv(filepath)
            progress_custom.update(task, advance=(1/20))
    print("--- %s seconds ---" % (time.time() - start_time))

