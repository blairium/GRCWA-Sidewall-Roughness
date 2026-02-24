import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import polars as pl
from colormaps.cmaps import Cmaps
plt.style.use('thesis')

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

def plot_roughness_effect():
    scriptdir = Path(__file__).resolve().parent # Get parent dir
    datadir = scriptdir.parent.parent / 'data'  # Get data dir location relative to script
    files = []                                  # Initiate list to hold file paths
    col_dict = {}                               # Initiate dict to hold individual values for mean/std
    for file_path in datadir.glob("*run_*.csv"):# Get file paths and populate list
        files.append(file_path)
    # Get columns for dict
    df = pl.read_csv(files[0])
    for col in df.columns[1:-1]:
        col_dict[col] = np.zeros((len(df[col]),len(files)))
        col_dict[col][:,0] = df[col].to_numpy()
    #Popupate dict of lists for each run
    for i,file in enumerate(files):
        df = pl.read_csv(file)
        for col in df.columns[1:-1]:
            col_dict[col][:,i] = df[col].to_numpy()
    # Average values in list
    mean_dict = {}
    for key,value in col_dict.items():
        mean_dict[key] = np.mean(value,axis=1)


    df_mean = pl.DataFrame(mean_dict)
    # Average values in list
    stdev_dict = {}
    for key,value in col_dict.items():
        stdev_dict[key] = np.std(value,axis=1)


    df_stdev = pl.DataFrame(stdev_dict)
    colours =Cmaps().glasgow.discrete(len(mean_dict.keys())+1)
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(df['grating_height'],df['Ideal'],label='Ideal',color=colours(0))
    for i,col in enumerate(df_mean.columns[2:]):
        if i % 2 == 1:
            pass
        else:
            ax.errorbar(df['grating_height'],df_mean[col],yerr=df_stdev[col] ,label=col,marker='o',linestyle='none',mfc='none',mec=colours(i+1),capsize=3)
    ax.set_ylabel('Diffraction Efficiency')
    ax.set_xlabel('Grating Height')
    ax.legend(title='Roughness')
    plt.savefig('de-roughness-plot-185eV-Ni-100nm-HP.png',dpi=500)
    plt.show()








if __name__ == "__main__":
    plot_roughness_effect()
