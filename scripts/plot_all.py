"""Script to plot various network metrics and their differences over time."""

import os

import copy  # Important for modifying the colormap safely

from pathlib import Path

from clustering import clustering_coefficients, feed_forward_loop, yearly_clustering, yearly_ffl

from deforestation import yearly_deforestation_in_degrees, yearly_deforestation_out_degrees

from inout import in_degrees, out_degrees, yearly_in_degrees, yearly_out_degrees

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import numpy as np

import xarray as xr


CARTOPY_AVAILABLE = False
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    print("Warning: 'cartopy' module not found.")
    print("To see coastlines and borders, install it via: 'pip install cartopy'")
    print("Falling back to standard plotting mode...\n")


CONNECTION_THRESHOLD = 1.14  # Threshold for strong connections


# --- Amazon Map Setup Functions ---
def setup_amazon_map(fig_size=(12, 8)):
    """Create an Amazon map with coastlines and borders.

    Args:
        fig_size (tuple): Size of the figure.

    Returns:
        tuple: (fig, ax1, ax2, kwargs) - The figure, the two axes, and the transform kwargs.

    """
    fig = plt.figure(figsize=fig_size)

    if CARTOPY_AVAILABLE:
        ax = plt.axes(projection=ccrs.PlateCarree())

        ax.set_extent([-81, -48, -22.5, 7.5], crs=ccrs.PlateCarree())

        COASTLINES = cfeature.NaturalEarthFeature(
            'physical', 'coastline', '50m', edgecolor='black', facecolor='none'
        )
        BORDERS = cfeature.NaturalEarthFeature(
            'cultural', 'admin_0_countries', '50m', edgecolor='black', facecolor='none'
        )

        ax.add_feature(COASTLINES, linewidth=1)
        ax.add_feature(BORDERS, linestyle='-', linewidth=0.5)

        kwargs = {'transform': ccrs.PlateCarree()}

        return fig, ax, kwargs

    else:
        print("Cartopy not available. Returning standard axes.")
        ax = plt.axes()
        kwargs = {}
        return fig, ax, kwargs


def setup_double_amazon_map(fig_size=(22, 8)):
    """Create an Amazon map with two subplots side by side.

    Args:
        fig_size (tuple): Size of the figure.

    Returns:
        tuple: (fig, ax1, ax2, kwargs) - The figure, the two axes, and the transform kwargs.

    """
    if CARTOPY_AVAILABLE:
        projection = ccrs.PlateCarree()
        kwargs = {'transform': projection}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size,
                                       subplot_kw={'projection': projection})

        COASTLINES = cfeature.NaturalEarthFeature(
            'physical', 'coastline', '50m', edgecolor='black', facecolor='none'
        )
        BORDERS = cfeature.NaturalEarthFeature(
            'cultural', 'admin_0_countries', '50m', edgecolor='black', facecolor='none'
        )

        ax1.set_extent([-81, -48, -22.5, 7.5], crs=projection)
        ax1.add_feature(COASTLINES, linewidth=1)
        ax1.add_feature(BORDERS, linestyle='-', linewidth=0.5)

        ax2.set_extent([-81, -48, -22.5, 7.5], crs=projection)
        ax2.add_feature(COASTLINES, linewidth=1)
        ax2.add_feature(BORDERS, linestyle='-', linewidth=0.5)

        return fig, ax1, ax2, kwargs

    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
        print("Cartopy not available. Returning standard axes.")
        kwargs = {}
        return fig, ax1, ax2, kwargs


# --- Individual Plot Functions ---
def plot_precipitation(scenario: int, year: int, month: int) -> None:
    """Plot precipitation values per node.

    Args:
        scenario (int): The SSP scenario identifier (e.g., "585").
        year (int): The year to visualize.
        month (int): The month number (1-12).

    Returns:
        None. Displays the plot.

    """
    filepath = f"../data/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    ds = xr.open_dataset(filepath)

    fig, ax, kwargs = setup_amazon_map(fig_size=(12, 8))

    sc = ax.scatter(
        x=ds["lon"],
        y=ds["lat"],
        c=ds["prec"],
        cmap="viridis",
        s=420,
        marker="s",
        vmin=0,
        vmax=450,
        **kwargs
    )

    plt.colorbar(sc, ax=ax, label="Precipitation")
    plt.title(f"Precipitation - SSP {scenario}, Year: {year}, Month: {month:02d}", fontsize=16)

    out_path = f"../results/plots/precipitation_{scenario}_{year}_{month:02d}.png"

    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(out_path)
    plt.show()
    plt.close(fig)


def plot_MAP_with_deforestation(scenario: int) -> None:
    """Plot Mean Annual Precipitation (MAP) comparison directly from data files.

    Args:
        scenario (int): The SSP scenario identifier (e.g., "245", "370", "585").

    Returns:
        None. Displays and saves the plot.
    """
    years_range = range(2030, 2051)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    colors = ["#ff9900", "#ff0000", "#8400ff"]

    for scenario in [245, 370, 585]:
        MAP_forest = []
        MAP_deforest = []

        print(f"Lese Daten für Szenario SSP{scenario}...")

        for y in years_range:
            for m in range(1, 13):
                filepath = f"../data/water/scenario_ssp{scenario}_decade{y}_month{m:02d}.nc"
                ds = xr.open_dataset(filepath)
                if m == 1:
                    prec_sum = ds["prec"].values
                else:
                    prec_sum += ds["prec"].values

            values_no_def = prec_sum.sum()/416
            values_def = (prec_sum.sum() + yearly_deforestation_in_degrees(scenario, y).sum()
                          - yearly_in_degrees(scenario, y).sum())/416

            MAP_forest.append(values_no_def)
            MAP_deforest.append(values_def)

        ax.set_ylim(1500, 2000)
        # Without Deforestation
        ax.plot(
            years_range,
            MAP_forest,
            label="No Deforestation - Scenario SSP"+str(scenario),
            color=colors[[245, 370, 585].index(scenario)],
            linestyle="-",
            alpha=0.9
        )

        # With Deforestation
        ax.plot(
            years_range,
            MAP_deforest,
            label="With Deforestation - Scenario SSP"+str(scenario),
            color=colors[[245, 370, 585].index(scenario)],
            linestyle=":",
            alpha=0.9
        )

    # Savannization Threshold Line
    ax.axhline(y=1800,
               color="#8A7102",
               linestyle="--",
               linewidth=2,
               alpha=0.5,
               label="Bistable Threshold (1800 mm)")

    title = "Mean Annual Precipitation Comparison with Deforestation"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.xticks(range(2030, 2051, 5))
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Annual Precipitation [mm]")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_path = "../results/MAP_Comparison.png"
    plt.savefig(output_path)
    print(f"Plot erstellt: {output_path}")

    plt.show()
    plt.close()


def plot_evaporation(scenario: int, year: int, month: int) -> None:
    """Plot evaporation values per node.

    Args:
        scenario (int): The SSP scenario identifier.
        year (int): The year to visualize.
        month (int): The month number (1-12).

    Returns:
        None. Displays the plot.

    """
    filepath = f"../data/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    ds = xr.open_dataset(filepath)

    fig, ax, kwargs = setup_amazon_map(fig_size=(12, 8))

    sc = ax.scatter(
        x=ds["lon"],
        y=ds["lat"],
        c=ds["evap"],
        cmap="viridis",
        s=420,
        marker="s",
        vmin=0,
        vmax=130,
        **kwargs
    )

    plt.colorbar(sc, ax=ax, label="Evaporation")
    plt.title(f"Evaporation - SSP {scenario}, Year: {year}, Month: {month:02d}", fontsize=16)

    out_path = f"../results/plots/evaporation_{scenario}_{year}_{month:02d}.png"

    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(out_path)
    plt.show()
    plt.close(fig)


def plot_network(
    scenario: int, year: int, month: int, CONNECTION_THRESHOLD=CONNECTION_THRESHOLD
) -> None:
    """Plot the strong network connections.

    Args:
        scenario (int): The SSP scenario identifier.
        year (int): The year to visualize.
        month (int): The month number (1-12).
        CONNECTION_THRESHOLD (float): Threshold to filter strong connections.

    Returns:
        None. Displays the plot.

    """
    filepath = f"../data/water/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    ds = xr.open_dataset(filepath)

    lon_np = ds["lon"].values
    lat_np = ds["lat"].values
    network_np = ds["network"].values

    print(f"Loaded data, finding connections stronger than {CONNECTION_THRESHOLD}...")

    # --- Find all connections above the threshold ---
    i_indices, j_indices = np.where(network_np > CONNECTION_THRESHOLD)

    print(f"Found {len(i_indices)} connections matching the criteria.")

    fig, ax, kwargs = setup_amazon_map(fig_size=(8, 8))

    if len(i_indices) > 0:
        for i, j in zip(i_indices, j_indices):
            start_point = (lon_np[j], lat_np[j])
            end_point = (lon_np[i], lat_np[i])

            # Skip connections from a node to itself
            if i == j:
                continue

            # Create arrows showing the direction of the connection
            ax.annotate(
                "",
                xy=end_point,
                xytext=start_point,
                arrowprops=dict(
                    arrowstyle="-|>", color="blue", linewidth=0.65, alpha=0.3
                ),
                **kwargs
            )
    else:
        print(
            "Warning: No connections found above the threshold. Only points will be plotted."
        )

    # --- Plot Nodes on Top ---
    ax.scatter(lon_np, lat_np, c="red", s=4, zorder=10)

    ax.set_title(f"Moisture flows (Connections > {CONNECTION_THRESHOLD})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)

    out_path = f"../results/plots/network_{scenario}_{year}_{month:02d}.png"

    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(out_path)
    plt.show()
    plt.close(fig)


def plot_degrees(scenario: int, year: int, month: int) -> None:
    """Plot in-degrees and out-degrees per node.

    Args:
        scenario (int): The SSP scenario identifier.
        year (int): The year to visualize.
        month (int): The month number (1-12).

    Returns:
        None. Saves/displays the plot.

    """
    filepath = f"../data/water/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    ds = xr.open_dataset(filepath)

    fig, ax1, ax2, kwargs = setup_double_amazon_map(fig_size=(22, 8))

    fig.suptitle(
        f"Network In- and Out-Degrees - Year: {year}, Month: {month:02d}", fontsize=16
    )

    # --- Plot 1: In-Degrees ---
    sc1 = ax1.scatter(
        x=ds["lon"],
        y=ds["lat"],
        c=in_degrees(ds),
        cmap="viridis",
        s=300,
        marker="s",
        vmin=0,
        vmax=100,
        **kwargs
    )

    ax1.set_title("In-Degrees (Connections To)")
    fig.colorbar(sc1, ax=ax1, label="Sum of Connections", shrink=0.78)

    # --- Plot 2: Out-Degrees ---
    sc2 = ax2.scatter(
        x=ds["lon"],
        y=ds["lat"],
        c=out_degrees(ds),
        cmap="viridis",
        s=300,
        vmin=0,
        vmax=350,
        marker="s",
        **kwargs
    )

    ax2.set_title("Out-Degrees (Connections From)")
    fig.colorbar(sc2, ax=ax2, label="Sum of Connections", shrink=0.78)

    out_dir = f"../results/plots/InOut/Scenario{scenario}/"
    out_path = out_dir + f"degrees_{scenario}_{year}_{month:02d}.png"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(out_path)
    plt.show()
    plt.close(fig)


def plot_yearly_degrees(scenario: int, year: int) -> None:
    """Plot yearly sum of in-degrees and out-degrees per node.

    Args:
        scenario (int): The SSP scenario identifier.
        year (int): The year to visualize.

    Returns:
        None. Displays the plot.

    """
    filepath = (
        "../data/water/scenario_ssp245_decade2030_month12.nc"  # Dummy file to get lon/lat
    )
    ds = xr.open_dataset(filepath)

    fig, ax1, ax2, kwargs = setup_double_amazon_map(fig_size=(22, 8))

    fig.suptitle(f"Network In- and Out-Degrees - Year: {year}", fontsize=16)

    # --- Plot 1: In-Degrees ---
    sc1 = ax1.scatter(
        x=ds["lon"],
        y=ds["lat"],
        c=yearly_in_degrees(scenario, year),
        cmap="viridis",
        s=300,
        vmin=0,
        vmax=870,
        marker="s",
        **kwargs
    )

    ax1.set_title("In-Degrees (Connections To)")
    fig.colorbar(sc1, ax=ax1, label="Sum of Connections", shrink=0.78)

    # --- Plot 2: Out-Degrees ---
    sc2 = ax2.scatter(
        x=ds["lon"],
        y=ds["lat"],
        c=yearly_out_degrees(scenario, year),
        cmap="viridis",
        s=300,
        vmin=0,
        vmax=1700,
        marker="s",
        **kwargs
    )

    ax2.set_title("Out-Degrees (Connections From)")
    fig.colorbar(sc2, ax=ax2, label="Sum of Connections", shrink=0.78)

    out_dir = f"../results/plots/YearInOut/Scenario{scenario}/"
    out_path = out_dir + f"yearly_degrees_{scenario}_{year}.png"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(out_path)
    plt.show()
    plt.close(fig)


def plot_diff_yearly_degrees(scenario: int, year: int) -> None:
    """Plot the difference (to 2030) in yearly sum of in-degrees and out-degrees per node.

    Args:
        scenario (int): The SSP scenario identifier.
        year (int): The year to visualize.

    Returns:
        None. Displays the plot.

    """
    filepath = "../data/water/scenario_ssp245_decade2030_month12.nc"
    ds = xr.open_dataset(filepath)

    fig, ax1, ax2, kwargs = setup_double_amazon_map(fig_size=(22, 8))

    fig.suptitle(
        f"Network Difference of In- and Out-Degrees - Year: {year}", fontsize=16
    )

    # --- Get Baseline Values for 2030 ---
    base_in = yearly_in_degrees(scenario, 2030)
    base_out = yearly_out_degrees(scenario, 2030)

    # --- Plot 1: In-Degrees ---
    sc1 = ax1.scatter(
        x=ds["lon"],
        y=ds["lat"],
        c=(yearly_in_degrees(scenario, year) - base_in),
        cmap="coolwarm_r",
        s=300,
        vmin=-190,
        vmax=190,
        marker="s",
        **kwargs
    )

    ax1.set_title("In-Degrees (Connections To)")
    fig.colorbar(sc1, ax=ax1, label="Sum of Connections", shrink=0.78)

    # --- Plot 2: Out-Degrees ---
    sc2 = ax2.scatter(
        x=ds["lon"],
        y=ds["lat"],
        c=(yearly_out_degrees(scenario, year) - base_out),
        cmap="coolwarm_r",
        s=300,
        vmin=-300,
        vmax=300,
        marker="s",
        **kwargs
    )

    ax2.set_title("Out-Degrees (Connections From)")
    fig.colorbar(sc2, ax=ax2, label="Sum of Connections", shrink=0.78)

    out_dir = f"../results/plots/YearInOut/Scenario{scenario}/"
    out_path = out_dir + f"yearly_diff_degrees_{scenario}_{year}.png"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(out_path)
    plt.show()
    plt.close(fig)


def plot_deforest_diff_yearly_degrees(scenario: int, year: int) -> None:
    """Plot the difference (to 2030) in yearly sum of in-degrees and out-degrees with deforestation.

    Args:
        scenario (int): The SSP scenario identifier.
        year (int): The year to visualize.

    Returns:
        None. Displays the plot.

    """
    filepath = "../data/water/scenario_ssp245_decade2030_month12.nc"
    ds = xr.open_dataset(filepath)

    fig, ax1, ax2, kwargs = setup_double_amazon_map(fig_size=(22, 8))

    fig.suptitle(
        f"Network Difference of In- and Out-Degrees with Deforestation - Year: {year}", fontsize=16
    )

    # --- Get Baseline Values for 2030 ---
    base_in = yearly_in_degrees(scenario, 2030)
    base_out = yearly_out_degrees(scenario, 2030)

    in_values = yearly_deforestation_in_degrees(scenario, year)
    out_values = yearly_deforestation_out_degrees(scenario, year)

    # --- Plot 1: In-Degrees ---
    sc1 = ax1.scatter(
        x=ds["lon"],
        y=ds["lat"],
        c=(in_values - base_in),
        cmap="coolwarm_r",
        s=300,
        vmin=-500,
        vmax=500,
        marker="s",
        **kwargs
    )

    ax1.set_title("In-Degrees (Connections To)")
    fig.colorbar(sc1, ax=ax1, label="Sum of Connections", shrink=0.78)

    # --- Plot 2: Out-Degrees ---
    sc2 = ax2.scatter(
        x=ds["lon"],
        y=ds["lat"],
        c=(out_values - base_out),
        cmap="coolwarm_r",
        s=300,
        vmin=-950,
        vmax=950,
        marker="s",
        **kwargs
    )

    ax2.set_title("Out-Degrees (Connections From)")
    fig.colorbar(sc2, ax=ax2, label="Sum of Connections", shrink=0.78)

    out_dir = f"../results/plots/Deforestation/Scenario{scenario}/"
    out_path = out_dir + f"yearly_deforest_diff_degrees_{scenario}_{year}.png"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(out_path)
    plt.show()
    plt.close(fig)


def plot_clustering(scenario: int, year: int, month: int) -> None:
    """Plot the clustering values per node.

    Args:
        scenario (int): The SSP scenario identifier.
        year (int): The year to visualize.
        month (int): The month number (1-12).

    Returns:
        None. Displays the plot.

    """
    filepath = f"../data/water/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    ds = xr.open_dataset(filepath)

    # Set self-connections to zero for accurate clustering calculation
    for i in range(ds.sizes["y"]):
        ds["network"].values[i][i] = 0

    fig, ax, kwargs = setup_amazon_map(fig_size=(12, 8))

    sc = ax.scatter(
        x=ds["lon"].values,
        y=ds["lat"].values,
        c=clustering_coefficients(ds),
        cmap="viridis",
        s=420,
        marker="s",
        vmin=0,
        vmax=1,
        **kwargs
    )

    plt.colorbar(sc, ax=ax, label="Clustering Coefficient")
    plt.title(
        f"Local Clustering Coefficients - Year: {year}, Month: {month:02d}", fontsize=16
    )

    out_dir = f"../results/plots/Clustering/Scenario{scenario}/"
    out_path = out_dir + f"clustering_{scenario}_{year}_{month:02d}.png"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(out_path)
    plt.show()
    plt.close(fig)


def plot_diff_yearly_clustering(scenario: int, year: int) -> None:
    """Plot the difference (to 2030) in yearly Clustering Coefficients per node.

    Args:
        scenario (int): The SSP scenario identifier.
        year (int): The year to visualize.

    Returns:
        None. Displays the plot.

    """
    filepath = "../data/water/scenario_ssp245_decade2030_month12.nc"
    ds = xr.open_dataset(filepath)

    fig, ax, kwargs = setup_amazon_map(fig_size=(12, 8))

    fig.suptitle(
        f"Network Difference of Clustering Coefficients - Year: {year}", fontsize=16
    )

    # --- Get Baseline Values for 2030 ---
    base_in = yearly_clustering(scenario, 2030)

    # --- Plot 1: In-Degrees ---
    sc = ax.scatter(
        x=ds["lon"],
        y=ds["lat"],
        c=(yearly_clustering(scenario, year) - base_in),
        cmap="coolwarm_r",
        s=320,
        vmin=-1,
        vmax=1,
        marker="s",
        **kwargs
    )

    fig.colorbar(sc, ax=ax, label="Difference in Clustering Coefficients", shrink=0.78)

    out_dir = f"../results/plots/Clustering/Scenario{scenario}/"
    out_path = out_dir + f"yearly_diff_clustering_{scenario}_{year}.png"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(out_path)
    # plt.show()
    plt.close(fig)


def plot_ffl(scenario: int, year: int, month: int) -> None:
    """Plot the amount of Feed Forward Loops per node.

    Args:
        scenario (int): The SSP scenario identifier.
        year (int): The year to visualize.
        month (int): The month number (1-12).

    Returns:
        None. Displays the plot.

    """
    filepath = f"../data/water/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    ds = xr.open_dataset(filepath)

    # Set self-connections to zero for accurate FFL calculation
    for i in range(ds.sizes["y"]):
        ds["network"].values[i][i] = 0

    fig, ax, kwargs = setup_amazon_map(fig_size=(12, 8))

    sc = ax.scatter(
        x=ds["lon"].values,
        y=ds["lat"].values,
        c=feed_forward_loop(ds),
        cmap="viridis",
        s=420,  # Point size
        marker="s",
        vmin=0,
        vmax=550,
        **kwargs
    )

    plt.colorbar(sc, ax=ax, label="# of Feed Forward Loops")
    plt.title(f"Feed Forward Loops - Year: {year}, Month: {month:02d}", fontsize=16)

    out_dir = f"../results/plots/FFL/Scenario{scenario}/"
    out_path = out_dir + f"ffl_{scenario}_{year}_{month:02d}.png"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(out_path)
    plt.show()
    plt.close(fig)


def plot_diff_yearly_ffl(scenario: int, year: int) -> None:
    """Plot the difference (to 2030) in yearly Feed Forward Loops per node.

    Args:
        scenario (int): The SSP scenario identifier.
        year (int): The year to visualize.

    Returns:
        None. Displays the plot.

    """
    filepath = "../data/water/scenario_ssp245_decade2030_month12.nc"
    ds = xr.open_dataset(filepath)

    fig, ax, kwargs = setup_amazon_map(fig_size=(12, 8))

    fig.suptitle(
        f"Network Difference of Feed Forward Loops - Year: {year}", fontsize=16
    )

    # --- Get Baseline Values for 2030 ---
    base_in = yearly_ffl(scenario, 2030)

    # --- Plot 1: In-Degrees ---
    sc = ax.scatter(
        x=ds["lon"],
        y=ds["lat"],
        c=(yearly_ffl(scenario, year) - base_in),
        cmap="coolwarm_r",
        s=320,
        vmin=-100,
        vmax=100,
        marker="s",
        **kwargs
    )

    fig.colorbar(sc, ax=ax, label="Difference in Feed Forward Loops", shrink=0.78)

    out_dir = f"../results/plots/FFL/Scenario{scenario}/"
    out_path = out_dir + f"yearly_diff_ffl_{scenario}_{year}.png"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(out_path)
    # plt.show()
    plt.close(fig)


def plot_deforestation(year: int) -> None:
    """Plot deforestation per year per node.

    Args:
        year (int): The year to visualize.

    Returns:
        None. Displays the plot.

    """
    data_plot = np.genfromtxt(f"../data/deforestation/deforestation_all_BaU_{year}.txt")

    lon = data_plot[:, 0]
    lat = data_plot[:, 1]
    deforest = 100*data_plot[:, 2]

    fig, ax, kwargs = setup_amazon_map(fig_size=(12, 8))

    colors = ["green", "black"]
    g_b_cmap = LinearSegmentedColormap.from_list("green_black", colors)

    sc = ax.scatter(
        x=lon,
        y=lat,
        c=deforest,
        cmap=g_b_cmap,
        s=420,
        marker="s",
        vmin=0,
        vmax=100,
        **kwargs
    )

    plt.colorbar(sc, ax=ax, label="Deforestation [%]")
    title = f"Deforestation - Year: {year}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    out_dir = "../results/plots/Deforestation/"
    out_path = out_dir + f"deforestation_{year}.png"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(out_path)
    # plt.show()
    plt.close(fig)


# --- Generalized Diff Plot Function ---
def _plot_scenario_diffs(metric_configs, title, output_filename, flag: str = "Forest") -> None:
    """Define a generalized function to plot yearly differences in % for all scenarios.

    Args:
        metric_configs (list of dict): Configuration for what to plot.
                Format: [{'prefix': 'indegrees', 'linestyle': '-', 'label': 'In-Degrees'}, ...]
        title (str): Plot title.
        output_filename (str): Output filename.

    Returns:
        None. Saves and displays the plot.

    """
    scenarios = ["245", "370", "585"]
    if flag == "Forest":
        years_to_scan = range(2030, 2100, 1)
    else:
        years_to_scan = range(2030, 2051, 1)

    colors = ["#ff9900", "#ff0000", "#8400ff"]

    results_path = Path("../results/cache")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for idx, scenario in enumerate(scenarios):
        scenario_color = colors[idx]

        # --- Loop over each metric configuration ---
        for config in metric_configs:
            prefix = config["prefix"]
            line_style = config.get("linestyle", "-")
            label_prefix = config.get("label", prefix.capitalize())

            # --- Load Year 2030 as Baseline for Comparison ---
            base_file = results_path / f"{prefix}_{scenario}_2030.nc"

            if not base_file.exists():
                print(
                    f"Skipping {prefix} for Scenario {scenario}: Baseline (2030) not found."
                )
                continue

            with xr.open_dataarray(base_file) as da_base:
                base_values = da_base.values

            # --- Collect Data for years ---
            plot_years = []
            diff_values = []

            for year in years_to_scan:
                file_path = results_path / f"{prefix}_{scenario}_{year}.nc"

                if file_path.exists():
                    with xr.open_dataarray(file_path) as da_curr:
                        curr_values = da_curr.values

                    # Calculate Difference in %
                    diff = np.nanmean(curr_values - base_values)
                    diff_perc = (diff / np.nanmean(base_values)) * 100

                    plot_years.append(year)
                    diff_values.append(diff_perc)

            # --- Plot the data for this metric and scenario ---
            if plot_years:
                ax.plot(
                    plot_years,
                    diff_values,
                    label=f"{label_prefix} Scenario {scenario}",
                    color=scenario_color,
                    linestyle=line_style,
                    alpha=0.9,
                )

            if flag == "Deforest":
                # --- Now plot deforestation data if available ---
                plot_years_def = []
                diff_values_def = []

                for year in years_to_scan:
                    file_path_deforest = results_path / f"Deforest_{prefix}_{scenario}_{year}.nc"

                    if file_path_deforest.exists():
                        with xr.open_dataarray(file_path_deforest) as da_curr_def:
                            curr_values_def = da_curr_def.values

                        # Calculate Difference in %
                        diff_def = np.nanmean(curr_values_def - base_values)
                        diff_perc_def = (diff_def / np.nanmean(base_values)) * 100

                        plot_years_def.append(year)
                        diff_values_def.append(diff_perc_def)

                # --- Plot the deforestation data ---
                if plot_years_def:
                    ax.plot(
                        plot_years_def,
                        diff_values_def,
                        label=f"{label_prefix} Deforestation Scenario {scenario}",
                        color=scenario_color,
                        linestyle=":",
                        alpha=0.9,
                    )

    plt.xticks(range(2030, 2051, 5))

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Average Difference per Node [%]", fontsize=12)

    ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.3)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="best", frameon=True, fontsize=10)
    plt.tight_layout()

    out_dir = os.path.dirname(output_filename)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    plt.show()
    plt.close(fig)


# --- Individual Difference Plot Functions ---
def plot_all_yearly_diff_degrees(flag: str = "Forest") -> None:
    """Plot course of yearly differences in in-degrees and out-degrees in %.

    Args:
        None.

    Returns:
        None. Displays the plot.

    """
    if flag == "Forest":
        configs = [
            {"prefix": "indegrees", "linestyle": "-", "label": "In-Degrees"},
            {"prefix": "outdegrees", "linestyle": "--", "label": "Out-Degrees"},
        ]
        _plot_scenario_diffs(
            metric_configs=configs,
            title="Network Degree Differences: (2030-2100)",
            output_filename="../results/YearlySummedDiffDegrees.png",
        )

    elif flag == "Deforest":
        configs = [
            {"prefix": "indegrees", "linestyle": "-", "label": "In-Degrees"},
        ]
        _plot_scenario_diffs(
            metric_configs=configs,
            title="Network Degree Differences with Deforestation: (2030-2050)",
            output_filename="../results/YearlySummedDiffDegrees_Deforest.png",
            flag="Deforest"
        )


def plot_all_yearly_diff_clustering(flag: str = "Forest") -> None:
    """Plot course of yearly differences in clustering in %.

    Args:
        None.

    Returns:
        None. Displays the plot.

    """
    if flag == "Forest":
        configs = [{"prefix": "clustering", "linestyle": "-", "label": "Clustering Diff"}]
        _plot_scenario_diffs(
            metric_configs=configs,
            title="Clustering Differences: (2030-2100)",
            output_filename="../results/YearlyDiffClustering.png",
        )

    elif flag == "Deforest":
        configs = [{"prefix": "clustering", "linestyle": "-", "label": "Clustering Diff"}]
        _plot_scenario_diffs(
            metric_configs=configs,
            title="Clustering Differences with Deforestation: (2030-2050)",
            output_filename="../results/YearlyDiffClustering_Deforest.png",
            flag="Deforest"
        )


def plot_all_yearly_diff_ffl(flag: str = "Forest") -> None:
    """Plot course of yearly differences in feed-forward loops in %.

    Args:
        None.

    Returns:
        None. Displays the plot.

    """
    if flag == "Forest":
        configs = [{"prefix": "ffl", "linestyle": "-", "label": "FFL Diff"}]
        _plot_scenario_diffs(
            metric_configs=configs,
            title="Feed Forward Loop Differences: (2030-2100)",
            output_filename="../results/YearlyDiffFFL.png",
        )

    elif flag == "Deforest":
        configs = [{"prefix": "ffl", "linestyle": "-", "label": "FFL Diff"}]
        _plot_scenario_diffs(
            metric_configs=configs,
            title="Feed Forward Loop Differences with Deforestation: (2030-2050)",
            output_filename="../results/YearlyDiffFFL_Deforest.png",
            flag="Deforest"
        )


def plot_dieoff(scenario: int, year: int, flag: str = "Forest") -> None:
    """Generate a plot showing dying cells (<1000mm) in black and others in White-Red.

    Args:
        scenario (int): The SSP scenario identifier.
        year (int): The specific year to plot (Added this arg as it was missing).
        flag (str): Toggle between "Forest" and "Deforest" logic.
    """
    filepath_base = "../data/water/scenario_ssp245_decade2030_month12.nc"
    ds = xr.open_dataset(filepath_base)

    for month in range(1, 13):
        filepath = f"../data/water/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
        ds_month = xr.open_dataset(filepath)

        if month == 1:
            prec_sum = copy.deepcopy(ds_month["prec"])
        else:
            prec_sum += ds_month["prec"]

    fig, ax, kwargs = setup_amazon_map(fig_size=(12, 8))

    import matplotlib.colors as mcolors
    colors = ["red", "white"]
    cmap = mcolors.LinearSegmentedColormap.from_list("white_red", colors)
    cmap.set_under('black')

    if flag == "Forest":
        title_text = f"Savanization of the Amazon Rainforest - Year: {year}"
        values = prec_sum
    elif flag == "Deforest":
        title_text = f"Savanization of the Amazon Rainforest with Deforestation - Year: {year}"
        values = prec_sum + yearly_deforestation_in_degrees(scenario, year) \
            - yearly_in_degrees(scenario, year)

    fig.suptitle(title_text, fontsize=16)

    colors = ['orange', 'forestgreen']
    bounds = [1000, 1500, 4000]

    cmap = mcolors.ListedColormap(colors)
    cmap.set_under('black')
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # --- 5. Plotting ---
    sc = ax.scatter(
        x=ds["lon"],
        y=ds["lat"],
        c=values,
        cmap=cmap,
        norm=norm,
        s=320,
        marker="s",
        **kwargs
    )

    cbar = fig.colorbar(sc, ax=ax, label="Incoming water (mm/year)", shrink=0.78, extend='min')

    cbar.set_ticks([1000, 1500])
    cbar.set_ticklabels(['1000 (Collapse)', '1500 (Risk)'])

    cbar.cmap.set_under('black')
    if flag == "Forest":
        out_dir = f"../results/plots/Dieoff/Scenario{scenario}/"
        out_path = out_dir + f"dieoff_{scenario}_{year}.png"

    elif flag == "Deforest":
        out_dir = f"../results/plots/Dieoff/Scenario{scenario}/"
        out_path = out_dir + f"deforest_dieoff_{scenario}_{year}.png"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(out_path)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    scenario = 585
    year = 2050
    month = 1

    # plot_precipitation(scenario=scenario, year=year, month=month)
    # plot_evaporation(scenario=scenario, year=year, month=month)
    # plot_network(scenario=scenario, year=year, month=month)
    # plot_degrees(scenario=scenario, year=year, month=month)
    # plot_clustering(scenario=scenario, year=year, month=month)
    # plot_yearly_degrees(scenario=scenario, year=year)
    # plot_deforest_diff_yearly_degrees(scenario=scenario, year=year)
    # plot_yearly_summed_diff_degrees(scenario=scenario)
    # plot_all_yearly_diff_degrees(flag="Forest")
    # plot_all_yearly_diff_clustering(flag="Forest")
    # plot_all_yearly_diff_ffl(flag="Forest")
    # plot_deforestation(year=2002)
    # plot_MAP_with_deforestation(scenario=scenario)

    scenario = 585
    for year in range(2030, 2100):
        plot_diff_yearly_ffl(scenario=scenario, year=year)
