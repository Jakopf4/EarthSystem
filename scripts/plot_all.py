"""Script to plot various network metrics and their differences over time."""

import os

import copy  # Important for modifying the colormap safely

from pathlib import Path

from clustering import clustering_coefficients, feed_forward_loop, yearly_clustering, yearly_ffl

from deforestation import yearly_deforestation_in_degrees, yearly_deforestation_out_degrees

from inout import in_degrees, out_degrees, yearly_in_degrees, yearly_out_degrees

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines

import numpy as np

import pandas as pd

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
    fig.colorbar(sc1, ax=ax1, label="Difference in Incoming Water [mm/year]", shrink=0.78)

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
    fig.colorbar(sc2, ax=ax2, label="Difference in Outgoing Water [mm/year]", shrink=0.78)

    out_dir = f"../results/plots/YearInOut/Scenario{scenario}/"
    out_path = out_dir + f"yearly_diff_degrees_{scenario}_{year}.png"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(out_path)
    # plt.show()
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
    bounds = [1000, 1800, 4000]

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

    cbar.set_ticks([1000, 1800])
    cbar.set_ticklabels(['1000 (Collapse)', '1800 (Risk)'])

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
    # plt.show()
    plt.close(fig)



### PAPER PLOTS ###

def plot_paper_fig1_overview(scenario: int, max_linewidth: float = 2.5) -> None:
    """
    Creates Figure 1 for the paper: A side-by-side comparison of the first year (2030) 
    and the last year (2099) showing total annual precipitation (background) 
    and averaged transport routes (arrows).
    Both color scales and arrow thickness are globally scaled for accurate comparison.
    """
    year_start = 2030
    year_end = 2099

    print(f"Erstelle zweiseitigen Overview-Plot (Fig 1) für SSP{scenario} ({year_start} vs {year_end})...")

    # Hilfsfunktion, um die akkumulierten Jahresdaten zu laden
    def load_annual_data(year):
        prec_sum = None
        network_sum = None
        lon = None
        lat = None
        
        for m in range(1, 13):
            filepath = f"../data/water/scenario_ssp{scenario}_decade{year}_month{m:02d}.nc"
            with xr.open_dataset(filepath) as ds:
                if m == 1:
                    prec_sum = ds["prec"].values.copy()
                    network_sum = ds["network"].values.copy()
                    lon = ds["lon"].values
                    lat = ds["lat"].values
                else:
                    prec_sum += ds["prec"].values
                    network_sum += ds["network"].values
                    
        # Gibt die Summe des Niederschlags und das gemittelte Netzwerk des Jahres zurück
        return prec_sum, network_sum / 12.0, lon, lat

    # 1. Daten für beide Jahre laden
    prec_start, network_start, lon_np, lat_np = load_annual_data(year_start)
    prec_end, network_end, _, _ = load_annual_data(year_end)

    # 2. Globale Max-Werte bestimmen (für konsistente Farbskalen UND Pfeildicken)
    global_vmax_prec = max(np.percentile(prec_start, 90), np.percentile(prec_end, 90))
    global_max_weight = max(np.max(network_start), np.max(network_end))

    # 3. Double-Map Setup aufrufen
    fig, ax1, ax2, kwargs = setup_double_amazon_map(fig_size=(22, 8))
    fig.suptitle(f"Amazon Moisture Recycling Network & Annual Precipitation - SSP {scenario}", 
                 fontsize=16, fontweight="bold", y=0.98)

    # Listen für die iterative Bearbeitung beider Subplots
    years = [year_start, year_end]
    axes = [ax1, ax2]
    prec_data = [prec_start, prec_end]
    network_data = [network_start, network_end]

    for ax, year, prec_vals, net_vals in zip(axes, years, prec_data, network_data):
        
        # 4. Hintergrund plotten (Gesamter Jahresniederschlag)
        sc = ax.scatter(
            x=lon_np,
            y=lat_np,
            c=prec_vals,
            cmap="Blues",
            s=300,
            marker="s",
            vmin=0,
            #vmax=global_vmax_prec
            vmax=2000, # Einheitliche für alle Szenarios
            alpha=0.8,
            **kwargs
        )
        
        # Colorbar für jedes Subplot
        cbar = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
        cbar.set_label("Annual Precipitation [mm/year]", fontsize=12)

        # 5. Verbindungen filtern
        i_indices, j_indices = np.where(net_vals > CONNECTION_THRESHOLD)

        print(f" -> Jahr {year}: {len(i_indices)} Pfeile über dem Threshold ({CONNECTION_THRESHOLD}) gefunden.")

        # 6. Pfeile in das jeweilige Subplot einzeichnen
        if len(i_indices) > 0:
            for i, j in zip(i_indices, j_indices):
                if i == j: # Selbstverbindungen ignorieren
                    continue

                start_point = (lon_np[j], lat_np[j])
                end_point = (lon_np[i], lat_np[i])
                weight = net_vals[i, j]

                # Skalierung von Dicke und Transparenz basierend auf dem GLOBALEN Maximum
                lw = max(0.2, (weight / global_max_weight) * max_linewidth)
                alpha_val = max(0.2, min(0.9, weight / global_max_weight * 1.5))

                ax.annotate(
                    "",
                    xy=end_point,
                    xytext=start_point,
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#D30808", # Dein gewünschtes Rot
                        linewidth=lw,
                        alpha=alpha_val
                    ),
                    **kwargs
                )
        
        # --- NEU: 6.5 Custom Legend für die Pfeile ---
        # Wir erstellen einen Dummy-Pfeil für die Legende
        arrow_legend = mlines.Line2D([], [], color='#D30808', marker='>',
                                     markersize=8, label='Atmospheric Water Transport', 
                                     linewidth=2)
        # Unten links ist bei Südamerika-Karten meistens der Pazifik (viel leerer Platz)
        ax.legend(handles=[arrow_legend], loc="lower left", fontsize=11, framealpha=0.9)
        # ---------------------------------------------

        # Subplot-Beschriftung
        ax.set_title(f"Year: {year} (Annual Total)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", alpha=0.4)

    # 7. Speichern & Schließen
    out_dir = "../results/plots/Paper_Plots/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"Fig1_Overview_Network_Comparison_{scenario}.png")
    
    plt.savefig(out_path)
    print(f"Vergleichsplot erfolgreich gespeichert unter: {out_path}")
    plt.close(fig)

def plot_paper_fig2_clustering_series() -> None:
    """
    Creates a side-by-side plot for the paper:
    Left: Clustering Difference [%] vs. Year
    Right: Clustering Difference [%] vs. Temperature Anomaly [°C]
    
    Reads temperature data from CSV files (Mean variable, already relative to 1850-1900).
    """
    scenarios = [245, 370, 585] 
    years_to_scan = range(2030, 2100, 1)
    colors = ["#ff9900", "#ff0000", "#8400ff"]

    print("\n--- STARTE DOPPEL-PLOT ANALYSE (CSV) ---")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    
    ax1.set_xlim(2030, 2100)

    for idx, scenario in enumerate(scenarios):
        scenario_color = colors[idx]
        print(f"\nVerarbeite Szenario SSP{scenario}:")

        # ========================================================
        # 1. TEMPERATURDATEN AUS .CSV DATEI LADEN
        # ========================================================
        # Dateinamen-Formatierung: Aus 245 wird "2_4_5"
        scen_str = str(scenario)
        scen_formatted = f"{scen_str[0]}_{scen_str[1]}_{scen_str[2]}"
        
        temp_file = f"../data/IPCC_Temperatures/tas_global_SSP{scen_formatted}.csv"
        temp_dict = {} 
        
        if os.path.exists(temp_file):
            try:
                # CSV mit Pandas einlesen
                df = pd.read_csv(temp_file)
                
                # Dictionary befüllen
                for _, row in df.iterrows():
                    # Das Jahr aus der CSV (z.B. 2015.0) in einen sauberen Integer umwandeln
                    y_int = int(float(row['Year']))
                    
                    if y_int in years_to_scan:
                        # Wir nutzen die Spalte "Mean". 
                        temp_dict[y_int] = float(row['Mean'])
                        
                print(f" -> {len(temp_dict)} Temperatur-Werte aus CSV extrahiert.")
            except Exception as e:
                print(f" -> FEHLER beim Lesen der CSV-Datei: {e}")
        else:
            print(f" -> FEHLER: Temperaturdatei nicht gefunden: {temp_file}")

        # ========================================================
        # 2. NETZWERK-DATEN ÜBER DEINE FUNKTION LADEN/BERECHNEN
        # ========================================================
        try:
            base_values = yearly_clustering(scenario, 2030)
        except Exception as e:
            print(f" -> ÜBERSPRINGE SZENARIO: Konnte Baseline (2030) nicht berechnen. Fehler: {e}")
            continue

        plot_years = []
        diff_values = []
        plot_temps = []

        for year in years_to_scan:
            try:
                curr_values = yearly_clustering(scenario, year)
                
                # Clustering Differenz in % berechnen
                diff = np.nanmean(curr_values - base_values)
                diff_perc = (diff / np.nanmean(base_values)) * 100

                temp_val = temp_dict.get(year, np.nan)

                plot_years.append(year)
                diff_values.append(diff_perc)
                plot_temps.append(temp_val)
            except Exception as e:
                pass

        print(f" -> {len(plot_years)} Jahre mit Netzwerk-Daten für den Plot verarbeitet.")

        # ========================================================
        # 3. ZEICHNEN DER PLOTS
        # ========================================================
        
        # --- Links Plotten (Gegen die Zeit) ---
        if len(plot_years) > 0:
            ax1.plot(
                plot_years, diff_values, label=f"SSP {scenario}",
                color=scenario_color, linestyle="-", linewidth=2.5, alpha=0.9
            )

        # --- Rechts Plotten (Gegen die Temperatur) ---
        valid_indices = [i for i, t in enumerate(plot_temps) if not np.isnan(t)]
        
        if len(valid_indices) > 0:
            valid_temps = np.array(plot_temps)[valid_indices]
            valid_diffs = np.array(diff_values)[valid_indices]
            
            # Da die Daten bereits gemittelt sind, plotten wir sie direkt dick und ohne Marker
            ax2.plot(
                valid_temps, valid_diffs,
                color=scenario_color, linestyle="-", linewidth=2.5, marker="", alpha=0.9
            )

    # ========================================================
    # 4. KOSMETIK UND SPEICHERN
    # ========================================================
    
    # --- Achse 1 (Zeit) ---
    ax1.set_title("Temporal Dynamics", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Clustering Coefficient Difference\n(relative to 2030) [%]", fontsize=12)
    ax1.set_xticks(range(2030, 2101, 10))
    ax1.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.3)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower left", frameon=True)

    # --- Achse 2 (Temperatur) ---
    ax2.set_title("Temperature Dependence", fontsize=14, fontweight="bold", pad=15)
    ax2.set_xlabel("Global Warming Level (°C relative to 1850-1900)", fontsize=12)
    ax2.set_xlim(left=1.4, right=5.2) 
    
    ax2.sharey(ax1)
    ax2.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.3)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Network Clustering Coefficient", fontsize=16, fontweight="bold", y=1.05)
    plt.tight_layout()

    out_dir = "../results/plots/Paper_Plots/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Fig2_Clustering.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot gespeichert unter: {out_path}")
    
    plt.close(fig)

def plot_paper_fig2_clustering_map(scenario: int) -> None:
    """
    Creates Figure 2 for the paper: A side-by-side spatial map comparison 
    of the Clustering Coefficient between 2030 and 2099 for a given scenario.
    Uses a global color scale across both maps for accurate scientific comparison.
    
    Args:
        scenario (int): The SSP scenario to plot (e.g., 245, 370, 585).
    """
    year_start = 2030
    year_end = 2099
    
    print(f"\nErstelle räumlichen Clustering-Vergleichsplot für SSP{scenario} ({year_start} vs {year_end})...")

    # ========================================================
    # 1. GEO-KOORDINATEN LADEN
    # ========================================================
    coord_file = f"../data/water/scenario_ssp{scenario}_decade{year_start}_month01.nc"
        
    try:
        with xr.open_dataset(coord_file) as ds_coord:
            lon_np = ds_coord["lon"].values
            lat_np = ds_coord["lat"].values
    except Exception as e:
        print(f" -> FEHLER beim Laden der Geokoordinaten aus {coord_file}: {e}")
        return

    # ========================================================
    # 2. CLUSTERING-DATEN LADEN/BERECHNEN
    # ========================================================
    try:
        data_start = yearly_clustering(scenario, year_start)
        data_end = yearly_clustering(scenario, year_end)
    except Exception as e:
        print(f" -> FEHLER beim Berechnen der Clustering-Werte: {e}")
        return

    # ========================================================
    # 3. FARBSKALA (GLOBAL) DEFINIEREN
    # ========================================================
    # Wir cappen beim 98. Perzentil, damit einzelne Extremwerte den Plot nicht blass machen
    global_vmax = max(np.percentile(data_start, 98), np.percentile(data_end, 98))
    global_vmin = min(np.nanmin(data_start), np.nanmin(data_end))
    
    # Optional: Wenn Clustering sowieso bei 0 startet, ist eine harte 0 oft besser fürs Paper
    global_vmin = max(0, global_vmin) 

    # ========================================================
    # 4. KARTEN SETUP & PLOTTEN
    # ========================================================
    fig, ax1, ax2, kwargs = setup_double_amazon_map(fig_size=(22, 8))
    fig.suptitle(f"Spatial Degradation: Local Clustering Coefficient (SSP {scenario})", 
                 fontsize=16, fontweight="bold", y=0.98)

    axes = [ax1, ax2]
    years = [year_start, year_end]
    datasets = [data_start, data_end]

    for ax, year, data_vals in zip(axes, years, datasets):
        
        # Räumliche Zellen plotten
        sc = ax.scatter(
            x=lon_np,
            y=lat_np,
            c=data_vals,
            cmap="Greens", # Wissenschaftlich hervorragend lesbar für Netzwerkdichten
            s=300,         # Kästchengröße, ggf. anpassen
            marker="s",
            #vmin=global_vmin,
            #vmax=global_vmax,
            vmin=0,
            vmax=1,
            alpha=0.9,
            **kwargs
        )
        
        # Colorbar für jedes Subplot
        cbar = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
        cbar.set_label("Local Clustering Coefficient", fontsize=12)

        # Beschriftung und Kosmetik
        ax.set_title(f"Year: {year}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", alpha=0.4)

    # ========================================================
    # 5. SPEICHERN
    # ========================================================
    out_dir = "../results/plots/Paper_Plots/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"Fig2_Clustering_Map_{scenario}.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Räumlicher Vergleichsplot erfolgreich gespeichert unter: {out_path}")
    
    plt.close(fig)

def plot_paper_fig3_ffl_series() -> None:
    """
    Creates Figure 3 for the paper:
    Left: Feed-Forward Loop Difference [%] vs. Year
    Right: Feed-Forward Loop Difference [%] vs. Temperature Anomaly [°C]
    
    Reads temperature data from CSV files (Mean variable, already relative to 1850-1900).
    """
    scenarios = [245, 370, 585] 
    years_to_scan = range(2030, 2100, 1)
    colors = ["#ff9900", "#ff0000", "#8400ff"]

    print("\n--- STARTE DOPPEL-PLOT ANALYSE: FEED-FORWARD LOOPS ---")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    
    ax1.set_xlim(2030, 2100)

    for idx, scenario in enumerate(scenarios):
        scenario_color = colors[idx]
        print(f"\nVerarbeite Szenario SSP{scenario}:")

        # ========================================================
        # 1. TEMPERATURDATEN AUS .CSV DATEI LADEN
        # ========================================================
        scen_str = str(scenario)
        scen_formatted = f"{scen_str[0]}_{scen_str[1]}_{scen_str[2]}"
        
        temp_file = f"../data/IPCC_Temperatures/tas_global_SSP{scen_formatted}.csv"
        temp_dict = {} 
        
        if os.path.exists(temp_file):
            try:
                df = pd.read_csv(temp_file)
                for _, row in df.iterrows():
                    y_int = int(float(row['Year']))
                    if y_int in years_to_scan:
                        temp_dict[y_int] = float(row['Mean'])
                        
                print(f" -> {len(temp_dict)} Temperatur-Werte aus CSV extrahiert.")
            except Exception as e:
                print(f" -> FEHLER beim Lesen der CSV-Datei: {e}")
        else:
            print(f" -> FEHLER: Temperaturdatei nicht gefunden: {temp_file}")

        # ========================================================
        # 2. NETZWERK-DATEN (FFL) BERECHNEN/LADEN
        # ========================================================
        try:
            # Nutzt nun yearly_ffl statt yearly_clustering
            base_values = yearly_ffl(scenario, 2030)
        except Exception as e:
            print(f" -> ÜBERSPRINGE SZENARIO: Konnte Baseline (2030) nicht berechnen. Fehler: {e}")
            continue

        plot_years = []
        diff_values = []
        plot_temps = []

        # Sicherheitscheck, falls die Base-Values 0 sein sollten (vermeidet Division by Zero)
        mean_base = np.nanmean(base_values)
        if mean_base == 0:
            print(f" -> WARNUNG: Mittlere FFL Baseline ist 0. Überspringe Prozentrechnung für SSP{scenario}.")
            continue

        for year in years_to_scan:
            try:
                curr_values = yearly_ffl(scenario, year)
                
                # FFL Differenz in % berechnen
                diff = np.nanmean(curr_values - base_values)
                diff_perc = (diff / mean_base) * 100

                temp_val = temp_dict.get(year, np.nan)

                plot_years.append(year)
                diff_values.append(diff_perc)
                plot_temps.append(temp_val)
            except Exception as e:
                pass

        print(f" -> {len(plot_years)} Jahre mit FFL-Daten für den Plot verarbeitet.")

        # ========================================================
        # 3. ZEICHNEN DER PLOTS
        # ========================================================
        
        # --- Links Plotten (Gegen die Zeit) ---
        if len(plot_years) > 0:
            ax1.plot(
                plot_years, diff_values, label=f"SSP {scenario}",
                color=scenario_color, linestyle="-", linewidth=2.5, alpha=0.9
            )

        # --- Rechts Plotten (Gegen die Temperatur) ---
        valid_indices = [i for i, t in enumerate(plot_temps) if not np.isnan(t)]
        
        if len(valid_indices) > 0:
            valid_temps = np.array(plot_temps)[valid_indices]
            valid_diffs = np.array(diff_values)[valid_indices]
            
            ax2.plot(
                valid_temps, valid_diffs,
                color=scenario_color, linestyle="-", linewidth=2.5, marker="", alpha=0.9
            )

    # ========================================================
    # 4. KOSMETIK UND SPEICHERN
    # ========================================================
    
    # --- Achse 1 (Zeit) ---
    ax1.set_title("Temporal Dynamics", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Year", fontsize=12)
    # Angepasstes Y-Label für FFL
    ax1.set_ylabel("Feed-Forward Loops Difference\n(relative to 2030) [%]", fontsize=12)
    ax1.set_xticks(range(2030, 2101, 10))
    ax1.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.3)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower left", frameon=True)

    # --- Achse 2 (Temperatur) ---
    ax2.set_title("Temperature Dependence", fontsize=14, fontweight="bold", pad=15)
    ax2.set_xlabel("Global Warming Level (°C relative to 1850-1900)", fontsize=12)
    ax2.set_xlim(left=1.4, right=5.2)
    
    ax2.sharey(ax1)
    ax2.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.3)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Angepasster Main Title
    fig.suptitle("Network Feed-Forward Loops", fontsize=16, fontweight="bold", y=1.05)
    plt.tight_layout()

    out_dir = "../results/plots/Paper_Plots/"
    os.makedirs(out_dir, exist_ok=True)
    # Angepasster Dateiname
    out_path = os.path.join(out_dir, "Fig2_FFL.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot gespeichert unter: {out_path}")
    
    plt.close(fig)

def plot_paper_fig3_ffl_map(scenario: int) -> None:
    """
    Creates Figure 3 for the paper: A side-by-side spatial map comparison 
    of the Feed-Forward Loops between 2030 and 2099 for a given scenario.
    Uses a global color scale across both maps for accurate scientific comparison.
    
    Args:
        scenario (int): The SSP scenario to plot (e.g., 245, 370, 585).
    """
    year_start = 2030
    year_end = 2099
    
    print(f"\nErstelle räumlichen FFL-Vergleichsplot für SSP{scenario} ({year_start} vs {year_end})...")

    # ========================================================
    # 1. GEO-KOORDINATEN LADEN
    # ========================================================
    coord_file = f"../data/water/scenario_ssp{scenario}_decade{year_start}_month01.nc"
        
    try:
        with xr.open_dataset(coord_file) as ds_coord:
            lon_np = ds_coord["lon"].values
            lat_np = ds_coord["lat"].values
    except Exception as e:
        print(f" -> FEHLER beim Laden der Geokoordinaten aus {coord_file}: {e}")
        return

    # ========================================================
    # 2. FFL-DATEN LADEN/BERECHNEN
    # ========================================================
    try:
        data_start = yearly_ffl(scenario, year_start)
        data_end = yearly_ffl(scenario, year_end)
    except Exception as e:
        print(f" -> FEHLER beim Berechnen der FFL-Werte: {e}")
        return

    # ========================================================
    # 3. FARBSKALA (GLOBAL) DEFINIEREN
    # ========================================================
    # Da FFLs absolute Zahlen (Counts) sind, MÜSSEN wir das Maximum dynamisch berechnen!
    # Wir cappen beim 98. Perzentil, damit einzelne Extremwerte den Plot nicht blass machen
    global_vmax = max(np.percentile(data_start, 98), np.percentile(data_end, 98))
    
    # Da die Anzahl der Loops nicht negativ sein kann, setzen wir vmin hart auf 0
    global_vmin = 0 

    # ========================================================
    # 4. KARTEN SETUP & PLOTTEN
    # ========================================================
    fig, ax1, ax2, kwargs = setup_double_amazon_map(fig_size=(22, 8))
    fig.suptitle(f"Spatial Degradation: Feed-Forward Loops (SSP {scenario})", 
                 fontsize=16, fontweight="bold", y=0.98)

    axes = [ax1, ax2]
    years = [year_start, year_end]
    datasets = [data_start, data_end]

    for ax, year, data_vals in zip(axes, years, datasets):
        
        # Räumliche Zellen plotten
        sc = ax.scatter(
            x=lon_np,
            y=lat_np,
            c=data_vals,
            cmap="Greens", # Du kannst hier auch z.B. "Oranges" oder "Purples" nehmen, um es vom Clustering abzuheben
            s=300,         # Kästchengröße
            marker="s",
            #vmin=global_vmin,
            #vmax=global_vmax, # Dynamisches Maximum für FFLs!
            vmin=0,
            vmax=120,
            alpha=0.9,
            **kwargs
        )
        
        # Colorbar für jedes Subplot
        cbar = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
        cbar.set_label("Number of Feed-Forward Loops", fontsize=12)

        # Beschriftung und Kosmetik
        ax.set_title(f"Year: {year}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", alpha=0.4)

    # ========================================================
    # 5. SPEICHERN
    # ========================================================
    out_dir = "../results/plots/Paper_Plots/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"Fig3_FFL_Map_{scenario}.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Räumlicher Vergleichsplot erfolgreich gespeichert unter: {out_path}")
    
    plt.close(fig)

def plot_MAP_comparison_secondary_veg() -> None:
    """
    Berechnet den Jahresniederschlag direkt aus den monatlichen NetCDF-Dateien 
    und speichert den Time-Series Plot als PNG-Datei für das Paper.
    Der Entwaldungseffekt wird dynamisch aus der Differenz der 'network'-Matrizen berechnet.
    """
    # =====================================================================
    # KONFIGURATION
    # =====================================================================
    scenarios = [245, 370, 585]
    years = np.arange(2030, 2051)
    
    base_dir = "../data/water/"
    sec_dir = "../data/water_secondary_vegetation/"
    out_dir = "../results/plots/Paper_Plots/"
    
    # Farben exakt wie in der Bildvorlage
    colors = {
        245: "#f5a623", # Orange/Gold
        370: "#d0021b", # Rot
        585: "#9013fe"  # Lila
    }
    
    print("\n--- Starte MAP Vergleichs-Plot (Speichermodus) ---")
    print("Berechne Daten und speichere Plot...")
    
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    
    # =====================================================================
    # DATENVERARBEITUNG & PLOTTING
    # =====================================================================
    for ssp in scenarios:
        print(f"Verarbeite SSP{ssp}...")
        
        map_baseline = []
        map_sec_veg = []
        
        for year in years:
            annual_prec_base = 0.0
            annual_prec_sec = 0.0
            valid_months = 0
            
            for month in range(1, 13):
                filename = f"scenario_ssp{ssp}_decade{year}_month{month:02d}.nc"
                filepath_base = os.path.join(base_dir, filename)
                filepath_sec = os.path.join(sec_dir, filename)
                
                if not os.path.exists(filepath_base) or not os.path.exists(filepath_sec):
                    continue
                    
                try:
                    with xr.open_dataset(filepath_base) as ds_base, xr.open_dataset(filepath_sec) as ds_sec:
                        # 1. Basis-Niederschlag (CMIP6 Forcing)
                        mean_prec_base = float(ds_base["prec"].mean(skipna=True))
                        
                        # 2. Feuchtigkeitsflüsse analysieren
                        rec_prec_base = ds_base["network"].sum(dim="x")
                        rec_prec_sec = ds_sec["network"].sum(dim="x")
                        
                        # 3. Differenz berechnen (Delta P)
                        delta_prec = float((rec_prec_sec - rec_prec_base).mean(skipna=True))
                        
                        # 4. Korrektur
                        mean_prec_sec = mean_prec_base + delta_prec
                        
                        annual_prec_base += mean_prec_base
                        annual_prec_sec += mean_prec_sec
                        valid_months += 1
                        
                except Exception as e:
                    print(f"  -> FEHLER bei {filename}: {e}")
                    continue
            
            print(f"  -> Jahr {year}  {mean_prec_base - mean_prec_sec:.2f} mm")

            if valid_months == 12:
                map_baseline.append(annual_prec_base)
                map_sec_veg.append(annual_prec_sec)
            else:
                map_baseline.append(np.nan)
                map_sec_veg.append(np.nan)
                
        # Plotting der Linien
        ax.plot(years, map_baseline, color=colors[ssp], linestyle="-", linewidth=2, label=f"No Deforestation - SSP{ssp}")
        ax.plot(years, map_sec_veg, color=colors[ssp], linestyle=":", linewidth=2.5, label=f"With Deforestation - SSP{ssp}")

    # =====================================================================
    # KOSMETIK & SPEICHERN
    # =====================================================================
    ax.axhline(y=1800, color="#b8b28f", linestyle="--", linewidth=2.5, label="Bistable Threshold (1800 mm)")
    ax.set_title("Mean Annual Precipitation Comparison with Deforestation", fontsize=15, fontweight="bold", pad=20)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Mean Annual Precipitation [mm]", fontsize=12)
    ax.set_xlim(2029, 2051)
    ax.set_ylim(1500, 2000) 
    ax.set_xticks(np.arange(2030, 2051, 5))
    ax.grid(True, linestyle="--", alpha=0.5, color="lightgray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    
    # Speichern
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "MAP_Comparison_SecVeg.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Plot erfolgreich gespeichert unter: {save_path}")



if __name__ == "__main__":
    scenario = 245
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
    # plot_paper_fig1_overview(scenario=scenario)
    # plot_paper_fig2_clustering_series()
    # plot_paper_fig2_clustering_map(scenario=scenario)
    # plot_paper_fig3_ffl_series()
    # plot_paper_fig3_ffl_map(scenario=scenario)
    plot_MAP_comparison_secondary_veg()
    

    #scenario = 585
    #for year in range(2030, 2051):
        #plot_dieoff(scenario=scenario, year=year, flag="Deforest")
