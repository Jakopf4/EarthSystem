"""Script to plot various network metrics and their differences over time."""

from pathlib import Path

from clustering import clustering_coefficients, feed_forward_loop

from inout import in_degrees, out_degrees, yearly_in_degrees, yearly_out_degrees

import matplotlib.pyplot as plt

import numpy as np

import xarray as xr


CONNECTION_THRESHOLD = 1.14  # Threshold for strong connections


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

    plt.figure(figsize=(12, 8))

    plt.scatter(
        x=ds["lon"],
        y=ds["lat"],
        c=ds["prec"],
        cmap="viridis",
        s=420,
        marker="s",
        vmin=0,
        vmax=450,
    )

    plt.colorbar(label="Precipitation")
    plt.grid(True)
    plt.title("Precipitation along Data Track")

    # plt.savefig(f"../results/plots/precipitation_{scenario}_{year}_{month:02d}.png")
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

    plt.figure(figsize=(12, 8))

    plt.scatter(
        x=ds["lon"],
        y=ds["lat"],
        c=ds["evap"],
        cmap="viridis",
        s=420,
        marker="s",
        vmin=0,
        vmax=130,
    )

    plt.colorbar(label="Evaporation")
    plt.grid(True)
    plt.title("Evaporation along Data Track")

    # plt.savefig(f"../results/plots/evaporation_{scenario}_{year}_{month:02d}.png")
    plt.show()
    plt.close()


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
    filepath = f"../data/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    ds = xr.open_dataset(filepath)

    lon_np = ds["lon"].values
    lat_np = ds["lat"].values
    network_np = ds["network"].values

    print(f"Loaded data, finding connections stronger than {CONNECTION_THRESHOLD}...")

    # --- Find all connections above the threshold ---
    i_indices, j_indices = np.where(network_np > CONNECTION_THRESHOLD)

    print(f"Found {len(i_indices)} connections matching the criteria.")

    fig, ax = plt.subplots(figsize=(12, 9))

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

    # plt.savefig(f"../results/plots/network_{scenario}_{year}_{month:02d}.png")
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
    filepath = f"../data/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    ds = xr.open_dataset(filepath)

    # Create subplots for in-degrees and out-degrees
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))

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
        vmin=0,
        vmax=100,
        marker="s",
    )

    ax1.set_title("In-Degrees (Connections To)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_xlim(-81, -48)
    ax1.set_ylim(-22, 7)
    ax1.set_aspect("equal")
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
    )

    ax2.set_title("Out-Degrees (Connections From)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_xlim(-81, -48)
    ax2.set_ylim(-22, 7)
    ax2.set_aspect("equal")
    fig.colorbar(sc2, ax=ax2, label="Sum of Connections", shrink=0.78)

    plt.savefig(
        f"../results/plots/InOut/Scenario{scenario}/degrees_{scenario}_{year}_{month:02d}.png"
    )
    # plt.show()
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
        "../data/scenario_ssp245_decade2030_month12.nc"  # Dummy file to get lon/lat
    )
    ds = xr.open_dataset(filepath)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))

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
    )

    ax1.set_title("In-Degrees (Connections To)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_xlim(-81, -48)
    ax1.set_ylim(-22, 7)
    ax1.set_aspect("equal")
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
    )

    ax2.set_title("Out-Degrees (Connections From)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_xlim(-81, -48)
    ax2.set_ylim(-22, 7)
    ax2.set_aspect("equal")
    fig.colorbar(sc2, ax=ax2, label="Sum of Connections", shrink=0.78)

    plt.savefig(
        f"../results/plots/YearInOut/Scenario{scenario}/yearly_degrees_{scenario}_{year}.png"
    )
    # plt.show()
    plt.close(fig)


def plot_diff_yearly_degrees(scenario: int, year: int) -> None:
    """Plot the difference (to 2030) in yearly sum of in-degrees and out-degrees per node.

    Args:
        scenario (int): The SSP scenario identifier.
        year (int): The year to visualize.

    Returns:
        None. Displays the plot.

    """
    filepath = "../data/scenario_ssp245_decade2030_month12.nc"
    ds = xr.open_dataset(filepath)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))

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
        cmap="coolwarm",
        s=300,
        vmin=-190,
        vmax=190,
        marker="s",
    )

    ax1.set_title("In-Degrees (Connections To)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_xlim(-81, -48)
    ax1.set_ylim(-22, 7)
    ax1.set_aspect("equal")
    fig.colorbar(sc1, ax=ax1, label="Sum of Connections", shrink=0.78)

    # --- Plot 2: Out-Degrees ---
    sc2 = ax2.scatter(
        x=ds["lon"],
        y=ds["lat"],
        c=(yearly_out_degrees(scenario, year) - base_out),
        cmap="coolwarm",
        s=300,
        vmin=-300,
        vmax=300,
        marker="s",
    )

    ax2.set_title("Out-Degrees (Connections From)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_xlim(-81, -48)
    ax2.set_ylim(-22, 7)
    ax2.set_aspect("equal")
    fig.colorbar(sc2, ax=ax2, label="Sum of Connections", shrink=0.78)

    plt.savefig(
        f"../results/plots/YearInOut/Scenario{scenario}/yearly_diff_degrees_{scenario}_{year}.png"
    )
    # plt.show()
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
    filepath = f"../data/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    ds = xr.open_dataset(filepath)

    # Set self-connections to zero for accurate clustering calculation
    for i in range(ds.sizes["y"]):
        ds["network"].values[i][i] = 0

    plt.figure(figsize=(12, 8))

    plt.scatter(
        x=ds["lon"].values,
        y=ds["lat"].values,
        c=clustering_coefficients(ds),
        cmap="viridis",
        s=420,
        marker="s",
        vmin=0,
        vmax=1,
    )

    plt.colorbar(label="Clustering Coefficient")
    plt.grid(True)
    plt.title(
        f"Local Clustering Coefficients - Year: {year}, Month: {month:02d}", fontsize=16
    )

    plt.savefig(
        f"../results/plots/Clustering/Scenario{scenario}"
        + f"/clustering_{scenario}_{year}_{month:02d}.png"
    )
    # plt.show()
    plt.close()


def plot_ffl(scenario: int, year: int, month: int) -> None:
    """Plot the amount of Feed Forward Loops per node.

    Args:
        scenario (int): The SSP scenario identifier.
        year (int): The year to visualize.
        month (int): The month number (1-12).

    Returns:
        None. Displays the plot.

    """
    filepath = f"../data/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    ds = xr.open_dataset(filepath)

    # Set self-connections to zero for accurate FFL calculation
    for i in range(ds.sizes["y"]):
        ds["network"].values[i][i] = 0

    plt.figure(figsize=(12, 8))

    plt.scatter(
        x=ds["lon"].values,
        y=ds["lat"].values,
        c=feed_forward_loop(ds),
        cmap="viridis",
        s=420,  # Point size
        marker="s",
        vmin=0,
        vmax=550,
    )

    plt.colorbar(label="# of Feed Forward Loops")
    plt.grid(True)
    plt.title(f"Feed Forward Loops - Year: {year}, Month: {month:02d}", fontsize=16)

    plt.savefig(
        f"../results/plots/FFL/Scenario{scenario}/ffl_{scenario}_{year}_{month:02d}.png"
    )
    # plt.show()
    plt.close()


# --- Generalized Diff Plot Function ---
def _plot_scenario_diffs(metric_configs, title, output_filename) -> None:
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
    years_to_scan = range(2030, 2100, 1)
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

    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    plt.show()
    plt.close(fig)


# --- Individual Difference Plot Functions ---
def plot_all_yearly_diff_degrees() -> None:
    """Plot course of yearly differences in in-degrees and out-degrees in %.

    Args:
        None.

    Returns:
        None. Displays the plot.

    """
    configs = [
        {"prefix": "indegrees", "linestyle": "-", "label": "In-Degrees"},
        {"prefix": "outdegrees", "linestyle": "--", "label": "Out-Degrees"},
    ]
    _plot_scenario_diffs(
        metric_configs=configs,
        title="Network Degree Differences: (2030-2100)",
        output_filename="../results/YearlySummedDiffDegrees_all_scenarios.png",
    )


def plot_all_yearly_diff_clustering() -> None:
    """Plot course of yearly differences in clustering in %.

    Args:
        None.

    Returns:
        None. Displays the plot.

    """
    configs = [{"prefix": "clustering", "linestyle": "-", "label": "Clustering Diff"}]
    _plot_scenario_diffs(
        metric_configs=configs,
        title="Clustering Differences: (2030-2100)",
        output_filename="../results/YearlyDiffClustering_all_scenarios.png",
    )


def plot_all_yearly_diff_ffl() -> None:
    """Plot course of yearly differences in feed-forward loops in %.

    Args:
        None.

    Returns:
        None. Displays the plot.

    """
    configs = [{"prefix": "ffl", "linestyle": "-", "label": "FFL Diff"}]
    _plot_scenario_diffs(
        metric_configs=configs,
        title="Feed Forward Loop Differences: (2030-2100)",
        output_filename="../results/YearlyDiffFFL_all_scenarios.png",
    )


if __name__ == "__main__":
    scenario = 245
    year = 2035
    month = 1

    # plot_precipitation(scenario=scenario, year=year, month=month)
    # plot_evaporation(scenario=scenario, year=year, month=month)
    # plot_network(scenario=scenario, year=year, month=month)
    # plot_degrees(scenario=scenario, year=year, month=month)
    # plot_clustering(scenario=scenario, year=year, month=month)
    # plot_yearly_degrees(scenario=scenario, year=year)
    # plot_diff_yearly_degrees(scenario=scenario, year=year)
    # plot_yearly_summed_diff_degrees(scenario=scenario)
    # plot_all_yearly_diff_degrees()
    # plot_all_yearly_diff_clustering()
    # plot_all_yearly_diff_ffl()
