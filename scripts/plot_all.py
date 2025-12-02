"""Script to plot a network of data tracks based on connection strengths."""
from clustering import clustering_coefficients, feed_forward_loop
from inout import in_degrees, out_degrees, yearly_in_degrees, yearly_out_degrees

import matplotlib.pyplot as plt

import numpy as np

import xarray as xr


CONNECTION_THRESHOLD = 1.14  # Define a threshold for strong connections


def plot_precipitation(scenario: str, year: int, month: int) -> None:
    """Plot precipitation data from the dataset."""
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

    # --- 3. Customize ---
    plt.colorbar(label="Precipitation")
    plt.grid(True)
    plt.title("Precipitation along Data Track")

    plt.show()
    # plt.savefig(f"../results/plots/precipitation_{scenario}_{year}_{month:02d}.png")
    plt.close()


def plot_evaporation(scenario: str, year: int, month: int) -> None:
    """Plot precipitation data from the dataset."""
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

    # --- 3. Customize ---
    plt.colorbar(label="Evaporation")
    plt.grid(True)
    plt.title("Evaporation along Data Track")

    plt.show()
    # plt.savefig(f"../results/plots/evaporation_{scenario}_{year}_{month:02d}.png")
    plt.close()


def plot_network(
    scenario: str, year: int, month: int, CONNECTION_THRESHOLD=CONNECTION_THRESHOLD
) -> None:
    """Plot a network of data tracks based on connection strengths."""
    filepath = f"../data/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    ds = xr.open_dataset(filepath)

    lon_np = ds["lon"].values
    lat_np = ds["lat"].values
    network_np = ds["network"].values

    print(f"Loaded data, finding connections stronger than {CONNECTION_THRESHOLD}...")

    # --- 1. Find all connections *above the threshold* ---
    i_indices, j_indices = np.where(network_np > CONNECTION_THRESHOLD)

    # We no longer need the 'segments' list
    print(f"Found {len(i_indices)} connections matching the criteria.")

    # --- 2. Create the Plot ---
    fig, ax = plt.subplots(figsize=(12, 9))

    # --- 3. Plot the Connections (Arrows) ---

    if len(i_indices) > 0:
        for i, j in zip(i_indices, j_indices):
            # A connection from i to j
            start_point = (lon_np[j], lat_np[j])
            end_point = (lon_np[i], lat_np[i])

            # Skip connections from a node to itself
            if i == j:
                continue

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

    # --- 4. Plot the Nodes (Points) ---
    ax.scatter(lon_np, lat_np, c="red", s=4, zorder=10)

    # --- 5. Final Plot Customization ---
    ax.set_title(f"Moisture flows (Connections > {CONNECTION_THRESHOLD})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)

    plt.show()
    # plt.savefig(f"../results/plots/network_{scenario}_{year}_{month:02d}.png")
    plt.close(fig)


def plot_degrees(scenario: str, year: int, month: int) -> None:
    """
    Plot in-degrees and out-degrees from the dataset.

    Input:
        Scenario (str): Number of the scenario
        Year (int): Year number of dataset
        Month (int): Month to look at

    Output:
        None, but plots saved to directory
    """
    filepath = f"../data/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    ds = xr.open_dataset(filepath)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))

    fig.suptitle(
        f"Network In- and Out-Degrees - Year: {year}, Month: {month:02d}", fontsize=16
    )

    # --- Plot 1: In-Degrees (on ax1) ---
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
    ax2.set_xlim(-81, -48)
    ax2.set_ylim(-22, 7)
    # ax1.grid(True)
    ax1.set_aspect("equal")
    fig.colorbar(sc1, ax=ax1, label="Sum of Connections", shrink=0.78)

    # --- Plot 2: Out-Degrees (on ax2) ---
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
    # ax2.grid(True)
    ax2.set_aspect("equal")
    fig.colorbar(sc2, ax=ax2, label="Sum of Connections", shrink=0.78)

    # plt.show()
    plt.savefig(
        f"../results/plots/InOut/Scenario{scenario}/degrees_{scenario}_{year}_{month:02d}.png"
    )
    plt.close(fig)


def plot_yearly_degrees(scenario: str, year: int) -> None:
    """
    Plot in-degrees and out-degrees from the dataset for an entire year.

    Input:
        Scenario (str): Number of the scenario
        Year (int): Year number of dataset

    Output:
        None, but plots saved to directory
    """
    filepath = "../data/scenario_ssp245_decade2030_month12.nc"
    ds = xr.open_dataset(filepath)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))

    fig.suptitle(
        f"Network In- and Out-Degrees - Year: {year}", fontsize=16
    )

    # --- Plot 1: In-Degrees (on ax1) ---
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
    ax2.set_xlim(-81, -48)
    ax2.set_ylim(-22, 7)
    # ax1.grid(True)
    ax1.set_aspect("equal")
    fig.colorbar(sc1, ax=ax1, label="Sum of Connections", shrink=0.78)

    # --- Plot 2: Out-Degrees (on ax2) ---
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
    # ax2.grid(True)
    ax2.set_aspect("equal")
    fig.colorbar(sc2, ax=ax2, label="Sum of Connections", shrink=0.78)

    plt.show()
    plt.savefig(
         f"../results/plots/YearInOut/Scenario{scenario}/yearly_degrees_{scenario}_{year}.png"
     )
    plt.close(fig)


def plot_diff_yearly_degrees(scenario: str, year: int) -> None:
    """
    Plot the difference of in-degrees and out-degrees from the dataset for an entire year.

    Input:
        Scenario (str): Number of the scenario
        Year (int): Year number of dataset

    Output:
        None, but plots saved to directory
    """
    filepath = "../data/scenario_ssp245_decade2030_month12.nc"
    ds = xr.open_dataset(filepath)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))

    fig.suptitle(
        f"Network Difference of In- and Out-Degrees - Year: {year}", fontsize=16
    )
    initial_yearly_in = yearly_in_degrees(scenario, 2030)
    initial_yearly_out = yearly_out_degrees(scenario, 2030)

    # --- Plot 1: In-Degrees (on ax1) ---
    sc1 = ax1.scatter(
        x=ds["lon"],
        y=ds["lat"],
        c=(yearly_in_degrees(scenario, year) - initial_yearly_in),
        cmap="coolwarm",
        s=300,
        vmin=-190,
        vmax=190,
        marker="s",
    )

    ax1.set_title("In-Degrees (Connections To)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax2.set_xlim(-81, -48)
    ax2.set_ylim(-22, 7)
    # ax1.grid(True)
    ax1.set_aspect("equal")
    fig.colorbar(sc1, ax=ax1, label="Sum of Connections", shrink=0.78)

    # --- Plot 2: Out-Degrees (on ax2) ---
    sc2 = ax2.scatter(
        x=ds["lon"],
        y=ds["lat"],
        c=(yearly_out_degrees(scenario, year) - initial_yearly_out),
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
    # ax2.grid(True)
    ax2.set_aspect("equal")
    fig.colorbar(sc2, ax=ax2, label="Sum of Connections", shrink=0.78)

    plt.show()
    plt.savefig(
         f"../results/plots/YearInOut/Scenario{scenario}/yearly_diff_degrees_{scenario}_{year}.png"
    )
    plt.close(fig)


def plot_clustering(scenario: str, year: int, month: int) -> None:
    """Plot local clustering coefficient from the dataset."""
    filepath = f"../data/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    ds = xr.open_dataset(filepath)

    for i in range(ds.sizes["y"]):
        ds["network"].values[i][i] = 0

    plt.figure(figsize=(12, 8))

    plt.suptitle(
        f"Local Clustering Coefficients - Year: {year}, Month: {month:02d}", fontsize=16
    )

    sc = plt.scatter(
        x=ds["lon"].values,
        y=ds["lat"].values,
        c=clustering_coefficients(ds),
        cmap="viridis",
        s=420,  # Point size
        marker="s",
        vmax=1,
        vmin=0,
    )

    plt.colorbar(sc, label="Clustering Coefficient")

    plt.grid(True)

    # plt.show()
    plt.savefig(f"../results/plots/Clustering/Scenario{scenario}/clustering_\
                {scenario}_{year}_{month:02d}.png")
    plt.close()


def plot_ffl(scenario: str, year: int, month: int) -> None:
    """Plot feed forward loops from the dataset."""
    filepath = f"../data/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    ds = xr.open_dataset(filepath)

    for i in range(ds.sizes["y"]):
        ds["network"].values[i][i] = 0

    plt.figure(figsize=(12, 8))

    plt.suptitle(f"Feed Forward Loops - Year: {year}, Month: {month:02d}", fontsize=16)

    sc = plt.scatter(
        x=ds["lon"].values,
        y=ds["lat"].values,
        c=feed_forward_loop(ds),
        cmap="viridis",
        s=420,  # Point size
        marker="s",
        vmin=0,
        vmax=550
    )

    plt.colorbar(sc, label="# Feed Forward Loops")

    plt.grid(True)

    # plt.show()
    plt.savefig(
        f"../results/plots/FFL/Scenario{scenario}/ffl_{scenario}_{year}_{month:02d}.png"
    )
    plt.close()


if __name__ == "__main__":
    scenario = "370"
    year = 2095
    month = 1

    # plot_precipitation(scenario=scenario, year=year, month=month)
    # plot_evaporation(scenario=scenario, year=year, month=month)
    # plot_network(scenario=scenario, year=year, month=month)
    # plot_degrees(scenario=scenario, year=year, month=month)
    # plot_clustering(scenario=scenario, year=year, month=month)
    plot_yearly_degrees(scenario=scenario, year=year)
    plot_diff_yearly_degrees(scenario=scenario, year=year)
