"""Script to plot and analyze in-degrees and out-degrees."""
import matplotlib.pyplot as plt

import numpy as np

import xarray as xr


def in_degrees(ds):
    """Convert radians to degrees."""
    in_values = np.zeros(ds.sizes["y"])
    for i in range(ds.sizes["y"]):
        in_values[i] = sum(ds["network"].values[:, i])
    return in_values


def out_degrees(ds):
    """Convert radians to degrees."""
    out_values = np.zeros(ds.sizes["y"])
    for i in range(ds.sizes["y"]):
        out_values[i] = sum(ds["network"].values[i])
    return out_values


def plot_degrees(scenario, year, month, flag="both"):
    """Plot in-degrees and out-degrees from the dataset."""
    filepath = f"../data/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    ds = xr.open_dataset(filepath)

    lon = ds["lon"]
    lat = ds["lat"]

    if flag == "in":
        fig, ax = plt.subplots(figsize=(12, 8))
        sc = ax.scatter(x=lon, y=lat, c=in_degrees(ds), cmap="viridis", s=300)

        fig.suptitle(
            f"Network In-Degrees - Year: {year}, Month: {month:02d}", fontsize=16
        )
        ax.grid(True)
        fig.colorbar(sc, ax=ax, label="Sum of Connections")

    elif flag == "out":
        fig, ax = plt.subplots(figsize=(12, 8))

        sc = ax.scatter(x=lon, y=lat, c=out_degrees(ds), cmap="viridis", s=300)

        fig.suptitle(
            f"Network Out-Degrees - Year: {year}, Month: {month:02d}", fontsize=16
        )
        ax.grid(True)
        fig.colorbar(sc, ax=ax, label="Sum of Connections")

    elif flag == "both":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))

        fig.suptitle(
            f"Network In- and Out-Degrees - Year: {year}, Month: {month:02d}",
            fontsize=16,
        )

        # --- Plot 1: In-Degrees (on ax1) ---
        sc1 = ax1.scatter(
            x=lon,
            y=lat,
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
        # ax1.grid(True)
        ax1.set_aspect("equal")
        fig.colorbar(sc1, ax=ax1, label="Sum of Connections",
                     shrink=0.78)

        # --- Plot 2: Out-Degrees (on ax2) ---
        sc2 = ax2.scatter(
            x=lon,
            y=lat,
            c=out_degrees(ds),
            cmap="viridis",
            s=300,
            vmin=0,
            vmax=350,
            marker="s",
        )
        ax2.set_title("Out-Degrees (Connections From)")
        ax2.set_xlabel("Longitude")
        # ax2.grid(True)
        ax2.set_aspect("equal")
        fig.colorbar(sc2, ax=ax2, label="Sum of Connections",
                     shrink=0.78)

    plt.savefig(f"../results/frame_{year}_{month:02d}.png")
    # plt.show()
    plt.close(fig)


def animate_degrees(flag="both"):
    """Animate in-degrees and out-degrees from the dataset."""
    # To be implemented
    pass


if __name__ == "__main__":
    month = 1
    year = 2030
    scenario = "245"

    flag = "both"  # Set this to "in", "out", or "both"
    plot_degrees(scenario, year, month, flag)
