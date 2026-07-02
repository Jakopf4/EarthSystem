"""Script to plot various network metrics and their differences over time."""

import os

import copy

from pathlib import Path

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
        filepath = f"../2-pre-processing/pre-processed-data/water_Raw/scenario_ssp{scenario}_decade{year}_Raw.nc"
        with xr.open_dataset(filepath) as ds:
            prec_sum = ds["prec"].values.copy()
            network_sum = ds["network"].values.copy() / 12
            lon = ds["lon"].values
            lat = ds["lat"].values
        return prec_sum, network_sum, lon, lat

    # 1. Daten für beide Jahre laden
    prec_start, network_start, lon_np, lat_np = load_annual_data(year_start)
    prec_end, network_end, _, _ = load_annual_data(year_end)

    # 2. Globale Max-Werte bestimmen (für konsistente Farbskalen UND Pfeildicken)
    # global_vmax_prec = max(np.percentile(prec_start, 90), np.percentile(prec_end, 90))
    # global_max_weight = max(np.max(network_start), np.max(network_end))
    global_max_weight = 90 / 12

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
            vmax=2000,  # = global_vmax_prec # Einheitliche für alle Szenarios
            alpha=0.8,
            **kwargs
        )

        # Colorbar für jedes Subplot
        cbar = fig.colorbar(sc, ax=ax, shrink=0.95, pad=0.02)
        cbar.set_label("Annual Precipitation [mm/year]", fontsize=12)

        # 5. Verbindungen filtern
        i_indices, j_indices = np.where(net_vals > CONNECTION_THRESHOLD)

        print(f" -> Jahr {year}: {len(i_indices)} Pfeile über dem Threshold ({CONNECTION_THRESHOLD}) gefunden.")

        # 6. Pfeile in das jeweilige Subplot einzeichnen
        if len(i_indices) > 0:
            for i, j in zip(i_indices, j_indices):
                if i == j:  # Selbstverbindungen ignorieren
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
                        color="#D30808",  # Dein gewünschtes Rot
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
    out_dir = "Plots/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"Fig1_Overview_Network_Comparison_{scenario}.png")

    plt.savefig(out_path)
    print(f"Vergleichsplot erfolgreich gespeichert unter: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    # Beispielaufruf für SSP 370
    plot_paper_fig1_overview(scenario=370, max_linewidth=2.5)
