"""Deforestation impact on water flow in network nodes."""

from io_utils import compute_and_cache

import numpy as np

import xarray as xr

# ==========================================
# 1. SETUP & ECHTE DATEN LADEN
# ==========================================
scenario = 245
year = 2030
month = 1


def deforestation_network(scenario: int, year: int, month: int):
    """Look at difference in water flow due to deforestation.

    Args:
        scenario (int): SSP scenario number.
        year (int): Year of the scenario.
        month (int): Month of the scenario.

    Returns:
        in_values (np.ndarray): Array of in-degrees per node.
        out_values (np.ndarray): Array of out-degrees per node.

    """
    nc_filepath = (
        f"../data/water/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    )
    txt_filepath = f"../data/deforestation/deforestation_all_BaU_{year}.txt"

    ds = xr.open_dataset(nc_filepath)
    data_deforestation = np.genfromtxt(txt_filepath)

    orig_matrix = ds["network"].values
    deforest_values = data_deforestation[:, 2]

    # Incoming water without deforestation values
    orig_col_sums = np.sum(orig_matrix, axis=0)

    # Create a copy to work on and keep original intact
    current_matrix = orig_matrix.copy()

    # Adjust all transportation values based on deforestation
    factor_matrix = 1 - deforest_values[:, np.newaxis]
    current_matrix = current_matrix * factor_matrix

    # Incoming water with deforestation values
    new_col_sums = np.sum(current_matrix, axis=0)

    ratios = np.zeros_like(orig_col_sums)

    # Avoid division by zero
    mask = orig_col_sums != 0
    # Calculate ratios of incoming water with and without deforestation
    ratios[mask] = new_col_sums[mask] / orig_col_sums[mask]

    # Reduce water flow based on lower incoming water
    deforested_matrix = current_matrix * ratios[:, np.newaxis]

    return deforested_matrix


def deforestation_in_degrees(scenario: int, year: int, month: int):
    """Calculate in-degrees of nodes in the network with deforestation.

    Args:
        scenario (int): SSP scenario number.
        year (int): Year of the scenario.
        month (int): Month of the scenario.

    Returns:
        in_values (np.ndarray): Array of in-degrees per node.

    """
    deforested_network = deforestation_network(scenario, year, month)

    in_values = np.nansum(deforested_network, axis=0)

    return in_values


def deforestation_out_degrees(scenario: int, year: int, month: int):
    """Calculate out-degrees of nodes in the network with deforestation.

    Args:
        scenario (int): SSP scenario number.
        year (int): Year of the scenario.
        month (int): Month of the scenario.

    Returns:
        out_values (np.ndarray): Array of out-degrees per node.

    """
    deforested_network = deforestation_network(scenario, year, month)

    out_values = np.nansum(deforested_network, axis=1)

    return out_values


def yearly_deforestation_in_degrees(scenario: int, year: int) -> np.ndarray:
    """Calculate yearly in-degrees of nodes in the network.

    Args:
        ds (xr.Dataset): Input dataset containing 'network' variable.

    Returns:
        np.ndarray: Array of yearly in-degrees per node.

    """
    return compute_and_cache(
        scenario,
        year,
        "Deforest_indegrees",
        deforestation_in_degrees,
        func_args={},
        do_average=False,
    )


def yearly_deforestation_out_degrees(scenario: int, year: int) -> np.ndarray:
    """Calculate yearly out-degrees of nodes in the network.

    Args:
        ds (xr.Dataset): Input dataset containing 'network' variable.

    Returns:
        np.ndarray: Array of yearly out-degrees per node.

    """
    return compute_and_cache(
        scenario,
        year,
        "Deforest_outdegrees",
        deforestation_out_degrees,
        func_args={},
        do_average=False,
    )


def store_data(scenario: int, year: int, month: int) -> None:
    """Store deforestation in/out degrees for all years.

    Args:
        scenario (int): Scenario identifier.
        years (list[int]): List of years to process.

    Returns:
        None, but saves data to cache.

    """
    filepath = (
        "../data/deforestation/"
        + f"deforestation_scenario_ssp{scenario}_decade_{year}_month{month:02d}.nc"
    )

    deforest_network = deforestation_network(scenario, year, month)

    dummy_file = "../data/water/scenario_ssp245_decade2030_month01.nc"
    dummy_ds = xr.open_dataset(dummy_file)
    lons = dummy_ds["lon"].values
    lats = dummy_ds["lat"].values

    ds = xr.Dataset(
        data_vars={
            "network": (["x", "y"], deforest_network),
            "lon": (["y"], lons),
            "lat": (["y"], lats),
        },
        attrs={"scenario": scenario, "year": year, "month": month},
    )

    ds.to_netcdf(filepath)


if __name__ == "__main__":
    scenario = 370
    years = list(range(2030, 2051, 1))

    for year in years:
        deforest_in_degrees = yearly_deforestation_in_degrees(scenario, year)
        deforest_out_degrees = yearly_deforestation_out_degrees(scenario, year)
