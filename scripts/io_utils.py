"""File to handle operations with caching for computed metrics."""
import os
from pathlib import Path

import numpy as np

import xarray as xr

# Directory Configuration
DATA_DIR = "../data"
OUTPUT_DIR = "../results/cache"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def get_filename(metric_name: str, scenario: int, year: int) -> str:
    """Get the filename for caching the computed metric.

    Args:
        metric_name (str): Name of the metric
        scenario (int): Scenario identifier
        year (int): Year of the data

    Returns:
        str: Full path to the cached file

    """
    return os.path.join(OUTPUT_DIR, f"{metric_name}_{scenario}_{year}.nc")


def compute_and_cache(
    scenario: int,
    year: int,
    metric_name: str,
    metric_func,
    func_args: dict = {},
    do_average: bool = True,
) -> np.ndarray:
    """
    Universal function to handle caching, computing, and saving.

    Args:
        scenario (int): Scenario identifier
        year (int): Year of the data
        metric_name (str): Name of the metric
        metric_func (callable): Function to compute the metric
        func_args (dict, optional): Additional arguments for the metric function. Defaults to {}.
        do_average (bool): Decide if yearly averaged or cumulative value is returned. Default True.

    Returns:
        np.ndarray: Computed metric values

    """
    filepath = get_filename(metric_name, scenario, year)

    if os.path.exists(filepath):
        with xr.open_dataarray(filepath) as da:
            return da.values

    # --- Only compute if missing ---
    print(f"Computing {metric_name} for {scenario} {year}...")
    cumulative_value = None

    for month in range(1, 13):
        file_p = os.path.join(
            DATA_DIR, f"scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
        )
        ds = xr.open_dataset(file_p)

        # Call the metric function with whatever arguments it needs (barrier, etc.)
        val = metric_func(ds, **func_args)

        if cumulative_value is None:
            cumulative_value = val
        else:
            cumulative_value += val

        ds.close()

    #  --- Average if decided ---
    final_value = cumulative_value / 12.0 if do_average else cumulative_value

    # --- Save the computed data ---
    da_out = xr.DataArray(
        final_value,
        dims=["node"],
        name=metric_name,
        attrs={"scenario": scenario, "year": year},
    )
    da_out.to_netcdf(filepath)

    return da_out.values
