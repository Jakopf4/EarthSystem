"""Script to plot a network of data tracks based on connection strengths."""
from io_utils import compute_and_cache

import numpy as np

import xarray as xr


def in_degrees(ds: xr.Dataset) -> np.ndarray:
    """Calculate in-degrees of nodes in the network.

    Args:
        ds (xr.Dataset): Input dataset containing 'network' variable.

    Returns:
        np.ndarray: Array of in-degrees per node.

    """
    in_values = np.zeros(ds.sizes["y"])
    for i in range(ds.sizes["y"]):
        in_values[i] = np.nansum(ds["network"].values[:, i])
    return in_values


def yearly_in_degrees(scenario: int, year: int) -> np.ndarray:
    """Calculate yearly in-degrees of nodes in the network.

    Args:
        ds (xr.Dataset): Input dataset containing 'network' variable.

    Returns:
        np.ndarray: Array of yearly in-degrees per node.

    """
    return compute_and_cache(
        scenario, year, "indegrees", in_degrees, func_args={}, do_average=False
    )


def year_average_diff_in_degrees(scenario: int, year: int) -> float:
    """Calculate the averaged difference in in-degrees of all nodes in one year.

    Args:
        ds (xr.Dataset): Input dataset containing 'network' variable.

    Returns:
        float: Averaged difference in in-degrees across all nodes in one year.

    """
    yearly_average_diff_in_value = yearly_in_degrees(
        scenario, year
    ) - yearly_in_degrees(scenario, 2030)
    return np.nansum(yearly_average_diff_in_value) / 416


def out_degrees(ds: xr.Dataset) -> np.ndarray:
    """Calculate out-degrees of nodes in the network.

    Args:
        ds (xr.Dataset): Input dataset containing 'network' variable.

    Returns:
        np.ndarray: Array of out-degrees per node.

    """
    out_values = np.zeros(ds.sizes["y"])
    for i in range(ds.sizes["y"]):
        out_values[i] = np.nansum(ds["network"].values[i])
    return out_values


def yearly_out_degrees(scenario: int, year: int) -> np.ndarray:
    """Calculate yearly out-degrees of nodes in the network.

    Args:
        ds (xr.Dataset): Input dataset containing 'network' variable.

    Returns:
        np.ndarray: Array of yearly out-degrees per node.

    """
    return compute_and_cache(
        scenario, year, "outdegrees", out_degrees, func_args={}, do_average=False
    )


def year_average_diff_out_degrees(scenario: int, year: int) -> float:
    """Calculate the averaged difference in out-degrees of all nodes in one year.

    Args:
        ds (xr.Dataset): Input dataset containing 'network' variable.

    Returns:
        float: Averaged difference in out-degrees across all nodes in one year.

    """
    yearly_average_diff_out_value = yearly_out_degrees(
        scenario, year
    ) - yearly_out_degrees(scenario, 2030)
    return np.nansum(yearly_average_diff_out_value) / 416


if __name__ == "__main__":
    scenario = 585
    years = list(range(2030, 2100, 20))

    for year in years:
        yearly_in_degrees(scenario, year)
        yearly_out_degrees(scenario, year)
