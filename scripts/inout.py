"""Script to plot a network of data tracks based on connection strengths."""
import numpy as np

import xarray as xr


def in_degrees(ds: xr.Dataset) -> np.ndarray:
    """Calculate in-degrees of nodes in the network."""
    in_values = np.zeros(ds.sizes["y"])
    for i in range(ds.sizes["y"]):
        in_values[i] = sum(ds["network"].values[:, i])
    return in_values


def yearly_in_degrees(scenario: str, year: int) -> np.ndarray:
    """Calculate yearly in-degrees of nodes in the network."""
    yearly_in_value = np.zeros(416)
    filepath = []
    ds = []
    for month in range(1, 13):
        filepath.append(f"../data/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc")
        ds.append(xr.open_dataset(filepath[month-1]))
        yearly_in_value += in_degrees(ds[month-1])
    return yearly_in_value


def out_degrees(ds: xr.Dataset) -> np.ndarray:
    """Calculate out-degrees of nodes in the network."""
    out_values = np.zeros(ds.sizes["y"])
    for i in range(ds.sizes["y"]):
        out_values[i] = sum(ds["network"].values[i])
    return out_values


def yearly_out_degrees(scenario: str, year: int) -> np.ndarray:
    """Calculate yearly out-degrees of nodes in the network."""
    yearly_out_value = np.zeros(416)
    filepath = []
    ds = []

    for month in range(1, 13):
        filepath.append(f"../data/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc")
        ds.append(xr.open_dataset(filepath[month-1]))
        yearly_out_value += out_degrees(ds[month-1])
    return yearly_out_value
