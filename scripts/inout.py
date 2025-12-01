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
    for month in range(1,13):
        filepath =[f"../data/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"]
        ds = [xr.open_dataset(filepath)]
        yearly_in_value[month] += in_degrees(ds[month])
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
    for month in range(1,13):
        filepath =[f"../data/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"]
        ds = [xr.open_dataset(filepath)]
        yearly_out_value[month] += out_degrees(ds[month])
    return yearly_out_value 
