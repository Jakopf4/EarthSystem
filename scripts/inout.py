"""Script to plot a network of data tracks based on connection strengths."""
import numpy as np

import xarray as xr


def in_degrees(ds: xr.Dataset) -> np.ndarray:
    """Calculate in-degrees of nodes in the network."""
    in_values = np.zeros(ds.sizes["y"])
    for i in range(ds.sizes["y"]):
        in_values[i] = sum(ds["network"].values[:, i])
    return in_values


def out_degrees(ds: xr.Dataset) -> np.ndarray:
    """Calculate out-degrees of nodes in the network."""
    out_values = np.zeros(ds.sizes["y"])
    for i in range(ds.sizes["y"]):
        out_values[i] = sum(ds["network"].values[i])
    return out_values
