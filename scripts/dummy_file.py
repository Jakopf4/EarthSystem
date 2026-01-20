"""File to create a dummy netCDF file for testing purposes."""

import os

import numpy as np

import xarray as xr

# 1. Define directory and filename
output_dir = "../data/"
filename = "dummy_scenario_ssp1_decade2030_month01.nc"

# Combine them into a full path (handles slash differences automatically)
full_path = os.path.join(output_dir, filename)

# IMPORTANT: Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 2. Define the data manually (Same as before)
network_data = np.array([[0, 2, 2, 2], [2, 0, 2, 0], [0, 0, 0, 0], [2, 0, 0, 0]], dtype='float64')
lon_data = np.array([0, 1, 0, 1], dtype='float64')
lat_data = np.array([0, 0, 1, 1], dtype='float64')
prec_data = np.array([1.0, 1.0, 1.0, 1.0], dtype='float64')
evap_data = np.array([1.0, 1.0, 1.0, 1.0], dtype='float64')

# 3. Create the xarray Dataset
ds = xr.Dataset(
    data_vars={
        "network": (("x", "y"), network_data),
        "lon":     (("y"), lon_data),
        "lat":     (("y"), lat_data),
        "prec":    (("y"), prec_data),
        "evap":    (("y"), evap_data),
    }
)

# 4. Save using the full path
ds.to_netcdf(full_path)

print(f"Successfully created file at: {full_path}")
