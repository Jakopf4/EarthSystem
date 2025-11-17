"""File to understand the structure of the dataset using xarray."""
import xarray as xr

# Filepath to a dataset
filepath = "../data/scenario_ssp245_decade2030_month01.nc"

ds = xr.open_dataset(filepath)

print(ds)
