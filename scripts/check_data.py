"""File to understand the structure of the dataset using xarray."""
import xarray as xr

# Filepath to the dataset
filepath = "../data/scenario_ssp245_decade2030_month01.nc"

ds = xr.open_dataset(filepath)

# Print information about the dataset
print(ds)
