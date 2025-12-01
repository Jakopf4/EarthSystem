"""File to understand the structure of the dataset using xarray."""
import xarray as xr

# Filepath to the dataset
# Share Socio econimic pathways
filepath = "../data/scenario_ssp245_decade2030_month01.nc"

ds = xr.open_dataset(filepath)

print(ds)

print(min(ds["evap"].values))
