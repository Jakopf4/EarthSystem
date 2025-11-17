"""Script to plot precipitation and evaporation data."""
import matplotlib.pyplot as plt

import xarray as xr

filepath = "../data/scenario_ssp245_decade2030_month01.nc"

try:
    with xr.open_dataset(filepath) as ds:
        print("--- Xarray Dataset Structure ---")
        print(ds)

        # --- 2. Create the Plot ---
        plt.figure(figsize=(12, 8))

        ds.plot.scatter(
            x="lon",
            y="lat",
            hue="prec",
            cmap="viridis",
            s=300,  # Point size
        )

        # --- 3. Customize ---
        plt.grid(True)
        plt.title("Precipitation along Data Track (using xarray)")
        plt.show()

except FileNotFoundError:
    print("--- ERROR ---")
    print(f"File not found at: {filepath}")
except KeyError as e:
    # This will catch if 'lon', 'lat', or 'prec' is missing
    print("--- ERROR: KeyError ---")
    print(f"Variable {e} not found in the file.")
    print("Check the 'Data variables' list in the printout above.")
