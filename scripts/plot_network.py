"""Script to plot a network of data tracks based on connection strengths."""
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import numpy as np

import xarray as xr

filepath = '../data/scenario_ssp245_decade2030_month01.nc'
CONNECTION_THRESHOLD = 1.2  # Define a threshold for strong connections

try:
    with xr.open_dataset(filepath) as ds:
        # Load the required data into numpy arrays
        lon_np = ds['lon'].values
        lat_np = ds['lat'].values
        network_np = ds['network'].values

except FileNotFoundError:
    print("--- ERROR ---")
    print(f"File not found at: {filepath}")
    sys.exit()
except KeyError as e:
    print("--- ERROR: KeyError ---")
    print(f"Variable {e} not found in the file.")
    sys.exit()

print(f"Loaded data, finding connections stronger than {CONNECTION_THRESHOLD}...")

# --- 1. Find all connections *above the threshold* ---
i_indices, j_indices = np.where(network_np > CONNECTION_THRESHOLD)

segments = []

for i, j in zip(i_indices, j_indices):
    # Only plot pairs where j > i to avoid drawing every
    # line twice and to skip self-connections (i == j)
    # Get the (lon, lat) coordinates for the start and end points

    # ToDo: Upper triangle only and lower triangle
    point1 = [lon_np[i], lat_np[i]]
    point2 = [lon_np[j], lat_np[j]]
    if i >= j:
        segments.append([point1, point2])
    if i < j:
        segments.append([point2, point1])

print(f"Found {len(segments)} connections matching the criteria.")

# --- 2. Create the Plot ---

fig, ax = plt.subplots(figsize=(12, 9))

# --- 3. Plot the Connections (Lines) ---

if len(segments) > 0:
    # Create the LineCollection
    lc = LineCollection(segments,
                        color='blue',      # All lines are blue
                        linewidth=0.5)   # Set line width

    ax.add_collection(lc)      # Add the lines to the plot
else:
    print("Warning: No connections found above the threshold. Only points will be plotted.")

# --- 4. Plot the Nodes (Points) ---

# Plot the nodes (points) on top of the lines so they are visible
ax.scatter(lon_np, lat_np,
           c='red',        # Make nodes a distinct color
           s=4,           # Set node size
           zorder=10)      # zorder=10 ensures they are plotted on top

# --- 5. Final Plot Customization ---

ax.set_title(f"Data Track Network (Connections > {CONNECTION_THRESHOLD})")
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.grid(True)

plt.show()
