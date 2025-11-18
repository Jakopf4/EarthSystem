import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

scenario = "245"
year = 2030
month = 12

filepath = f"../data/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
ds = xr.open_dataset(filepath)

for i in range(ds.sizes["y"]):
    ds["network"].values[i][i] = 0

mean_connection_strength = np.mean(ds["network"].values)


# Defining neighbourhoods based on mean connection strength

neighbourhood_of_point = np.zeros((ds.sizes["y"],ds.sizes["y"]))

for i in range(ds.sizes["y"]):
    for j in range(ds.sizes["y"]):
        if ds["network"].values[i, j] > mean_connection_strength:
            neighbourhood_of_point[i][j] = 1

# Possible number of connections per neighbourhood

possible_connections = np.zeros(ds.sizes["y"])

for i in range(ds.sizes["y"]):
    for j in range(ds.sizes["y"]):
        if neighbourhood_of_point[i][j] != 0:
            possible_connections[i] += 1

for i in range(ds.sizes["y"]):
    possible_connections[i] = possible_connections[i]*(possible_connections[i] - 1)

# Actual number of connections per neighbourhood

actual_connections = np.zeros(ds.sizes["y"])

for i in range(ds.sizes["y"]):
    for j in range(ds.sizes["y"]):
        if neighbourhood_of_point[i][j] != 0:
            for k in range(ds.sizes["y"]):
                if neighbourhood_of_point[i][k] !=0 and neighbourhood_of_point[j][k] != 0: 
                    actual_connections[i] += 1

# Clustering coefficient per neighbourhood

clustering_coefficients = np.zeros(ds.sizes["y"])

for i in range(ds.sizes["y"]):
    if possible_connections[i] != 0:
        clustering_coefficients[i] = actual_connections[i]/possible_connections[i]
    else:
        clustering_coefficients[i] = 0

# Plotting the clustering coefficients

plt.figure(figsize=(12, 8))

sc = plt.scatter(
    x=ds["lon"].values,
    y=ds["lat"].values,
    c=clustering_coefficients,
    cmap="viridis",
    s=450,  # Point size
    marker='s',
    vmax=1,
    vmin=0
)

plt.colorbar(sc, label="Clustering Coefficient")

plt.grid(True)
plt.title("Clustering Coefficients of Network Neighbourhoods", fontsize=16)
plt.show()


