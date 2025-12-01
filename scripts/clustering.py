"""Clustering and FFLs."""
import numpy as np

import xarray as xr

CONNECTION_THRESHOLD = 1.14  # Define a threshold for strong connections


def find_neighbourhood(ds: xr.Dataset, barrier: float) -> np.ndarray:
    """Find neighbourhood of all points in the dataset."""
    neighbourhood_of_point = np.zeros((ds.sizes["y"], ds.sizes["y"]))

    for i in range(ds.sizes["y"]):
        for j in range(ds.sizes["y"]):
            if ds["network"].values[i, j] > barrier:
                neighbourhood_of_point[i][j] = 1

    return neighbourhood_of_point


def actual_connections(ds: xr.Dataset, barrier: float) -> np.ndarray:
    """Calculate the actual connections inside a neighbourhood (Vectorized)."""
    A = find_neighbourhood(ds, barrier)

    # 2. Calculate common neighbors (Triangles)
    # Logic: A @ A.T counts paths of length 2.
    # Multiplying by A keeps only those that are also directly connected (closing the triangle).
    common_neighbors_count = A @ A.T
    valid_connections = A * common_neighbors_count

    # 3. Sum along rows to get triangles per node
    return valid_connections.sum(axis=1)


def clustering_coefficients(ds: xr.Dataset, barrier=CONNECTION_THRESHOLD) -> np.ndarray:
    """Calculate the local clustering coefficient from the dataset."""
    possible_connections = find_neighbourhood(ds, barrier).sum(axis=1)
    possible_connections = possible_connections * (possible_connections - 1)
    actual = actual_connections(ds, barrier)
    clustering_coefficients = np.zeros(ds.sizes["y"])

    for i in range(ds.sizes["y"]):
        if possible_connections[i] != 0:
            clustering_coefficients[i] = actual[i]/possible_connections[i]
        else:
            clustering_coefficients[i] = 0

    return clustering_coefficients


def feed_forward_loop(ds: xr.Dataset, barrier=CONNECTION_THRESHOLD) -> np.ndarray:
    """Calculate the feed forward loops for nodes from the dataset."""
    number_feed_forward = actual_connections(ds, barrier)
    return number_feed_forward
