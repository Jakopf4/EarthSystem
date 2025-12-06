"""Script to calculate clustering coefficients and feed forward loops from network data."""

from pathlib import Path

from io_utils import compute_and_cache

import numpy as np

import xarray as xr

# --- Configuration ---
CONNECTION_THRESHOLD = 1.14
DATA_DIR = "../data"
OUTPUT_DIR = "../results/cache"

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# --- Core Calculations ---
def find_neighbourhood(ds: xr.Dataset, barrier: float = CONNECTION_THRESHOLD) -> np.ndarray:
    """Find neighbourhood of all points.

    Args:
        ds (xr.Dataset): Input dataset containing 'network' variable.
        barrier (float): Threshold to determine connections.

    Returns:
        np.ndarray: Adjacency matrix representing connections.

    """
    adj_matrix = (ds["network"].values > barrier).astype(int)
    return adj_matrix


def actual_connections(ds: xr.Dataset, barrier: float = CONNECTION_THRESHOLD) -> np.ndarray:
    """Calculate the actual connections inside a directed neighbourhood.

    Args:
        ds (xr.Dataset): Input dataset containing 'network' variable.
        barrier (float): Threshold to determine connections.

    Returns:
        np.ndarray: Array of actual connections per node.

    """
    A = find_neighbourhood(ds, barrier)

    # Calculate common neighbors
    common_neighbors_count = A @ A.T
    valid_connections = A * common_neighbors_count

    # Sum along rows to get triangles per node
    return valid_connections.sum(axis=1)


def clustering_coefficients(ds: xr.Dataset, barrier=CONNECTION_THRESHOLD) -> np.ndarray:
    """Calculate the local clustering coefficient.

    Args:
        ds (xr.Dataset): Input dataset containing 'network' variable.
        barrier (float): Threshold to determine connections.

    Returns:
        np.ndarray: Array of clustering coefficients per node.

    """
    A = find_neighbourhood(ds, barrier)

    # --- Calculate possible connections (k * (k-1)) ---
    degree = A.sum(axis=1)
    possible_connections = degree * (degree - 1)

    # --- Calculate actual connections (triangles) ---
    actual = actual_connections(ds, barrier)

    # --- Compute Coefficient ---
    coeffs = np.zeros_like(possible_connections, dtype=float)

    # Only divide where possible_connections > 0
    mask = possible_connections > 0
    coeffs[mask] = actual[mask] / possible_connections[mask]

    return coeffs


def feed_forward_loop(ds: xr.Dataset, barrier=CONNECTION_THRESHOLD) -> np.ndarray:
    """Calculate the feed forward loops for nodes.

    Args:
        ds (xr.Dataset): Input dataset containing 'network' variable.
        barrier (float): Threshold to determine connections.

    Returns:
        np.ndarray: Array of feed forward loops per node.

    """
    # This is just # of actual_connections
    return actual_connections(ds, barrier)


# --- Sums and Averages ---
def yearly_clustering_coefficients(
    scenario: int, year: int, barrier: float = CONNECTION_THRESHOLD
) -> np.ndarray:
    """Calculate yearly averaged clustering coefficients per node.

    Args:
        scenario (int): Scenario identifier.
        year (int): Year of the data.
        barrier (float): Threshold to determine connections.

    Returns:
        np.ndarray: Yearly averaged clustering coefficients per node.

    """
    # Use the generalized compute_and_cache function
    return compute_and_cache(
        scenario,
        year,
        "clustering",
        clustering_coefficients,
        func_args={"barrier": barrier},
        do_average=True,
    )


def yearly_feed_forward_loop(
    scenario: int, year: int, barrier=CONNECTION_THRESHOLD
) -> np.ndarray:
    """Calculate yearly averaged # of FFLs per node.

    Args:
        scenario (int): Scenario identifier.
        year (int): Year of the data.
        barrier (float): Threshold to determine connections.

    Returns:
        np.ndarray: Yearly averaged feed forward loops per node.

    """
    return compute_and_cache(
        scenario,
        year,
        "ffl",
        feed_forward_loop,
        func_args={"barrier": barrier},
        do_average=True,
    )


def yearly_average_diff_clustering_coefficients(
    scenario: int, year: int, barrier=CONNECTION_THRESHOLD
) -> float:
    """Calculate difference in yearly average of clustering coefficient.

    Args:
        scenario (int): Scenario identifier.
        year (int): Year of the data.
        barrier (float): Threshold to determine connections.

    Returns + Saves:
        np.ndarray: Difference in yearly average of clustering coefficients.

    """
    current_year = yearly_clustering_coefficients(scenario, year, barrier)
    base_year = yearly_clustering_coefficients(scenario, 2030, barrier)

    diff = current_year - base_year
    return np.nansum(diff) / len(diff)  # Take average over all nodes


def yearly_average_diff_feed_forward_loop(
    scenario: int, year: int, barrier=CONNECTION_THRESHOLD
) -> float:
    """Calculate difference in yearly average of feed forward loops.

    Args:
        scenario (int): Scenario identifier.
        year (int): Year of the data.
        barrier (float): Threshold to determine connections.

    Returns + Saves:
        np.ndarray: Difference in yearly average of feed forward loops.

    """
    current_year = yearly_feed_forward_loop(scenario, year, barrier)
    base_year = yearly_feed_forward_loop(scenario, 2030, barrier)

    diff = current_year - base_year
    return np.nansum(diff) / len(diff)


if __name__ == "__main__":
    scenario = 245
    years = list(range(2030, 2100, 10))

    for year in years:
        clustering_vals = yearly_clustering_coefficients(scenario, year)
        ffl_vals = yearly_feed_forward_loop(scenario, year)
