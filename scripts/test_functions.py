"""Test file for the in_degrees function."""

from pathlib import Path

import numpy as np

import xarray as xr


def test_in_degrees():
    """Test the in_degrees function."""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    filepath = project_root / "data" / "dummy_scenario_ssp1_decade2030_month01.nc"

    ds = xr.open_dataset(filepath)

    from inout import in_degrees

    result = in_degrees(ds)

    # Assertions
    expected = np.array([4, 2, 4, 2])
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, expected)


def test_out_degrees():
    """Test the out_degrees function."""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    filepath = project_root / "data" / "dummy_scenario_ssp1_decade2030_month01.nc"

    ds = xr.open_dataset(filepath)

    from inout import out_degrees

    result = out_degrees(ds)

    # Assertions
    expected = np.array([6, 4, 0, 2])
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, expected)


def test_neighbourhood():
    """Test the neighbourhood finding function."""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    filepath = project_root / "data" / "dummy_scenario_ssp1_decade2030_month01.nc"

    ds = xr.open_dataset(filepath)

    from clustering import find_neighbourhood

    result = find_neighbourhood(ds, barrier=1.14)

    # Assertions
    expected = np.array([[0, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, expected)


def test_clustering():
    """Test the clustering coefficients function."""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    filepath = project_root / "data" / "dummy_scenario_ssp1_decade2030_month01.nc"

    ds = xr.open_dataset(filepath)

    from clustering import clustering_coefficients

    result = clustering_coefficients(ds, barrier=1.14)

    # Assertions
    expected = np.array([1/6, 1/2, 0, 0])
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, expected)


def test_ffl():
    """Test the feed forward loop function."""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    filepath = project_root / "data" / "dummy_scenario_ssp1_decade2030_month01.nc"

    ds = xr.open_dataset(filepath)

    from clustering import feed_forward_loop

    result = feed_forward_loop(ds, barrier=1.14)

    # Assertions
    expected = np.array([1, 1, 0, 0])
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, expected)


if __name__ == "__main__":
    test_in_degrees()
    print("In-degrees test passed.")

    test_out_degrees()
    print("Out-degrees test passed.")

    test_neighbourhood()
    print("Neighbourhood test passed.")

    test_clustering()
    print("Clustering test passed.")

    test_ffl()
    print("Feed forward loop test passed.")
