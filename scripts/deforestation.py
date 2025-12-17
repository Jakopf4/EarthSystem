import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# ==========================================
# 1. SETUP & ECHTE DATEN LADEN
# ==========================================
scenario = 245
year = 2030
month = 1

def deforestation_inout(scenario: int, year: int, month: int):
    """Look at difference in water flow due to deforestation.
    
    Args:
        scenario (int): SSP scenario number.
        year (int): Year of the scenario.
        month (int): Month of the scenario.

    Returns:
        in_values (np.ndarray): Array of in-degrees per node.
        out_values (np.ndarray): Array of out-degrees per node.
        
    """

    nc_filepath = f"../data/water/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    txt_filepath = f"../data/deforestation/deforestation_all_BaU_{year}.txt"

    ds = xr.open_dataset(nc_filepath)
    data_deforestation = np.genfromtxt(txt_filepath)

    orig_matrix = ds["network"].values
    deforest_values = data_deforestation[:, 2]

    # Incoming water without deforestation values
    orig_col_sums = np.sum(orig_matrix, axis=0)

    # Create a copy to work on and keep original intact
    current_matrix = orig_matrix.copy()

    # Adjust all transportation values based on deforestation
    factor_matrix = (1 - deforest_values[:, np.newaxis])
    current_matrix = current_matrix * factor_matrix

    # Incoming water with deforestation values
    new_col_sums = np.sum(current_matrix, axis=0)

    ratios = np.zeros_like(orig_col_sums)

    # Avoid division by zero
    mask = orig_col_sums != 0
    # Calculate ratios of incoming water with and without deforestation
    ratios[mask] = new_col_sums[mask] / orig_col_sums[mask]

    # Reduce water flow based on lower incoming water
    deforested_matrix = current_matrix * ratios[:, np.newaxis]

    in_values = np.nansum(deforested_matrix, axis=0)
    out_values = np.nansum(deforested_matrix, axis=1)

    return in_values, out_values


def yearly_deforestation_inout(scenario: int, year: int):
    """Calculate yearly in-degrees and out-degrees of nodes in the network with deforestation.
    Args:
        scenario (int): SSP scenario number.
        year (int): Year of the scenario.

    Returns:
        in_values (np.ndarray): Array of in-degrees per node.
        out_values (np.ndarray): Array of out-degrees per node.

    """
    for month in range(1, 13):
        in_values_month, out_values_month = deforestation_inout(scenario, year, month)
        
        if month == 1:
            in_values_year = in_values_month
            out_values_year = out_values_month
        else:
            in_values_year += in_values_month
            out_values_year += out_values_month
     
    return in_values_year, out_values_year
