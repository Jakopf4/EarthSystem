## Project Explanation

This project investigates the resilience and dynamics of the Amazon rainforest’s hydrological cycle by modeling atmospheric moisture transport, where water is recycled through evapotranspiration and precipitation.

The core of this analysis involves transforming raw, gridded spatio-temporal water flux data into plots and animations. In the initial data, each grid cell represents a network node with information about evaporation, precipitation as well as the water flow into every other grid cell.
One can analyze the system's behavior under various IPCC climate change scenarios (SSP245, SSP370 and SSP585). Key characteristics such as Clustering Coefficients, In-/Out-Degrees, and Feed-Forward Loops (FFLs) are calculated to evaluate the network's local stability, the identification of moisture sources and sinks, and the persistence of directional flow patterns.

Another significant component of this study is a sensitivity analysis focused on deforestation. One can simulate land-use changes by integrating external deforestation maps and reducing the evapotranspiration capacity of affected nodes. To model the system-wide impact of these local perturbations, we apply a flux correction algorithm that redistributes the remaining moisture flow. By calculating the difference between the perturbed and original network states, we quantify the non-linear propagation of water loss across the basin.


## Usage

1. Setup  
First, clone the repository to your local machine:

    git clone https://github.com/Jakopf4/EarthSystem.git
    cd EarthSystem

    Data Verification: Before running scripts, ensure the data/water directory is populated with the necessary .nc (NetCDF) files.

2. Choosing a Script  
Navigate to the scripts/ folder. Choose the script that matches your goal:

    anim_gen.py → Animations: Use this to generate time-lapse videos of a specific physical quantity.

    plot_all.py → Static Plots: Use this to generate single plots or calculated indices.

3. Configuration & Execution  
Currently, parameters are configured by modifying the script entry point.

    Open the file (anim_gen.py or plot_all.py) in your code editor.

    Scroll to the bottom of the file (the main execution block).

    Modify the function arguments to match your desired parameters:

        Scenario (e.g., 'ssp585')

        Year (e.g., 2030)

        Month (e.g., 01)

    Run the script from the root directory:

        python3 scripts/anim_gen.py

4. Outputs  
    Visualizations: Results are saved to the results/ directory (organized by logical subfolders).

    Logs/Errors: Execution status and error messages will appear directly in the terminal console.



## Project Structure

├── data/  
│   ├── deforestation/  
│   │   ├── deforestation_all_BaU_...       # Deforestation maps  
│   │   ├── deforestation_scenario_...      # Network data including deforestation effects  
│   │   └── ...  
│   ├── water/                              # Original data  
│   │   ├── scenario_ssp245_...  
│   │   ├── scenario_ssp370_...  
│   │   └── scenario_ssp585_...  
│   └── dummy_scenario_ssp1...nc            # Dummy data for testing  
│  
├── results/  
│   ├── cache/                              # Intermediate computed data  
│   └── plots/                              # Cache of plots for animations  
│  
├── scripts/                                # All executing functions + Testing  
│   ├── anim_gen.py  
│   └── ...  
│  
├── README.md  
└── requirements.txt  