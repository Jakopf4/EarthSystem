scripts:

     check_data.py:     Create a basic overview of provided data structure
     io_utils.py:       Basic methods to handle functions with storing data
     inout.py:          Handles calculations with In-/Out-Degrees of the Data Nodes
     clustering.py:     Handles calculations with Clustering and FFLs of the Data Nodes
     deforestation.py:  Handles all calculations, but with included deforestation data
     plot_all.py:       Handles all plotting functions used in the project
     anim_gen.py:       Creates an animation from previously stored plots


tests:

    Execute with (in folder /scripts) python3 -m test_functions
         or with (in folder /scripts) python3 -m pytest

    dummy_file.py:      Creates a dummy network with hand calculated results to compare
    test_functions.py:  Uses the dummy file and known results to check the functionality of the functions