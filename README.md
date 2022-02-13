# Final Project - Deep Learning Course

## Summary

The library includes data and scripts to reproduce the experiments reported in the report.\
**Please note, using the default configuration of 10 validation folds takes a very long
time (for us 3 days) for each run!**

### Instructions

First run `pip install -r requirements.txt` to install required dependecies.
Note this might take a while due to the installation of torch-geometric and related packages.

The datasets needed to reproduce the experiments are in the 'DATA' folder,
to create a new instance of the data proceed in the following steps:

`python PrepareDatasets.py DATA/CHEMICAL --dataset-name [ENZYMES/NCI1] --outer-k 10`

Where 'outer-k' are the number of cross validation folds to be used (see method section in the report)
and [ENZYMES/NCI1] means you should choose exactly one of the options.

Please note that dataset folders should be organized as follows:
    
    DATA
        CHEMICAL:
            NCI1
            ENZYMES
        ENZYMES
        NCI1

Then, you can launch experiments by typing:

`python Launch_Experiments.py --config-file <config> --dataset-name <name> --result-folder <your-result-folder>`

Where `<config>` is your config file (e.g. config_GIN.yml), and `<name>` is the dataset name chosen as before.

### Configurations

The available configurations to run are: `GIN, GIN_FA, GIN_wCA`.
* For all configuration, if you wish to run on `cuda` you must change the `device` variable
to `cuda` in the `.yml` file.
* The `config_GIN*.yml` file refers to the baseline configuration. 
The `last_layer_fa` variable refers to the FA layer at the last position and is **boolean**.
* The `config_GIN_FA*.yml` file refers to the variation of the FA configuration. 
The `last_layer_fa` variable refers to the **position** of the FA layer and is an **integer**.
* The `config_GIN_wCA*.yml` file refers to the variation of the wCA configuration. 
The `last_layer_fa` variable refers to the **position** of the wCA layer and is an **integer**.
* Configuration files where the `*` is empty (e.g `config_GIN.yml`) should be used for the ENZYMES dataset.
Configuration files where the `*` is `_NCI1` should be used for the NCI1 dataset.
The Difference is the number of configurations, for NCI1
there is no hyper-parameter search (single configuration) , but both configurations can be used for both datasets.