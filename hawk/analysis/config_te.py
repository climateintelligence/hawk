import itertools

import numpy as np

# Load here the dataset
# ...

np.random.seed(42)

# Define the different dataframes to use
datasets = {
    "normal": {
        "full": df_ticino,
        "train": df_ticino_train,
        "test": df_ticino_test,
        "var_names": df_ticino.columns,
    },
}

# Constants
# - `threshold` is set to be large in the forward direction (give me all the information) and 0 in the backward direction.
# - `k` rule of thumb: $1/20$ of the number of samples (try 5,10,20,30...) (TODO)
lagtarget = [1]
threshold_forward = float("inf")
threshold_backward = 0
k = 10

# Variables set by the configuration
lagfeatures_options = [[0], [0, 1]]
directions = ["forward", "backward"]
dataset_names = [
    "normal",
]

# Generating the configurations
configurations = []

for lagfeatures, direction, dataset_name in itertools.product(lagfeatures_options, directions, dataset_names):
    threshold = threshold_forward if direction == "forward" else threshold_backward
    configuration = {
        "params": {
            "lagfeatures": lagfeatures,
            "lagtarget": lagtarget,
            "direction": direction,
            "threshold": threshold,  # NOTE: the threshold is set here, although it is not used during the simulation, but only during the postprocessing, might be better to change this behavior
            "k": k,
        },
        "dataset_name": dataset_name,
    }
    configurations.append(configuration)


def load_te():
    return datasets, configurations
