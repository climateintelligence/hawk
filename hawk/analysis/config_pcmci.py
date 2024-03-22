import itertools

import numpy as np
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.parcorr import ParCorr

from ..data.data_ticino import (
    df_ticino,
    df_ticino_snowlakes,
    df_ticino_snowlakes_test,
    df_ticino_snowlakes_tigramite,
    df_ticino_snowlakes_train,
    df_ticino_test,
    df_ticino_tigramite,
    df_ticino_train,
    var_names_ticino,
    var_names_ticino_snowlakes,
)

seed = 42
np.random.seed(seed)

# Define the tests
parcorr = ParCorr(significance="analytic")
cmiknn = CMIknn(significance="shuffle_test", knn=0.1, shuffle_neighbors=5, transform="ranks", sig_samples=200)

# Create the dictionary of tests
independence_tests = {
    "parcorr": parcorr,
    "cmiknn": cmiknn,
}

# Create the dictionary of datasets
datasets = {
    "snowlakes": {
        "full_tigramite": df_ticino_snowlakes_tigramite,
        "full": df_ticino_snowlakes,
        "train": df_ticino_snowlakes_train,
        "test": df_ticino_snowlakes_test,
        "var_names": var_names_ticino_snowlakes,
    },
}

# Variables
lag_options = [
    0,
    1,
]
independence_tests_options = [
    "parcorr",
    "cmiknn",
]
# NOTE add here if you want the base algorithm as well
algorithm_options = [
    "pcmci_plus",
]
dataset_options = [
    "normal",
    "snowlakes",
]

# Generating the configurations
configurations = []

for lag, independencetest, algorithm, dataset_name in itertools.product(lag_options, independence_tests_options, algorithm_options, dataset_options):
    configuration = {
        "params": {
            "lag": lag,
            "independencetest": independencetest,
            "algorithm": algorithm,
        },
        "dataset_name": dataset_name,
    }
    configurations.append(configuration)


def load_ticino():
    return datasets, configurations, independence_tests
