import itertools

import pandas as pd
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.parcorr import ParCorr

from .metrics import regression_analysis
from .pcmci_tools import initialize_tigramite_df
from .postprocessing import (
    run_postprocessing_pcmci,
    run_postprocessing_tefs,
    run_postprocessing_tefs_wrapper,
)
from .simulation import run_simulation_pcmci, run_simulation_tefs


class CausalAnalysis:
    def __init__(
        self,
        df_train,
        df_test,
        target_column_name,
        pcmci_test_choice,
        pcmci_max_lag,
        tefs_direction,
        tefs_use_contemporary_features,
        tefs_max_lag_features,
        tefs_max_lag_target,
        workdir,
    ):
        df_full = pd.concat([df_train, df_test], axis=1).reset_index(drop=True)
        df_full_tigramite = initialize_tigramite_df(df_full)

        self.datasets = {
            "normal": {
                "full_tigramite": df_full_tigramite,
                "full": df_full,
                "train": df_train,
                "test": df_test,
                "var_names": df_train.columns.tolist(),
            },
        }

        self.target_column_name = target_column_name
        self.pcmci_test_choice = pcmci_test_choice
        self.pcmci_max_lag = pcmci_max_lag
        self.tefs_direction = tefs_direction
        self.tefs_use_contemporary_features = tefs_use_contemporary_features
        self.tefs_max_lag_features = tefs_max_lag_features
        self.tefs_max_lag_target = tefs_max_lag_target
        self.workdir = workdir

        self.tefs_features_lags = []
        if self.tefs_use_contemporary_features:
            self.tefs_features_lags.append(0)
        self.tefs_features_lags.extend(list(range(1, self.tefs_max_lag_features + 1)))

        self.tefs_target_lags = list(range(1, self.tefs_max_lag_target + 1))

        self.pcmci_features_lags = list(range(0, self.pcmci_max_lag + 1))

        self.baseline = None
        self.plot_pcmci = None
        self.details_pcmci = None
        self.plot_tefs = None
        self.details_tefs = None
        self.plot_tefs_wrapper = None
        self.details_tefs_wrapper = None

    def run_baseline_analysis(self):
        baseline = {}

        features_names = self.datasets["normal"]["var_names"]

        configs = []

        # Autoregressive baselines
        for i in range(1, self.tefs_max_lag_target):
            configs.append((f"AR({i})", {self.target_column_name: list(range(1, i + 1))}))

        # With all features
        configs.append(
            (
                "All features",
                {feature: self.tefs_features_lags for feature in features_names},
            )
        )

        for label, inputs_names_lags in configs:
            baseline[label] = {
                "inputs": inputs_names_lags,
                "r2": regression_analysis(
                    inputs_names_lags=inputs_names_lags,
                    target_name=self.target_column_name,
                    df_train=self.datasets["normale"]["train"],
                    df_test=self.datasets["normale"]["test"],
                ),
            }

        return baseline

    def run_tefs_analysis(
        self,
        k=10,
        threshold_forward=float("inf"),
        threshold_backward=0,
    ):
        # Grid of options

        lagtarget_options = [self.tefs_target_lags[: i + 1] for i in range(len(self.tefs_target_lags))]

        lagfeatures_options = [self.tefs_features_lags[: i + 1] for i in range(len(self.tefs_features_lags))]

        if self.tefs_direction == "both":
            directions = ["forward", "backward"]
        else:
            directions = [self.tefs_direction]

        dataset_names = [
            "normal",
        ]

        # Create the configurations
        configurations = []

        for lagfeatures, lagtarget, direction, dataset_name in itertools.product(
            lagfeatures_options, lagtarget_options, directions, dataset_names
        ):
            threshold = threshold_forward if direction == "forward" else threshold_backward
            configuration = {
                "params": {
                    "lagfeatures": lagfeatures,
                    "lagtarget": lagtarget,
                    "direction": direction,
                    "threshold": threshold,
                    "k": k,
                },
                "dataset_name": dataset_name,
            }
            configurations.append(configuration)

        # Run the analysis
        results = []
        for config in configurations:
            results.append(
                run_simulation_tefs(
                    datasets=self.datasets,
                    target_column_name=self.target_column_name,
                    config=config,
                )
            )

        return results

    def run_pcmci_analysis(
        self,
    ):
        lag_options = [self.pcmci_features_lags[: i + 1] for i in range(len(self.pcmci_features_lags))]

        # Define the tests
        parcorr = ParCorr(significance="analytic")
        cmiknn = CMIknn(
            significance="shuffle_test",
            knn=0.1,
            shuffle_neighbors=5,
            transform="ranks",
            sig_samples=200,
        )

        # Create the dictionary of tests
        independence_tests = {
            "parcorr": parcorr,
            "cmiknn": cmiknn,
        }

        independence_tests_options = [
            "parcorr",
            "cmiknn",
        ]

        algorithm_options = [
            "pcmci_plus",
        ]

        dataset_options = [
            "normal",
        ]

        # Generating the configurations
        configurations = []

        for lag, independencetest, algorithm, dataset_name in itertools.product(
            lag_options, independence_tests_options, algorithm_options, dataset_options
        ):
            configuration = {
                "params": {
                    "lag": lag,
                    "independencetest": independencetest,
                    "algorithm": algorithm,
                },
                "dataset_name": dataset_name,
            }
            configurations.append(configuration)

        # Run the analysis
        results = []
        for config in configurations:
            results.append(
                run_simulation_pcmci(
                    datasets=self.datasets,
                    config=config,
                    independence_tests=independence_tests,
                )
            )

        return results

    def run(self):
        self.baseline = self.run_baseline_analysis()
        tefs_results = self.run_tefs_analysis()
        pcmci_results = self.run_pcmci_analysis()

        self.plot_pcmci, self.details_pcmci = run_postprocessing_pcmci(
            results_pcmci=pcmci_results,
            target_column_name=self.target_column_name,
            datasets=self.datasets,
            destination_path=self.workdir,
        )
        self.plot_tefs, self.details_tefs = run_postprocessing_tefs(
            results_tefs=tefs_results,
            target_column_name=self.target_column_name,
            datasets=self.datasets,
            destination_path=self.workdir,
        )
        self.plot_tefs_wrapper, self.details_tefs_wrapper = run_postprocessing_tefs_wrapper(
            results_tefs=tefs_results,
            target_column_name=self.target_column_name,
            datasets=self.datasets,
            destination_path=self.workdir,
        )
