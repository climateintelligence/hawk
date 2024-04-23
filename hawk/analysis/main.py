import itertools
import os

import pandas as pd
from tefs.metrics import regression_analysis
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.parcorr import ParCorr

from .file_management import save_to_pkl_file
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
        response=None,
    ):
        self.response = response

        # Move target column as last column for the get_connected_variables
        # function which requires it (TODO this would be interesting to be fixed)
        df_train = df_train[[col for col in df_train.columns if col != target_column_name] + [target_column_name]]
        df_test = df_test[[col for col in df_test.columns if col != target_column_name] + [target_column_name]]

        df_full = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
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
        if self.tefs_max_lag_features > 0:
            self.tefs_features_lags.extend(list(range(1, self.tefs_max_lag_features + 1)))

        self.tefs_target_lags = list(range(1, self.tefs_max_lag_target + 1))

        self.pcmci_features_lags = list(range(0, self.pcmci_max_lag + 1))

        self.baseline = None
        self.plot_pcmci = {}
        self.details_pcmci = None
        self.plot_tefs = {}
        self.details_tefs = None
        self.plot_tefs_wrapper = {}
        self.details_tefs_wrapper = None

    def run_baseline_analysis(self):
        baseline = {}

        features_names = self.datasets["normal"]["var_names"]
        
        print(f"features_names: {features_names}")

        configs = []

        # Only autoregressive baselines from 1 to the maximum target lag 
        for i in range(1, self.tefs_max_lag_target+1):
            configs.append((f"AR({i})", {self.target_column_name: list(range(1, i + 1))}))
            
        print(f"AR configs: {configs}")

        # All features without AR
        configs.append(
            (
                "All features",
                {feature: self.tefs_features_lags for feature in features_names if feature != self.target_column_name},
            )
        )
        
        print(f"Full configs: {configs}")
        
        # All features with AR
        configs.append(
            (
                "All features and AR",
                {feature: self.tefs_features_lags if feature != self.target_column_name else list(range(1, self.tefs_max_lag_target+1)) for feature in features_names},
            )
        )
        
        print(f"Full configs: {configs}")        

        for label, inputs_names_lags in configs:
            baseline[label] = {
                "inputs": inputs_names_lags,
                "r2": regression_analysis(
                    inputs_names_lags=inputs_names_lags,
                    target_name=self.target_column_name,
                    df_train=self.datasets["normal"]["train"],
                    df_test=self.datasets["normal"]["test"],
                ),
            }
            
        print(f"Baselines: {baseline}")

        target_file = os.path.join(self.workdir, "baseline.pkl")
        save_to_pkl_file(target_file, baseline)

        return target_file

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
            
        print(f"TEFS configurations: {configurations}")

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
        
        print(f"TEFS results: {results}")

        return results

    def run_pcmci_analysis(
        self,
    ):
        lag_options = self.pcmci_features_lags  # list from 0 to max_lag

        # Define the tests
        parcorr = ParCorr(significance="analytic")
        #cmiknn = CMIknn(
        #    significance="shuffle_test",
        #    knn=0.1,
        #    shuffle_neighbors=5,
        #    transform="ranks",
        #    sig_samples=200,
        #)
        cmiknn = CMIknn()

        # Create the dictionary of tests
        independence_tests = {
            "parcorr": parcorr,
            "cmiknn": cmiknn,
        }

        if self.pcmci_test_choice == "ParCorr":
            independence_tests_options = ["parcorr"]
        elif self.pcmci_test_choice == "CMIknn":
            independence_tests_options = ["cmiknn"]

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
            
        print(f'PCMCI configurations: {configurations}')

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
        print(f"PCMCI+ results: {results}")
        
        save_to_pkl_file("/work/bk1318/b382633/pcmci_results.pkl", results)
        
        return results

    def run(self):
        if self.response:
            self.response.update_status("Performing baseline analysis", 16)
        self.baseline = self.run_baseline_analysis()
        if self.response:
            self.response.update_status("Performing TEFS analysis", 33)
        tefs_results = self.run_tefs_analysis()
        #if self.response:
        #    self.response.update_status("Performing PCMCI analysis", 66)
        #pcmci_results = self.run_pcmci_analysis()

        #if self.response:
        #    self.response.update_status("Postprocessing PCMCI", 80)
        #self.plot_pcmci, self.details_pcmci = run_postprocessing_pcmci(
        #    results_pcmci=pcmci_results,
        #    target_column_name=self.target_column_name,
        #    datasets=self.datasets,
        #    destination_path=self.workdir,
        #    image_formats=["pdf", "png"],
        #)
        if self.response:
            self.response.update_status("Postprocessing TEFS", 90)
        self.plot_tefs, self.details_tefs = run_postprocessing_tefs(
            results_tefs=tefs_results,
            target_column_name=self.target_column_name,
            datasets=self.datasets,
            destination_path=self.workdir,
            image_formats=["pdf", "png"],
        )
        #if self.response:
        #    self.response.update_status("Postprocessing TEFS Wrapper", 95)
        #self.plot_tefs_wrapper, self.details_tefs_wrapper = run_postprocessing_tefs_wrapper(
        #    results_tefs=tefs_results,
        #    target_column_name=self.target_column_name,
        #    datasets=self.datasets,
        #    destination_path=self.workdir,
        #    image_formats=["pdf", "png"],
        #)
