import itertools

from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.parcorr import ParCorr

import hawk.analysis.pcmci_tools as pcmci_tools
import hawk.analysis.simulation_pcmci as simulation_pcmci
import hawk.analysis.simulation_tefs as simulation_tefs
from hawk.analysis.metrics import regression_analysis


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
        self.df_train = df_train
        self.df_test = df_test
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

        features_names = self.df_train.columns.tolist()

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
                    df_train=self.df_train,
                    df_test=self.df_test,
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

        # Define the different dataframes to use
        datasets = {
            "normal": {
                "full": df_ticino,
                "train": df_ticino_train,
                "test": df_ticino_test,
                "var_names": df_ticino.columns,
            },
        }

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
        for config in configurations:
            simulation_tefs.run(
                datasets=datasets,
                config=config,
            )

    def run_pcmci_analysis(
        self,
    ):
        lag_options = [self.pcmci_features_lags[: i + 1] for i in range(len(self.pcmci_features_lags))]

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

        results = []
        for config in configurations:
            results.append(
                simulation_pcmci.run(
                    datasets=datasets,
                    config=config,
                    independence_tests=independence_tests,
                )
            )

    def run(self):
        self.baseline = self.run_baseline_analysis()
