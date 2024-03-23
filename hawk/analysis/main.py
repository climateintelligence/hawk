import hawk.analysis.pcmci_tools as pcmci_tools
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
        tefs_use_comtemporary_features,
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
        self.tefs_use_comtemporary_features = tefs_use_comtemporary_features
        self.tefs_max_lag_features = tefs_max_lag_features
        self.tefs_max_lag_target = tefs_max_lag_target
        self.workdir = workdir

        self.tefs_features_lags = []
        if self.tefs_use_comtemporary_features:
            self.tefs_features_lags.append(0)
        self.tefs_features_lags.extend(
            list(range(1, self.tefs_max_lag_features + 1))
        )

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
        configs.append(("All features", {feature: self.tefs_features_lags for feature in features_names}))


        for label, inputs_names_lags in configs:
            baseline[label] = {
                "inputs": inputs_names_lags,
                "r2": regression_analysis(
                    inputs_names_lags=inputs_names_lags,
                    target_name=self.target_column_name,
                    df_train=self.df_train,
                    df_test=self.df_test,
                )
            }

        return baseline
    
    def run(self):
        self.baseline = self.run_baseline_analysis()
