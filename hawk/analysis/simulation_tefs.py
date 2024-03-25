import time

from tefs import TEFS


def run(
    datasets,
    config,
    n_jobs=1,
):
    params = config["params"]
    dataset_name = config["dataset_name"]
    dataframe = datasets[dataset_name]

    # extract the parameters
    direction = params["direction"]
    lagfeatures = params["lagfeatures"]
    lagtarget = params["lagtarget"]
    k = params["k"]

    # Construct a unique identifier for the configuration
    # param_str = "_".join(f"{k}{v}" for k, v in params.items())
    # param_str = param_str.replace(" ", "")
    # config_id = f"dataset{dataset_name}_{param_str}"

    features = dataframe["full"].drop(columns=["target"])
    target = dataframe["full"]["target"]
    var_names = list(features.columns)

    # run the feature selection algorithm
    start_time = time.time()
    fs = TEFS(
        features=features.values,
        target=target.values,
        k=k,
        lag_features=lagfeatures,
        lag_target=lagtarget,
        direction=direction,
        verbose=1,
        var_names=var_names,
        n_jobs=n_jobs,
    )
    fs.fit()
    end_time = time.time()
    execution_time = end_time - start_time

    # Save results to the dictionary
    return {
        "results": fs,
        "params": params,
        "dataset_name": dataset_name,
        "execution_time": execution_time,
    }
