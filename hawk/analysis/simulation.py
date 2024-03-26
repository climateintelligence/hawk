import time

from tefs import TEFS
from tigramite.pcmci import PCMCI


def run_simulation_pcmci(
    datasets,
    config,
    independence_tests,
):
    params = config["params"]
    dataset_name = config["dataset_name"]
    dataframe = datasets[dataset_name]

    independence_test = independence_tests[params["independencetest"]]
    algorithm = params["algorithm"]
    lag = params["lag"]

    # Construct a unique identifier for the configuration
    # param_str = "_".join(f"{k}{v}" for k, v in params.items())
    # config_id = f"dataset{dataset_name}_{param_str}"

    print(f"Running experiment with config: {config}")

    pcmci = PCMCI(dataframe=dataframe["full_tigramite"], cond_ind_test=independence_test, verbosity=2)

    start_time = time.time()
    if algorithm == "pcmci":
        results = pcmci.run_pcmci(tau_max=lag, pc_alpha=0.05, alpha_level=0.01)
    elif algorithm == "pcmci_plus":
        results = pcmci.run_pcmciplus(tau_min=0, tau_max=lag)
    else:
        raise ValueError(f"Invalid algorithm {algorithm}")
    end_time = time.time()
    execution_time = end_time - start_time

    q_matrix = pcmci.get_corrected_pvalues(
        p_matrix=results["p_matrix"],
        tau_max=lag,
        fdr_method="fdr_bh",
    )

    graph = pcmci.get_graph_from_pmatrix(
        p_matrix=q_matrix,
        alpha_level=0.01,
        tau_min=0,
        tau_max=lag,
        link_assumptions=None,
    )

    results["graph"] = graph

    return {
        "results": results,
        "params": params,
        "dataset_name": dataset_name,
        "execution_time": execution_time,
    }


def run_simulation_tefs(
    datasets,
    config,
    target_column_name,
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

    features = dataframe["full"].drop(columns=[target_column_name])
    target = dataframe["full"][target_column_name]
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
