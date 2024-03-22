import argparse
import os
import time

import thesis.constants as constants
import thesis.file_management as file_management
from tefs import TEFS
from thesis import datasets_and_configurations_loaders


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run simulation script for Transfer Entropy analysis.")
    parser.add_argument("--basin", type=str, help="Name of the basin")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs to run")
    args = parser.parse_args()

    loader = datasets_and_configurations_loaders["te"].get(args.basin)

    if not loader:
        raise ValueError("Invalid basin name")

    datasets, configurations = loader()

    # Run experiments
    for config in configurations:
        params = config["params"]
        dataset_name = config["dataset_name"]
        dataframe = datasets[dataset_name]

        # extract the parameters
        direction = params["direction"]
        lagfeatures = params["lagfeatures"]
        lagtarget = params["lagtarget"]
        k = params["k"]

        # Construct a unique identifier for the configuration
        param_str = "_".join(f"{k}{v}" for k, v in params.items())
        param_str = param_str.replace(" ", "")
        config_id = f"dataset{dataset_name}_{param_str}"
        target_file = os.path.join(constants.path_results, f"te_{args.basin}_{config_id}.pkl")
        if os.path.exists(target_file):
            print(f"Skipping config {config_id} because results already exist")
            continue

        print(f"Running experiment with config: {config}")

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
            n_jobs=args.n_jobs,
        )
        fs.fit()
        end_time = time.time()
        execution_time = end_time - start_time

        # Save results to the dictionary
        current_result = {
            "results": fs,
            "params": params,
            "dataset_name": dataset_name,
            "basin": args.basin,
            "execution_time": execution_time,
        }

        # Save the object to a pickle file
        file_management.save_to_pkl_file(target_file=target_file, data=current_result)

        print("-" * 80)


if __name__ == "__main__":
    main()
