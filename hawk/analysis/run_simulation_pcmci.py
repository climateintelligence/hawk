import argparse
import os
import time

import thesis.constants as constants
import thesis.file_management as file_management
from thesis import datasets_and_configurations_loaders
from tigramite.pcmci import PCMCI


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run simulation script for Transfer Entropy analysis.")
    parser.add_argument("--basin", type=str, help="Name of the basin")
    args = parser.parse_args()

    loader = datasets_and_configurations_loaders["pcmci"].get(args.basin)

    if not loader:
        raise ValueError("Invalid basin name")

    datasets, configurations, independence_tests = loader()

    # Run experiments
    for config in configurations:
        params = config["params"]
        dataset_name = config["dataset_name"]
        dataframe = datasets[dataset_name]

        independence_test = independence_tests[params["independencetest"]]
        algorithm = params["algorithm"]
        lag = params["lag"]

        # Construct a unique identifier for the configuration
        param_str = "_".join(f"{k}{v}" for k, v in params.items())
        config_id = f"dataset{dataset_name}_{param_str}"
        target_file = os.path.join(constants.path_results, f"pcmci_{args.basin}_{config_id}.pkl")
        if os.path.exists(target_file):
            print(f"Skipping config {config_id} because results already exist")
            continue

        print(f"Running experiment with config: {config}")

        pcmci = PCMCI(dataframe=dataframe["full_tigramite"], cond_ind_test=independence_test, verbosity=2)

        # if inspect_data:
        #     # Investigating data dependencies and lag functions
        #     correlations = pcmci.get_lagged_dependencies(tau_max=20, val_only=True)['val_matrix']

        #     matrix_lags = None #np.argmax(np.abs(correlations), axis=2)
        #     tp.plot_scatterplots(dataframe=dataframe, add_scatterplot_args={'matrix_lags':matrix_lags}); plt.show()

        #     tp.plot_densityplots(dataframe=dataframe, add_densityplot_args={'matrix_lags':matrix_lags}); plt.show()

        start_time = time.time()
        if algorithm == "pcmci":
            results = pcmci.run_pcmci(tau_max=lag, pc_alpha=0.05, alpha_level=0.01)
        elif algorithm == "pcmci_plus":
            results = pcmci.run_pcmciplus(tau_min=0, tau_max=lag)
        else:
            raise ValueError(f"Invalid algorithm {algorithm}")
        end_time = time.time()
        execution_time = end_time - start_time

        # if show_p_val:
        #     print("p-values")
        #     print(results['p_matrix'].round(3))
        #     print("MCI partial correlations")
        #     print(results['val_matrix'].round(2))

        q_matrix = pcmci.get_corrected_pvalues(
            p_matrix=results["p_matrix"],
            tau_max=lag,
            fdr_method="fdr_bh",
        )

        # if print_significant_links:
        #     pcmci.print_significant_links(
        #             p_matrix = q_matrix,
        #             val_matrix = results['val_matrix'],
        #             alpha_level = 0.01)

        graph = pcmci.get_graph_from_pmatrix(
            p_matrix=q_matrix,
            alpha_level=0.01,
            tau_min=0,
            tau_max=lag,
            link_assumptions=None,
        )

        results["graph"] = graph

        # Save results to the dictionary
        current_result = {
            "results": results,
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
