import os
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# from tigramite import plotting as tp
from .file_management import save_to_pkl_file
from .metrics import regression_analysis
from .pcmci_tools import get_connected_variables


# Adjusted custom sort key function to handle lag sequences and replace them with the last lag
def general_custom_sort_key(s):
    # Find all sequences like [0,1], [1,2,3,5,6], etc.
    sequences = re.findall(r"\[\d+(?:,\d+)*\]", s)

    # Process each found sequence
    for seq in sequences:
        # Split the sequence into individual numbers, convert to int, and take the last number
        last_num = seq.strip("[]").split(",")[-1]

        # Replace the original sequence with a format that ensures correct sorting
        s = s.replace(seq, f"[{last_num}]", 1)  # Replace only the first occurrence to maintain structure

    return s


def plot_feature_presence_and_r2(df_presence, scores_values, scores_labels):
    # Create a figure with two subplots
    fig, (ax_bar, ax_heatmap) = plt.subplots(
        2,
        gridspec_kw={"height_ratios": [3, 10]},
        sharex=True,
        figsize=(12, 5),
    )

    # Plot the heatmap of R2 values
    sns.heatmap(
        np.stack(scores_values),
        annot=True,
        fmt=".3f",
        ax=ax_bar,
        cmap="Greens",
        vmin=0,
        vmax=0.5,
        cbar=False,
        linewidths=0.2,
        linecolor="black",
        clip_on=False,
        annot_kws={"size": 8},
        yticklabels=scores_labels,
    )
    # ax_bar.set_yticks([])
    ax_bar.set_title(r"$R^2$ Value of the configuration")
    ax_bar.tick_params(left=True, bottom=False)
    ax_bar.set_yticklabels(ax_bar.get_yticklabels(), rotation=0)

    cmap_presence = mcolors.ListedColormap(["white", "black", "#851010"])
    bounds = [0, 1, 2, 3]  # Boundaries for 0 -> white, 1 -> black, 2 -> red (#851010)
    norm = mcolors.BoundaryNorm(bounds, cmap_presence.N)

    # Plot the heatmap of presences
    sns.heatmap(
        df_presence,
        cmap=cmap_presence,
        norm=norm,
        cbar=False,
        ax=ax_heatmap,
        linewidths=0.2,
        linecolor="black",
        clip_on=False,
    )
    ax_heatmap.set_ylabel("Feature name")
    ax_heatmap.set_xlabel("Simulation ID")
    ax_heatmap.tick_params(left=True, bottom=False)

    return fig, (ax_bar, ax_heatmap)


def run_postprocessing_pcmci(
    results_pcmci,
    target_column_name,
    datasets,
    destination_path,
):
    all_basin_variables = set()
    results_table_pcmci = []
    for simulation in results_pcmci:
        dataframe = datasets[simulation["dataset_name"]]
        var_names = dataframe["var_names"]
        all_basin_variables.update(var_names)

        results = simulation["results"]

        # Plot only the connections to any of the target variables
        temp_graph = results["graph"].copy()

        # Show only the connections to the target variables
        # Identify the indexes of the target variables
        # target_vars = np.where(["target" in var for var in var_names.values])[0]
        # for i in range(temp_graph.shape[0]):
        #     for j in range(temp_graph.shape[1]):
        #         # if the edge is not connected to the target variables
        #         if i not in target_vars and j not in target_vars:
        #             # remove the edge
        #             temp_graph[i, j, :] = ''
        #             temp_graph[j, i, :] = ''

        # Base arguments for tp.plot_graph
        plot_args = {
            "val_matrix": results["val_matrix"],
            "graph": temp_graph,
            "var_names": var_names,
            "link_colorbar_label": "cross-MCI",
            "node_colorbar_label": "auto-MCI",
            "show_autodependency_lags": False,
        }

        # Additional arguments to include if the independence_test is CMIknn
        if simulation["params"]["independencetest"] == "cmiknn":
            plot_args.update(
                {
                    "vmin_edges": 0.0,
                    "vmax_edges": 0.1,
                    "edge_ticks": 0.05,
                    "cmap_edges": "OrRd",
                    "vmin_nodes": 0,
                    "vmax_nodes": 0.1,
                    "node_ticks": 0.1,
                    "cmap_nodes": "OrRd",
                }
            )

        # Plot causal graph
        # target_file = os.path.join(constants.path_figures, "algorithm_results", basin_name, "pcmci", key + ".pdf")
        # if not os.path.exists(target_file):
        #     fig, ax = plt.subplots()
        #     tp.plot_graph(**plot_args, fig_ax=(fig, ax))
        #     os.makedirs(os.path.dirname(target_file), exist_ok=True)
        #     plt.savefig(target_file, bbox_inches="tight")
        #     plt.close(fig)

        # # Plot time series graph if lag > 0
        # if simulation["params"]["lag"] > 0:
        #     target_file = os.path.join(
        #         constants.path_figures, "algorithm_results", basin_name, "pcmci", key + "_timeseries.pdf"
        #     )
        #     if not os.path.exists(target_file):
        #         fig, ax = plt.subplots()
        #         tp.plot_time_series_graph(
        #             figsize=(6, 4),
        #             fig_ax=(fig, ax),
        #             val_matrix=results["val_matrix"],
        #             graph=results["graph"],
        #             var_names=var_names,
        #             link_colorbar_label="MCI",
        #         )
        #         os.makedirs(os.path.dirname(target_file), exist_ok=True)
        #         plt.savefig(target_file, bbox_inches="tight")
        #         plt.close(fig)

        # Extract the selected features
        selected_features = get_connected_variables(results["graph"], var_names)

        # Compute the R2 scores
        inputs_names_lags = {feature: [0] for feature in selected_features}
        score_r2 = (
            regression_analysis(
                inputs_names_lags=inputs_names_lags,
                target_name=target_column_name,
                df_train=dataframe["train"],
                df_test=dataframe["test"],
            )
            if len(selected_features) > 0
            else np.nan
        )

        inputs_names_lags = {feature: list(range(0, simulation["params"]["lag"] + 1)) for feature in selected_features}
        score_r2_lag = (
            regression_analysis(
                inputs_names_lags=inputs_names_lags,
                target_name=target_column_name,
                df_train=dataframe["train"],
                df_test=dataframe["test"],
            )
            if len(selected_features) > 0
            else np.nan
        )

        inputs_names_lags = {feature: list(range(0, simulation["params"]["lag"] + 1)) for feature in selected_features}
        inputs_names_lags[target_column_name] = list(range(1, simulation["params"]["lag"] + 1))
        score_r2_lag_ar = regression_analysis(
            inputs_names_lags=inputs_names_lags,
            target_name=target_column_name,
            df_train=dataframe["train"],
            df_test=dataframe["test"],
        )

        # Table of results
        results_table_pcmci.append(
            {
                "selected_features": " ".join(selected_features),
                "score_r2": score_r2,
                "score_r2_lag": score_r2_lag,
                "score_r2_lag_ar": score_r2_lag_ar,
                "dataset": simulation["dataset_name"],
                "algorithm": simulation["params"]["algorithm"],
                "independencetest": simulation["params"]["independencetest"],
                "lag": simulation["params"]["lag"],
                "execution_time": simulation["execution_time"],
            }
        )

    # Export the file to pkl
    target_file_results_details = os.path.join(destination_path, "results_details_pcmci.pkl")
    save_to_pkl_file(target_file_results_details, results_table_pcmci)

    # Feature presences heatmap
    if target_column_name in all_basin_variables:
        all_basin_variables.remove(target_column_name)
    all_basin_variables = sorted(list(all_basin_variables))
    df_presence = pd.DataFrame(index=all_basin_variables, columns=range(len(results_pcmci)))
    scores = []
    scores_lag = []
    scores_lag_ar = []

    for index, simulation in enumerate(results_pcmci):
        scores.append(simulation["score_r2"])
        scores_lag.append(simulation["score_r2_lag"])
        scores_lag_ar.append(simulation["score_r2_lag_ar"])

        # loop through the rows of the df, if the feature is in the list of selected features, put a 1
        for feature in df_presence.index:
            if feature in simulation["selected_features"]:
                df_presence.loc[feature, index] = 1
            else:
                df_presence.loc[feature, index] = 0
            if feature not in datasets[simulation["dataset_name"]]["var_names"]:
                df_presence.loc[feature, index] = 2

    df_presence = df_presence.astype(float)
    scores = np.array(scores)
    scores_lag = np.array(scores_lag)
    scores_lag_ar = np.array(scores_lag_ar)

    fig, ax = plot_feature_presence_and_r2(
        df_presence=df_presence,
        scores_values=[scores, scores_lag, scores_lag_ar],
        scores_labels=[r"$R^2$", r"$R^2$ (lag)", r"$R^2$ (lag + AR)"],
    )
    target_file_plot = os.path.join(destination_path, "algorithm_results", "pcmci", "feature_presence.pdf")
    os.makedirs(os.path.dirname(target_file_plot), exist_ok=True)
    plt.savefig(target_file_plot, bbox_inches="tight")
    plt.close(fig)

    return target_file_plot, target_file_results_details


def run_postprocessing_tefs(
    results_tefs,
    target_column_name,
    datasets,
    destination_path,
):
    all_basin_variables = set()
    results_table_te = []
    for simulation in results_tefs:
        dataset_name = simulation["dataset_name"]
        dataframe = datasets[dataset_name]
        var_names = dataframe["var_names"]
        all_basin_variables.update(var_names)

        results = simulation["results"]
        lagfeatures = simulation["params"]["lagfeatures"]
        lagtarget = simulation["params"]["lagtarget"]

        # Plot the results
        # fig, ax = plt.subplots()
        # results.plot_te_results(ax=ax)
        # target_dir = os.path.join(constants.path_figures, "algorithm_results", basin_name, "te", key + ".pdf")
        # os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        # plt.savefig(target_dir, bbox_inches="tight")
        # plt.close(fig)

        # Extract the selected features
        selected_features_names = results.select_features(simulation["params"]["threshold"])

        # get the r2 score on the test set
        inputs_names_lags = {feature: [0] for feature in selected_features_names}
        score_r2 = (
            regression_analysis(
                inputs_names_lags=inputs_names_lags,
                target_name=target_column_name,
                df_train=dataframe["train"],
                df_test=dataframe["test"],
            )
            if len(selected_features_names) > 0
            else np.nan
        )

        inputs_names_lags = {feature: lagfeatures for feature in selected_features_names}
        score_r2_lag = (
            regression_analysis(
                inputs_names_lags=inputs_names_lags,
                target_name=target_column_name,
                df_train=dataframe["train"],
                df_test=dataframe["test"],
            )
            if len(selected_features_names) > 0
            else np.nan
        )

        inputs_names_lags = {feature: lagfeatures for feature in selected_features_names}
        inputs_names_lags[target_column_name] = lagtarget
        score_r2_lag_ar = regression_analysis(
            inputs_names_lags=inputs_names_lags,
            target_name=target_column_name,  # TODO change to use the target column name given by the user
            df_train=dataframe["train"],
            df_test=dataframe["test"],
        )

        # Table of results
        results_table_te.append(
            {
                "selected_features": " ".join(selected_features_names),
                "score_r2": score_r2,
                "score_r2_lag": score_r2_lag,
                "score_r2_lag_ar": score_r2_lag_ar,
                "dataset": dataset_name,
                "lagfeatures": simulation["params"]["lagfeatures"],
                "lagtarget": simulation["params"]["lagtarget"],
                "direction": simulation["params"]["direction"],  # not putting threshold and k
                "execution_time": simulation["execution_time"],
            }
        )

    # Export the file to pkl
    target_file_results_details = os.path.join(destination_path, "results_details_te.pkl")
    save_to_pkl_file(target_file_results_details, results_table_te)

    # Feature presences heatmap
    if target_column_name in all_basin_variables:
        all_basin_variables.remove(target_column_name)
    all_basin_variables = sorted(list(all_basin_variables))
    df_presence = pd.DataFrame(index=all_basin_variables, columns=range(len(results_tefs)))
    scores = []
    scores_lag = []
    scores_lag_ar = []

    for index, simulation in enumerate(results_tefs):
        scores.append(simulation["score_r2"])
        scores_lag.append(simulation["score_r2_lag"])
        scores_lag_ar.append(simulation["score_r2_lag_ar"])

        # loop through the rows of the df, if the feature is in the list of selected features, put a 1
        for feature in df_presence.index:
            if feature in simulation["selected_features"]:
                df_presence.loc[feature, index] = 1
            else:
                df_presence.loc[feature, index] = 0
            if feature not in datasets[simulation["dataset_name"]]["var_names"]:
                df_presence.loc[feature, index] = 2

    df_presence = df_presence.astype(float)
    scores = np.array(scores)
    scores_lag = np.array(scores_lag)
    scores_lag_ar = np.array(scores_lag_ar)

    fig, ax = plot_feature_presence_and_r2(
        df_presence=df_presence,
        scores_values=[scores, scores_lag, scores_lag_ar],
        scores_labels=[r"$R^2$", r"$R^2$ (lag)", r"$R^2$ (lag + AR)"],
    )
    target_file_plot = os.path.join(destination_path, "algorithm_results", "te", "feature_presence.pdf")
    os.makedirs(os.path.dirname(target_file_plot), exist_ok=True)
    plt.savefig(target_file_plot, bbox_inches="tight")
    plt.close(fig)

    return target_file_plot, target_file_results_details


def run_postprocessing_tefs_wrapper(
    results_tefs,
    target_column_name,
    datasets,
    destination_path,
):
    results_table_tefs_wrapper = []
    target_file_train_test = os.path.join(destination_path, "tefs_as_wrapper", "wrapper.pdf")
    # target_file_cv = os.path.join(constants.path_figures, "tefs_as_wrapper_cv", f"{basename}_wrapper_cv.pdf")

    fig, ax = plt.subplots(figsize=(10, 5))

    for simulation in results_tefs:
        # --------------------- Load corresponding dataset ---------------------
        dataset_name = simulation["dataset_name"]
        dataframe = datasets[dataset_name]

        features_columns = dataframe["full"].drop(columns=[target_column_name]).columns

        # --------------------- Select features using threshold (conservative) ---------------------
        # selected_features_names_with_threshold = simulation["results"].select_features(simulation["params"]["threshold"]) # noqa
        # n_features_selected_with_threshold = len(selected_features_names_with_threshold) # noqa

        # --------------------- Compute test R2 for each number of features ---------------------
        test_r2_train_test = []
        # test_r2_cv = []
        num_total_features = len(features_columns)
        for num_features in range(0, num_total_features + 1):
            if num_features == 0:
                selected_features_names = []
            else:
                selected_features_names = simulation["results"].select_n_features(num_features)

            lagfeatures = simulation["params"]["lagfeatures"]
            lagtarget = simulation["params"]["lagtarget"]

            inputs_names_lags = {feature: lagfeatures for feature in selected_features_names}
            inputs_names_lags[target_column_name] = lagtarget

            # --- Compute the train_test version ---
            test_r2_train_test.append(
                regression_analysis(
                    inputs_names_lags=inputs_names_lags,
                    target_name=target_column_name,
                    df_train=dataframe["train"],
                    df_test=dataframe["test"],
                )
            )

            # # --- Compute the cross-validation version ---
            # # To perform a cross-validation, we need to concatenate the train and test sets
            # unified_df = pd.concat([dataframe["train"], dataframe["test"]], axis=0).reset_index(drop=True)

            # # Fixed window size
            # # n_samples = unified_df.shape[0]
            # # n_splits = 5
            # # cv_scheme = TimeSeriesSplit(
            # #     n_splits=n_splits,
            # #     max_train_size=n_samples // (n_splits + 1),
            # # )

            # # Regular KFold
            # cv_scheme = KFold(n_splits=4)  # 4 splits is about using the same test set size

            # test_r2_cv.append(
            #     regression_analysis(
            #         inputs_names_lags=inputs_names_lags,
            #         target_name=target_columns[0],
            #         df=unified_df,
            #         cv_scheme=cv_scheme,
            #     )
            # )

        test_r2_train_test = np.array(test_r2_train_test)
        # test_r2_cv = np.array(test_r2_cv)

        results_table_tefs_wrapper.append({"test_r2_train_test": test_r2_train_test})

        # Export the file to pkl
        target_file_results_details = os.path.join(destination_path, "results_details_tefs_wrapper.pkl")
        save_to_pkl_file(target_file_results_details, results_table_tefs_wrapper)

        param_str = "_".join(f"{k}{v}" for k, v in simulation["params"].items())
        ax.plot(test_r2_train_test, marker="o", label=param_str)
        maxima = np.where(test_r2_train_test == test_r2_train_test.max())[0]
        ax.plot(
            maxima,
            test_r2_train_test[maxima],
            marker="o",
            color="red",
            linestyle="None",
            label="Maximum",
            markersize=6,
        )
        # ax.plot(
        #     n_features_selected_with_threshold,
        #     test_r2_train_test[n_features_selected_with_threshold],
        #     marker="o",
        #     color="green",
        #     linestyle="None",
        #     label="TEFS (conservative)",
        #     markersize=10,
        # )

    ax.set_xlabel("Number of features")
    ax.set_ylabel("Test $R^2$")
    ax.set_title("TEFS Wrapper")
    ax.legend()
    if num_total_features < 30:
        step = 1
    elif num_total_features < 80:
        step = 5
    else:
        step = 10
    ax.set_xticks(range(0, num_total_features + 1, step))
    ax.set_xticklabels(range(0, num_total_features + 1, step))
    ax.set_ylim(-0.1, 0.55)
    ax.grid()

    os.makedirs(os.path.dirname(target_file_train_test), exist_ok=True)
    plt.savefig(target_file_train_test, bbox_inches="tight")
    plt.close(fig)

    return target_file_train_test, target_file_results_details

    # # --------------------- Plot cross-validation version ---------------------
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(test_r2_cv.mean(axis=1), marker="o", label="Cross-validation")
    # maxima = np.where(test_r2_cv.mean(axis=1) == test_r2_cv.mean(axis=1).max())[0]
    # ax.plot(maxima, test_r2_cv.mean(axis=1)[maxima], marker="o", color="red", linestyle="None", label="Maximum", markersize=10) # noqa
    # ax.plot(n_features_selected_with_threshold, test_r2_cv.mean(axis=1)[n_features_selected_with_threshold], marker="o", color="green", linestyle="None", label="TEFS (conservative)", markersize=10) # noqa

    # # plot confidence interval bands from cross-validation based on mean and standard deviation (90% confidence)
    # alpha = 0.1
    # quantile = scipy.stats.norm.ppf(1 - alpha / 2)
    # ax.fill_between(range(test_r2_cv.shape[0]), test_r2_cv.mean(axis=1) - test_r2_cv.std(axis=1) * quantile / np.sqrt(test_r2_cv.shape[1]), test_r2_cv.mean(axis=1) + test_r2_cv.std(axis=1) * quantile / np.sqrt(test_r2_cv.shape[1]), alpha=0.3) # noqa

    # ax.set_xlabel("Number of features")
    # ax.set_ylabel("Test $R^2$")

    # if simulation["params"]["threshold"] == np.inf:
    #     threshold_text = "\infty"
    # elif simulation["params"]["threshold"] == -np.inf:
    #     threshold_text = "-\infty"
    # else:
    #     threshold_text = simulation["params"]["threshold"]

    # title_text = f"TEFS on basin {basin_name.upper()} with dataset {dataset_name}\n[lagfeatures $={simulation['params']['lagfeatures']}$, lagtarget $={simulation['params']['lagtarget']}$, direction = {simulation['params']['direction']}, threshold $={threshold_text}]$" # noqa
    # ax.set_title(title_text)
    # ax.legend()
    # if num_total_features < 30:
    #     step = 1
    # elif num_total_features < 80:
    #     step = 5
    # else:
    #     step = 10
    # ax.set_xticks(range(0, num_total_features + 1, step))
    # ax.set_xticklabels(range(0, num_total_features + 1, step))
    # ax.set_ylim(-0.1, 0.55)
    # ax.grid()

    # os.makedirs(os.path.dirname(target_file_cv), exist_ok=True)
    # plt.savefig(target_file_cv, bbox_inches="tight")
    # plt.close(fig)
