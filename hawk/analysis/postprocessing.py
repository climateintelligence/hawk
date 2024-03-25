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
    datasets,
    destination_path,
):
    all_basin_variables = set()
    results_table_pcmci = []
    for simulation in results_pcmci:
        dataframe = datasets[simulation["dataset_name"]]
        var_names = dataframe["var_names"]
        all_basin_variables.update(var_names.values)

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
                target_name="target",
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
                target_name="target",
                df_train=dataframe["train"],
                df_test=dataframe["test"],
            )
            if len(selected_features) > 0
            else np.nan
        )

        inputs_names_lags = {feature: list(range(0, simulation["params"]["lag"] + 1)) for feature in selected_features}
        inputs_names_lags["target"] = list(range(1, simulation["params"]["lag"] + 1))
        score_r2_lag_ar = regression_analysis(
            inputs_names_lags=inputs_names_lags,
            target_name="target",
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
    if "target" in all_basin_variables:
        all_basin_variables.remove("target")
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
                target_name="target",
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
                target_name="target",
                df_train=dataframe["train"],
                df_test=dataframe["test"],
            )
            if len(selected_features_names) > 0
            else np.nan
        )

        inputs_names_lags = {feature: lagfeatures for feature in selected_features_names}
        inputs_names_lags["target"] = lagtarget
        score_r2_lag_ar = regression_analysis(
            inputs_names_lags=inputs_names_lags,
            target_name="target",
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
    if "target" in all_basin_variables:
        all_basin_variables.remove("target")
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
