import numpy as np
import pandas as pd
from tigramite import data_processing as pp


def get_connected_variables(graph: np.ndarray, var_names: list[str]) -> list[str]:
    """
    Get the variables connected to the target in the graph.
    The target is assumed to be the last variable.
    The connection is considered of any type: from, to, or undefined.

    :param graph: the graph of the PCMCI algorithm, i.e. what's returned by PCMCI.run_pcmci(), array of shape [N, N, tau_max+1]
    :param var_names: the names of the variables
    """

    assert len(graph.shape) == 3, "The graph must be a 3D array"
    assert graph.shape[0] == graph.shape[1], "The graph must be square"

    # Inspecting the results object
    # results['p_matrix']
    # results['val_matrix']
    # results['graph']
    # results['graph'][-1] # last element (target connections, target is always last)

    # in the array replace the empty with 0, otherwise it's 1 (there's a connection)
    np.where(graph[-1] == "", 0, 1)

    # transpose it and add it to a dataframe with the variable names
    # each row is a lag (when lag is 1, I have a row for lag 0 and one for lag 1)
    target_connections = pd.DataFrame(np.where(graph[-1] == "", 0, 1).T, columns=var_names)

    # ignore autocorrelation (a connection from a variable to itself)
    target_connections = target_connections.drop(var_names[-1], axis=1)

    # drop all columns with only zeros (no connection) and keep the names
    connected_variables = list(target_connections.loc[:, (target_connections != 0).any(axis=0)].columns.values)

    return connected_variables


def initialize_tigramite_df(df: pd.DataFrame):
    """
    Initialize a tigramite dataframe from a pandas dataframe

    :param df: pandas dataframe
    :return: tigramite dataframe and variable names tuple
    """

    var_names = df.columns

    dataframe = pp.DataFrame(df.values, datatime={0: np.arange(len(df))}, var_names=var_names)

    return dataframe, var_names
