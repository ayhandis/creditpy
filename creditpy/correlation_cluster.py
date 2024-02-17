import pandas as pd
import numpy as np


def correlation_cluster(data, clustering_data, clusters, target_column):
    """
    Average Correlations of Clusters

    This function calculates correlations for clusters using the output of a clustering function.

    Parameters:
    data : pandas DataFrame
        A raw data set.
    clustering_data : pandas DataFrame
        Output of the clustering function.
    clusters : str
        The column name of "clusters" in the clustering function output.
    target_column : str
        The name of the target column to exclude from correlation calculations.

    Returns:
    pandas DataFrame
        A DataFrame containing average correlations for each cluster.

    Examples:
    correlation_cluster(my_data, clustering_output, "Groups", "target_column")
    """
    cluster_values = sorted(clustering_data[clusters].unique())
    cluster_correlations = []

    for cluster in cluster_values:
        # Filter data based on cluster
        cluster_indices = clustering_data[clustering_data[clusters] == cluster].index
        cluster_data = data.loc[cluster_indices]

        # Drop target column
        cluster_data = cluster_data.drop(columns=[target_column])

        # Calculate correlations
        correlations = cluster_data.corr().values
        np.fill_diagonal(correlations, np.nan)  # Exclude diagonal values
        cluster_correlations.append(np.nanmean(correlations))

    cor_summary = pd.DataFrame({
        'Clusters': cluster_values,
        'Correlation': cluster_correlations
    })

    return cor_summary
