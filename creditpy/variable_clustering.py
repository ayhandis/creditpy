import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gap_statistic import OptimalK


def variable_clustering(data, default_flag, cluster_number="optimal"):
    """
    Perform variable clustering for a given dataset.

    This function clusters variables in the dataset based on the specified default flag.
    The number of clusters can be determined optimally using the gap statistic method or manually.

    Parameters:
    data (DataFrame): The dataset to be clustered.
    default_flag (str): The name of the default flag variable in the dataset.
    cluster_number (int or str): The number of clusters to generate. If "optimal" is selected, the optimal number
                                  of clusters is determined automatically using the gap statistic method (default is "optimal").

    Returns:
    DataFrame: A DataFrame containing the variable clusters.

    Example:
    >>> import pandas as pd
    >>> # Assume data is defined
    >>> credit_data = pd.read_csv('credit_data.csv')  # Assuming 'credit_data.csv' contains the dataset
    >>> variable_clusters = variable_clustering(credit_data, "default_flag", cluster_number="optimal")
    >>> print(variable_clusters)
    """

    def optimal_cluster(data):
        opt_k = OptimalK(parallel_backend='multiprocessing')
        n_clusters = opt_k(data.values, cluster_array=np.arange(1, data.shape[1]))
        return n_clusters.optimal_k

    def given_cluster(data, cluster_number):
        kmeans = KMeans(n_clusters=cluster_number, init='k-means++', max_iter=1000, n_init=1000)
        kmeans.fit(data.values.T)
        return kmeans.labels_

    # Remove default_flag column from data
    data1 = data.drop(columns=[default_flag])

    if cluster_number == "optimal":
        cluster_number = optimal_cluster(data1)
        clusters = given_cluster(data1, cluster_number)
    else:
        clusters = given_cluster(data1, cluster_number)

    variable_clusters = pd.DataFrame({'Group': clusters, 'Variable': data1.columns})
    return variable_clusters


