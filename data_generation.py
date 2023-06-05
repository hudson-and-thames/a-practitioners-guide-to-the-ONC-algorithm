"""
Code refactored from: Detection of False Investment Strategies Using Unsupervised Learning Methods. by
Marcos Lopez de Prado and Michael Lewis. 2018.
"""

from scipy.linalg import block_diag
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


def random_block_corr(num_cols: int, num_blocks: int,
                      volatility: float, noise: float,
                      min_block_size: int = 1,
                      apply_cov_noise: bool = True, random_state: int = None):
    """
    Create a random block correlation matrix that has a clear predefined number of clusters.

    :param num_cols: (int) Number of columns in the correlation matrix.
    :param num_blocks: (int) Number of clusters in the correlation matrix, known as blocks.
    :param volatility: (float) Stdev used in constructing the root timeseries using a random walk.
    :param noise: (float) Noise added to the collection of series to create a cluster which is highly correlated.
    :param min_block_size: (int) Minimum number of trials in a cluster.
    :param apply_cov_noise: (bool) Add noise to the corr matrix else it looks unnatural.
    :param random_state: (int) Random seed number.
    :return: (DataFrame) Random block correlation matrix with predefined number of clusters.
    """
    # Set random instance
    random_instance = check_random_state(random_state)

    # Perfect block cov (no noise)
    cov = _random_block_cov(num_cols, num_blocks, min_block_size=min_block_size,
                            volatility=volatility, noise=noise, random_state=random_instance)

    # Add noise
    # This noise is so that a pair not in the cluster will have a value. Else the corr matrix has clusters and nothing
    # else. To see an example of this, plot two heatmaps with apply_noise_second on and off.
    if apply_cov_noise:
        print('apply cov noise')
        cov_noise = _random_block_cov(num_cols, num_blocks=1, min_block_size=min_block_size,
                                      volatility=volatility, noise=volatility, random_state=random_instance)
        cov += cov_noise

    # Create correlation matrix
    corr = _cov_to_corr(cov)

    # As DataFrame
    corr = pd.DataFrame(corr)
    return corr


def random_block_timeseries(num_cols, num_blocks, min_block_size, random_seed, volatility, noise, mean=0, num_obs=2500):
    random_instance = check_random_state(random_seed)

    # Get num trials per cluster
    parts = _num_trials_per_cluster(min_block_size, num_blocks, num_cols, random_instance=random_instance)

    # Create random seeds in order to create clusters
    seeds = [int(np.random.uniform(low=1, high=100000)) for i in parts]

    # Create block timeseries
    counter = 0
    store = []
    for i in parts:
        # Generate time series
        random_instance = check_random_state(seeds.pop())
        data = pd.DataFrame(_gen_correlated_timeseries(num_cols=i, num_obs=num_obs,
                                                       random_instance=random_instance,
                                                       volatility=volatility, noise=noise,
                                                       mean=mean))
        # Reset indexs
        names = []
        for i in data.columns:
            names.append(counter)
            counter += 1
        data.columns = names

        # Concat data
        store.append(data)

    # Case to DF
    data = pd.concat(store, axis=1)
    return data


def _random_block_cov(num_cols: int, num_blocks: int, min_block_size: int,
                      volatility: float, noise: float, random_state: int = None) -> pd.DataFrame:
    """
    Generate a random correlation matrix with a given number of blocks/clusters.

    :param num_cols: (int) Number of columns in covariance matrix.
    :param num_blocks: (int) Number of clusters in the covariance matrix.
    :param min_block_size: (int) Min number of trials in a cluster.
    :param volatility: (float) Standard deviation when sampling from normal distribution for clusters.
    :param random_state: (int) Random seed number.
    :return: (DataFrame) Generated covariance matrix.
    """
    # Set random instance
    random_instance = check_random_state(random_state)

    # Create the parts needed to then iterate on and create a covariance matrix with clusters.
    parts = _num_trials_per_cluster(min_block_size, num_blocks, num_cols, random_instance)

    # Create covariance clusters and append
    cov = None
    for n_cols in parts:
        # Create sub cov matrix (a cluster with some noise)
        cov_ = _sub_cov_matrix(int(max(n_cols * (n_cols + 1) / 2, 100)),
                               n_cols, volatility, noise, random_state=random_instance)

        # Append clusters to cov
        if cov is None:
            # If first iteration, update with sub cov matrix.
            cov = cov_.copy()
        else:
            # Append the sub cov matrix to the existing cov matrix.
            cov = block_diag(cov, cov_)

    # Return the full cov matrix with all the clusters included.
    return cov


def _num_trials_per_cluster(min_block_size, num_blocks, num_cols, random_instance):
    parts = random_instance.choice(range(1, num_cols - (min_block_size - 1) * num_blocks),
                                   num_blocks - 1, replace=False)
    parts.sort()
    parts = np.append(parts, num_cols - (min_block_size - 1) * num_blocks)
    parts = np.append(parts[0], np.diff(parts)) - 1 + min_block_size
    return parts


def _sub_cov_matrix(num_obs: int, num_cols: int,
                    volatility: float, noise: float,
                    random_state: int = None) -> np.array:
    """
    Create a sub correlation matrix, a cluster. Which is 1 time series that has noise around it to create a cluster.

    :param num_obs: (int) Number of observations in the time series on which a cov matrix is created.
    :param num_cols: (int) Number of desired columns in the cov matrix, not the cov matrix is square.
    :param volatility: (float) The amount of noise to add to the time series before transforming into a cov matrix.
    :param random_state: (int) Random seed number.
    :return: (np.array) Covariance matrix as a numpy array.
    """
    # Set random instance
    random_instance = check_random_state(random_state)
    # Edge case
    if num_cols == 1:
        return np.ones((1, 1))

    # Generate correlated time series
    time_series_matrix = _gen_correlated_timeseries(num_cols, num_obs, random_instance, volatility, noise)

    # Create covariance matrix
    cov = np.cov(time_series_matrix, rowvar=False)
    return cov


def _gen_correlated_timeseries(num_cols, num_obs, random_instance, volatility, noise, mean=0.0):
    # Draw random samples from a normal (Gaussian) distribution
    time_series_matrix = random_instance.normal(size=(num_obs, 1), loc=mean, scale=volatility)

    # Create num cols with exact same values as col 1 (ar0)
    time_series_matrix = np.repeat(time_series_matrix, num_cols, axis=1)

    # Add noise
    time_series_matrix += random_instance.normal(loc=0.0, scale=noise, size=time_series_matrix.shape)

    return time_series_matrix


def _cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a covariance matrix to a correlation matrix and clip the values at [-1, 1].

    :param cov: (DataFrame) Covariance matrix.
    :return: (DataFrame) Correlation matrix.
    """
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)

    # Numerical error: Cap at [-1, 1]
    corr[corr < -1] = -1
    corr[corr > 1] = 1

    return corr
