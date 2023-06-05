from typing import Tuple, Dict

from scipy.stats import norm
from scipy import stats

import numpy as np
import pandas as pd


def expected_max_sr(num_trials: int, std_sr: float, mean_sr: float = 0) -> float:
    """
    Enhanced Snippet 8.1, page 110 of Lopez de Prado, M, 2020. Machine learning
    for asset managers. Cambridge University Press.
    Based on the False Strategy Theorem, we can compute the Expected Maximum Sharpe Ratio.
    This value allows us to compute the probability of a false investment strategy using the
    Deflated Sharpe Ratio.
    The E[Max_SR] is a function of the number of trials run and the variance of the set of Sharpe Ratios.
    Note: The mean SR and std SR are not non-annualised figures.
    :param num_trials: (int) The number of INDEPENDENT trials run when backtesting. Use ONC clustering to solve this.
    :param mean_sr: (float) Usually we estimate this value to be 0, this value can be adjusted to change
     the null hypothesis.
    :param std_sr: (float) Standard deviation of the set of Sharpe Ratios.
    :return: (float) The Maximum Expected Sharpe Ratio
    """

    # Error handling
    if not std_sr or std_sr == 0:
        raise ValueError('The standard deviation of the Sharpe Ratio '
                         'may not be 0 or a null value.')

    # Constants
    em_constant = np.euler_gamma
    e = np.e

    # False Strategy Theorem
    # Portion related to number of trials
    trials_portion = (1 - em_constant) * norm.ppf(1 - 1 / num_trials) + em_constant \
                     * norm.ppf(1 - (num_trials * e) ** -1)
    # Incorporating the mean and variance of the SR's.
    exp_max_sr = mean_sr + std_sr * trials_portion

    return exp_max_sr


def z_stat_psr(sharpe_ratio: float, num_observations: int, threshold_sharpe_ratio: float = 0,
               skew: float = 0, kurt: float = 3) -> float:
    """
    This is the z-stat that is used in the Probabilistic Sharpe Ratio.
    Note: the sharpe ratio is non-annualised!
    :param sharpe_ratio: (float) Non-annualised sharpe ratio.
    :param num_observations: (int) Number of observations in the time series.
    :param threshold_sharpe_ratio: (float)
    :param skew: (float) Skewness of the returns
    :param kurt: (float) Kurtosis of the returns
    :return: (float) z-statistic used in probabilistic sharpe ratio calculation
    """
    numerator = (sharpe_ratio - threshold_sharpe_ratio) * np.sqrt(num_observations - 1)
    denominator = np.sqrt(1 - skew * sharpe_ratio + (kurt - 1) / 4 * sharpe_ratio ** 2)
    z_stat = numerator / denominator
    return z_stat


def probabilistic_sharpe_ratio(sharpe_ratio: float, num_observations: int, threshold_sharpe_ratio: float = 0,
                               skew: float = 0, kurtosis: float = 3) -> float:
    """
    Probability that the true SR exceeds a given user-defined threshold SR. Under the general assumption that
    returns are stationary and ergodic (not necessarily IID Normal). Used directly to determine the probability
    that a discovery made after a single trial is a false positive.
    PSR increases with greater SR, or longer track records, or positively skewed returns, but is decreases with
    fatter tails (kurtosis).
    Note: the sharpe ratio is non-annualised!
    Reference: Lopez de Prado and Lewis, 2018.
    :param sharpe_ratio: (float) Non-annualised sharpe ratio.
    :param num_observations: (int) Number of observations in the time series.
    :param threshold_sharpe_ratio: (float) User defined benchmark level sharpe ratio. Non-annialised.
    :param skew: (float) Skewness of the returns
    :param kurtosis: (float) Kurtosis of the returns
    :return: (float) probability that the true Sharpe ratio exceeds a given threshold sharpe ratio.
    """
    psr = stats.norm.cdf(z_stat_psr(sharpe_ratio, num_observations, threshold_sharpe_ratio, skew, kurtosis))
    return psr


def deflated_sharpe_ratio(sharpe_ratio: float, num_observations: int, skew: float, kurtosis: float,
                          num_independent_trials: int, std_sr: float) -> float:
    """
    A statistical test that the true sharpe ratio is greater than 0. The null hypothesis is that it is not
    greater than 0.
    The output is a value between 0-1, and at a 95% confidence level, we expect the DFS output to be > 0.95.
    This is based on the False Strategy Theorem, and the DFS is expanded on in the works of Lopez de Prado and
    Lewis (2019).
    :param sharpe_ratio: (float) non-annualised sharpe ratio
    :param num_observations: (int) Number of observations in the time series of returns.
    :param skew: (float) Skew of the returns.
    :param kurtosis: (float) Kurtosis of the returns.
    :param num_independent_trials: (int) This is E[K], which is determined by first applying the ONC algorithm.
    :param std_sr: (float) The standard deviation of the IVPortfolio's sharpe ratio, for each cluster.
    :return: (float) The probability that a strategy is not a false positive.
    """
    # Probability that the true SR exceeds the max expected sharpe ratio.
    # Returns a confidence level, example: to reject the null hyp, H_0: SR=0, with a 5% significance level, the DSR
    # must be greater than 0.95.

    max_sr = expected_max_sr(num_trials=num_independent_trials, std_sr=std_sr, mean_sr=0)
    dsr = probabilistic_sharpe_ratio(sharpe_ratio, num_observations, threshold_sharpe_ratio=max_sr,
                                     skew=skew, kurtosis=kurtosis)
    return dsr


def familywise_type1_err(psr_z_statistic: float, num_clusters: int = 1) -> Tuple[float, float]:
    """
    Compute the probability of Familywise type I errors. alpha_K in equations. As well as the False positive rate.
    :param psr_z_statistic: (float) The z-statistic used to compute the probabilistic sharpe ratio.
    :param num_clusters: (int) The number of clusters derived from the ONC algorithm, or the num of independent trials.
    :return: (Tuple[float, float]) The false positive rate (1 test) and the corrected false positive rate (many tests).
    """
    # Type I error for a single test
    prob_false_positive = stats.norm.cdf(-psr_z_statistic)  # Also known as alpha

    # Multi-testing correction
    # Probability of Familywise type I errors / FWER
    # Probability that at least one of the positives is false
    fwer = 1 - (1 - prob_false_positive) ** num_clusters  # family false positive probability

    return fwer, prob_false_positive


def sidaks_correction(alpha_k, k):
    """
    Compute the type I error probability under multiple testing, αK.
    For a FWER αK, Sidak’s correction gives us a single-trial significance level.
    """
    alpha = 1 - (1 - alpha_k)**(1/k)
    return alpha


def weights_min_var(cov: pd.DataFrame) -> np.array:
    """
    Get the weights of the minimum variance portfolio. This code is from
    Lopez de Prado and Lewis (2018).
    :param cov: (pd.DataFrame) Covariance matrix of the clusters returns.
    :return: (np.array) Weights of the inverse variance portfolio.
    """
    # Compute the minimum-variance portfolio
    ivp = 1/np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def cluster_statistics(clusters: Dict, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Create a table of useful statistics, based on each cluster, which can be used to determine the
    likelihood of a false investment strategy.
    This table is based on Lopez de Prado (2019).
    :param clusters: (Dict) Key: Cluster number, Value: all the trials that are in that cluster.
    :param returns: (pd.DataFrame) DataFrame of all the strategies/trials - returns.
    :return: (pd.DataFrame) Table of cluster statistics.
    """
    exp_k = len(clusters)

    # Cluster statistics
    # Get cluster SR, S_rets, and weights_min_var
    skew, kurt, observations = {}, {}, {}
    cluster_sr, cluster_ann_sr, cluster_rets, cluster_weights, strat_count = {}, {}, {}, {}, {}
    for key, value in clusters.items():
        # Subset cluster by constituents
        strats_rets = returns.loc[:, value]

        # Compute IVP weights, create S_rets, S_sharpe_ratio
        cov = strats_rets.cov()
        weights = weights_min_var(cov)
        rets_aggregate_strategy = (np.array(weights) * strats_rets).sum(axis=1)
        sr = rets_aggregate_strategy.mean() / rets_aggregate_strategy.std()

        # Time Series Metrics
        skew[key] = rets_aggregate_strategy.skew()
        kurt[key] = rets_aggregate_strategy.kurtosis()
        observations[key] = rets_aggregate_strategy.shape[0]

        # Cluster Metrics
        strat_count[key] = int(len(value))
        cluster_weights[key] = np.array(weights)
        cluster_rets[key] = rets_aggregate_strategy
        cluster_sr[key] = sr
        cluster_ann_sr[key] = sr * np.sqrt(252)

    # Std of SR
    df_cluster_sr = pd.DataFrame(cluster_sr, index=['sr'])  # Convert to DF
    # Don't need to add the freq as all the strategies, they all have the same number of observations.
    std_sr = df_cluster_sr.std(axis=1)[0]

    # Create Table
    stats = [strat_count, cluster_ann_sr, cluster_sr, skew, kurt, observations]
    stat_names = ['Strat Count', 'aSR', 'SR', 'Skew', 'Kurt', 'T']
    table_statistics = pd.DataFrame(stats, index=stat_names)

    # Add sqrt(V[SR_k])
    table_statistics.loc['sqrt(V[SR_k])'] = std_sr
    # Add E[max SR]
    table_statistics.loc['E[max SR_k]'] = expected_max_sr(num_trials=exp_k, std_sr=std_sr, mean_sr=0)

    # Get DFS for each cluster
    dfs_store = {}
    for i in range(0, exp_k):
        stat = table_statistics[i]
        dfs = deflated_sharpe_ratio(sharpe_ratio=stat['SR'], num_observations=stat['T'],
                                    skew=stat['Skew'], kurtosis=stat['Kurt'],
                                    num_independent_trials=exp_k,
                                    std_sr=stat['sqrt(V[SR_k])'])
        dfs_store[i] = dfs

    # Add to table
    table_statistics.loc['DFS'] = dfs_store

    # Transpose table for visualisation
    table_statistics = table_statistics.T

    return table_statistics