from typing import Dict, Tuple

import pickle
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

import matplotlib.pyplot as plt
import seaborn as sns


class ONC_range_change:
    """
    Optimal Number of Clusters (ONC) algorithm described:
    Marcos Lopez de Prado, Michael J. Lewis, Detection of False Investment Strategies Using Unsupervised
    Learning Methods, 2019.
    """
    def __init__(self, correlation_matrix: pd.DataFrame, repeat: int = 10, max_scan: int = None):
        """
        Note: This algorithm is very slow. For 1000x1000 matrix, 30 repeat, On my 8 core - 16 thread -
        i9 CPU, it takes 4.8 hours with multi-threading.

        :param correlation_matrix: Corr matrix of the set of returns.
        :param repeat: How many times to repeat the algorithm. The higher, the greater the chance of a stable solution.
        """
        self.correlation = correlation_matrix
        self.repeat = repeat

        self.max_scan = max_scan

        # Private results
        self.__block_correlation = None
        self.__optimal_clusters = None
        self.__silh_scores = None

    def fit(self):
        """
        Fit the ONC algorithm. This part of the computation takes a long time to run. Rather load results from disk if
        available.
        """
        # This is a recursive function.
        corr_matrix, opt_clusters, silh_scores = self._cluster_kmeans_top(self.correlation, self.repeat, self.max_scan)

        # Save to model object.
        self.__block_correlation = corr_matrix
        self.__optimal_clusters = opt_clusters
        self.__silh_scores = silh_scores

    def get_block_correlation(self) -> pd.DataFrame:
        """
        Return the sorted, block correlation matrix.

        :return: (pd.DataFrame) Sorted block correlation matrix.
        """
        return self.__block_correlation

    def get_optimal_clusters(self) -> Dict:
        """
        Return a dictionary of cluster groups and their constituents.

        :return: (Dict) of cluster groups and constituents.
        """
        return self.__optimal_clusters

    def get_silh_scores(self) -> pd.Series:
        """
        Return pd.Series of the silhouette scores for each trial. These scores measure how good an element
        is clustered.

        :return: (pd.Series) Silhouette scores for each trial
        """
        return self.__silh_scores

    def plot_block_correlation(self, save: bool = False, pre_fix: str = '0'):
        """
        Plots a block correlation matrix using a heatmap and a red/blue diverging colour pallet.
        Method can also save the plot to disk as a png file.

        :param save: (bool) If True, saves png to disk with prefix in the file name.
        :param pre_fix: (str) prefix to add to file name.
        """
        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        sns.heatmap(self.__block_correlation, cmap=cmap,
                    annot=False, cbar_kws={'label': 'Correlation Coefficient'},
                    vmin=-1, vmax=1)

        # Save Png image to disk.
        if save:
            plt.savefig('{}_block_correlation.png'.format(pre_fix))

    def save_results(self, pre_fix: str = '0'):
        """
        Save the block correlation matrix, the optimal clusters, and the silh_scores to disk. These files can be used
        to load results into new ONC model objects.

        :param pre_fix: (str) prefix to add to file name.
        """
        # Save block correlation as CSV
        self.__block_correlation.to_csv('{}_block_correlation.csv'.format(pre_fix))
        # Save heatmap
        self.plot_block_correlation(save=True, pre_fix=pre_fix)

        # Pickle optimal cluster dict
        with open('{}_opt_clusters.pkl'.format(pre_fix), 'wb') as f:
            pickle.dump(self.__optimal_clusters, f)

        # Save silhouette scores as CSV
        self.__silh_scores.to_csv('{}_silh.csv'.format(pre_fix))#

    def load_results(self, pre_fix: str):
        """
        Loads existing saved results, into this model object. This allows users to save time when presenting results.

        :param pre_fix: (str) prefix used in existing file name.
        """
        # Load block correlation as CSV
        self.__block_correlation = pd.read_csv('{}_block_correlation.csv'.format(pre_fix), index_col=0)

        # Load optimal cluster dict
        with open('{}_opt_clusters.pkl'.format(pre_fix), 'rb') as f:
            self.__optimal_clusters = pickle.load(f)

        # Load silhouette scores as CSV
        self.__silh_scores = pd.read_csv('{}_silh.csv'.format(pre_fix), index_col=0)

    # Private Methods
    # -----------------------------------------------------------------------------------------------------------------

    def _cluster_kmeans_top(self, corr_mat: pd.DataFrame, repeat: int = 10, max_scan: int = None) -> Tuple[pd.DataFrame, Dict, pd.Series]:
        """
        This is the starting point for the ONC algorithm, and it is called recursively.

        Improves the initial clustering by leaving clusters with high scores unchanged and modifying clusters with
        below average scores.

        :param corr_mat: Correlation matrix of the set of returns.
        :param repeat: How many times to repeat the algorithm, to ensure the very best results.
        :return: (Tuple) Return 3 needed elements (correlation matrix, optimized clusters, silh scores).
        """

        # Initial clustering step
        max_num_clusters = corr_mat.shape[1]-1

        print('Max_scan:', max_scan)
        if max_scan:
            corr1, clusters, silh = self._kmeans_base(corr_mat, max_scan, repeat)
        elif max_num_clusters < max_scan:
            corr1, clusters, silh = self._kmeans_base(corr_mat, max_num_clusters, repeat)
        else:
            raise ValueError('Logic broken')

        # Get cluster quality scores + redo low quality clusters
        cluster_quality = {i: float('Inf') if np.std(silh[clusters[i]]) == 0 else np.mean(silh[clusters[i]]) /
                              np.std(silh[clusters[i]]) for i in clusters.keys()}
        avg_quality = np.mean(list(cluster_quality.values()))
        redo_clusters = [i for i in cluster_quality.keys() if cluster_quality[i] < avg_quality]

        if len(redo_clusters) <= 2:
            # If 2 or less clusters have a quality rating less than the average then stop
            return corr1, clusters, silh
        else:
            print('improving now')
            keys_redo = []
            for i in redo_clusters:
                keys_redo.extend(clusters[i])

            corr_tmp = corr_mat.loc[keys_redo, keys_redo]
            mean_redo_tstat = np.mean([cluster_quality[i] for i in redo_clusters])
            # Recursive call
            _, top_clusters, _ = self._cluster_kmeans_top(corr_tmp, repeat=repeat, max_scan=max_scan)

            # Make new clusters (improved)
            corr_new, clusters_new, silh_new = self._improve_clusters(corr_mat,
                                                                      {i: clusters[i] for i in clusters.keys() if
                                                                       i not in redo_clusters},
                                                                      top_clusters)

            # ---------------------------------------------------------------------------------------------------------
            # Debugging
            new_tstat = []
            for i in clusters_new:
                set_ = silh_new[clusters_new[i]]
                mean = np.mean(set_)
                std = np.std(set_)
                if std == 0:
                    print(('There is an observation which is an outlier: Index {}'.format(set_.index.values)))
                    # raise Warning('There is an observation which is an outlier: Index {}'.format(set_.index.values))
                    std = 1000000

                score = mean / std
                new_tstat.append(score)

            # Compute mean
            new_tstat_mean = np.mean(new_tstat)

            return self._check_improve_clusters(new_tstat_mean, mean_redo_tstat,
                                                (corr1, clusters, silh),
                                                (corr_new, clusters_new, silh_new))

    @staticmethod
    def _kmeans_base(corr: pd.DataFrame, max_num_clusters: int,
                     repeat: int) -> Tuple[pd.DataFrame, Dict, pd.Series]:
        """
        Initial clustering step using KMeans.

        :param corr: (pd.DataFrame) Correlation matrix of the set of returns.
        :param max_num_clusters: (int)
        :param repeat: (int) How many times to repeat the algorithm, to ensure the very best results.
        :return: (Tuple) (ordered correlation matrix, clusters, silhouette scores)
        """

        # Distance matrix
        distance = np.sqrt(((1 - corr.fillna(0))/2))
        silhouette = pd.Series(dtype='float64')

        # Get optimal num clusters.
        # Repeat n times.
        for _ in range(repeat):
            # Cluster in sizes from 2 to max clusters
            for num_clusters in range(3, max_num_clusters+1):
                kmeans_ = KMeans(n_clusters=num_clusters, n_init=1)
                kmeans_ = kmeans_.fit(distance)
                silh_ = silhouette_samples(distance, kmeans_.labels_)
                stat = (silh_.mean()/silh_.std(), silhouette.mean()/silhouette.std())

                # Update silhouette and kmeans if quality is better than previous.
                if np.isnan(stat[1]) or stat[0] > stat[1]:
                    silhouette = silh_
                    kmeans = kmeans_

        # Number of clusters equals to length(kmeans labels)
        new_idx = np.argsort(kmeans.labels_)
        print(len(np.unique(kmeans.labels_)))

        # Reorder rows
        corr1 = corr.iloc[new_idx]
        # Reorder columns
        corr1 = corr1.iloc[:, new_idx]

        # Cluster members
        clusters = {i: corr.columns[np.where(kmeans.labels_ == i)[0]].tolist()
                    for i in np.unique(kmeans.labels_)}
        silhouette = pd.Series(silhouette, index=distance.index)

        return corr1, clusters, silhouette

    @staticmethod
    def _improve_clusters(corr_mat: pd.DataFrame, clusters: Dict,
                          top_clusters: Dict) -> Tuple[pd.DataFrame, Dict, pd.Series]:
        """
        Improve number clusters using silh scores.

        :param corr_mat: (pd.DataFrame) Correlation matrix.
        :param clusters: (dict) Clusters elements.
        :param top_clusters: (dict) Improved clusters elements.
        :return: (tuple) [ordered correlation matrix, clusters, silh scores].
        """

        clusters_new, new_idx = {}, []
        for i in clusters.keys():
            clusters_new[len(clusters_new.keys())] = list(clusters[i])

        for i in top_clusters.keys():
            clusters_new[len(clusters_new.keys())] = list(top_clusters[i])

        new_idx = [item for sublist in list(clusters_new.values()) for item in sublist]
        corr_new = corr_mat.loc[new_idx, new_idx]

        dist = ((1 - corr_mat.fillna(0)) / 2.0) ** 0.5

        kmeans_labels = np.zeros(len(corr_mat.columns))
        for i in clusters_new:
            idxs = [corr_mat.index.get_loc(k) for k in clusters_new[i]]
            kmeans_labels[idxs] = i

        silh_scores_new = pd.Series(silhouette_samples(dist, kmeans_labels), index=corr_mat.index)

        return corr_new, clusters_new, silh_scores_new

    @staticmethod
    def _check_improve_clusters(new_tstat_mean: float, mean_redo_tstat: float, old_cluster: Tuple,
                                new_cluster: Tuple) -> Tuple[pd.DataFrame, Dict, pd.Series]:
        """
        Checks cluster improvement condition based on t-statistic and returns the best set/cluster.

        :param new_tstat_mean: (float) T-statistics.
        :param mean_redo_tstat: (float) Average t-statistcs for cluster improvement.
        :param old_cluster: (tuple) Old cluster correlation matrix, optimized clusters, silh scores.
        :param new_cluster: (tuple) New cluster correlation matrix, optimized clusters, silh scores.
        :return: (tuple) The best set given t-stats, between new and old set. A set is correlation matrix,
         optimized clusters, silh scores
        """

        if new_tstat_mean <= mean_redo_tstat:
            print('old clustering won')
            return old_cluster

        return new_cluster
