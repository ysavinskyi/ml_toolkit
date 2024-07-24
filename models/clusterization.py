from sklearn.cluster import KMeans, DBSCAN, MeanShift, SpectralClustering
from sklearn.preprocessing import StandardScaler


class Clusterization:
    """
    Handles the clusterization of data
    """
    def __init__(self, algorithm):
        """
        Initializes the class
        """
        self._algorithm = algorithm.lower()

    def fit(self, data, k=None):
        """
        Applies algorithm to a data
        """
        result = None
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        if k is None and self._algorithm in ['k-means', 'spectral']:
            raise Exception('Specify the "k" attribute for method')

        if 'k-means' in self._algorithm:
            result = self._k_means(data, k)

        elif 'dbscan' in self._algorithm:
            result = self._dbscan(data)

        elif 'mean_shift' in self._algorithm:
            result = self._mean_shift(data)

        elif 'spectral' in self._algorithm:
            result = self._spectral(data, k)
        else:
            raise Exception('Valid algorithms are "k-means", "mean_shift", "dbscan", "spectral"')

        return result

    @staticmethod
    def _k_means(data, n_clusters):
        """
        Implementation of K-Means algorithm
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=11)
        kmeans.fit(data)

        return kmeans

    @staticmethod
    def _dbscan(data, epsilon=0.15, min_samples=10):
        """
        Implementation of DBSCAN algorithm
        """
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        dbscan.fit_predict(data)

        return dbscan

    @staticmethod
    def _mean_shift(data, bandwidth=0.5):
        """
        Implementation of Mean Shift algorithm
        """
        mean_shift = MeanShift(bandwidth=bandwidth)
        mean_shift.fit_predict(data)

        return mean_shift

    @staticmethod
    def _spectral(data, n_clusters):
        """
        Implementation of Spectral Clusterization algorithm
        """
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
        spectral.fit_predict(data)

        return spectral
