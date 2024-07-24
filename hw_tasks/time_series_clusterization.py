import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from models.clusterization import Clusterization


def main(algorithm_name, num_of_clusters):
    initial_clusters_n = num_of_clusters

    dist, _ = make_blobs(n_samples=1000, centers=initial_clusters_n, cluster_std=0.60)

    plt.scatter(dist[:, 0], dist[:, 1], cmap='viridis')
    plt.title(f'Original scatter with {num_of_clusters} centers')
    plt.show()

    model = Clusterization(algorithm_name)
    clusters = model.fit(dist, k=initial_clusters_n)

    plt.scatter(dist[:, 0], dist[:, 1], c=clusters.labels_, cmap='viridis')
    plt.title(f'Clustered scatter by {algorithm_name} algorithm')
    plt.show()


if __name__ == '__main__':
    exit = None
    while not exit:
        print('==== This script will show you how different clustering algorithms work on random distribution\n')

        cluster_count = int(input('The number of different centered clusters:\n'))

        algorithm_n = int(input("""
        Select the algorithm of clusterization you want to use:
        1 - K-means
        2 - Mean Shift
        3 - DBSCAN
        4 - Spectral Clustering\n
        """))
        algorithm_list = ["k-means", "mean_shift", "dbscan", "spectral"]
        main(algorithm_list[algorithm_n-1], cluster_count)

        exit_q = input("To exit type 'y' into field, to continue press Enter:\n")
        exit = 'y' in exit_q

print('==== Thank you for testing!')
