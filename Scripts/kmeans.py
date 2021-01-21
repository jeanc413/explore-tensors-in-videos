import numpy as np
import tensorly as tl


def tensor_distance(x1, x2):
    # Tensor distance norm
    return tl.norm(x1-x2)


class KMeans:
    def __init__(self, tensor_list, distance=tensor_distance, k=5, max_iterations=100):
        # Init tensors
        self.tensor_list = tensor_list
        self.n_samples = len(tensor_list)
        self.n_features = tl.shape(tensor_list[0].core)

        # Init Clustering parameters
        self.k = k
        self.max_iterations = max_iterations
        self.distance = distance

        # list of sample indices inside each cluster
        self.clusters = [[] for _ in range(self.k)]

        # mean feature tensor for each cluster
        self.centroids = []

    def predict(self, centroids=None):
        # initialize centroids
        if centroids is None:
            self.centroids = np.random.choice(self.n_samples, self.k, replace=False)
            self.centroids = [self.tensor_list[i].core for i in self.centroids]

        # Optimization
        for _ in range(self.max_iterations):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)

            # update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # check for convergence
            if self._is_converged(centroids_old, self.centroids):
                break
        # return cluster labels
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # Puts together on an array the assigned cluster for each tensor
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Init temporal empty cluster
        clusters = [[] for _ in range(self.k)]
        # Checks for each tensor the closest tensor centroid and returns this as clusters list
        for idx, sample in enumerate(self.tensor_list):
            centroids_idx = self._closest_centroid(sample, centroids, self.distance)
            clusters[centroids_idx].append(idx)
        return clusters

    @staticmethod
    def _closest_centroid(sample, centroids, distance):
        # Computes the distance from the current sample to all existent centroids
        distances = [distance(sample.core, point) for point in centroids]
        # Checks the closest centroid and returns it's index
        closest_idx = np.argmin(distances)
        return int(closest_idx)

    def _get_centroids(self, clusters):
        # Create empty centroid(Tensor shaped) list
        centroids = [tl.zeros(self.n_features) for _ in range(self.k)]
        # Computes(Mean) the centroid per cluster and returns it
        for cluster_idx, cluster in enumerate(clusters):
            cores = [self.tensor_list[i].core for i in cluster]
            cluster_mean = tl.mean(cores, axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    @staticmethod
    def euclidian_distance(x, y):
        # Simple vector distance norm
        return np.linalg.norm(x-y)

    def _is_converged(self, centroids_old, centroids):
        # Verify if there's no more improvement for the current iteration and returns True as converging criteria
        distances = [self.euclidian_distance(centroids_old[i], centroids[i]) for i in range(self.k)]
        return sum(distances) == 0
