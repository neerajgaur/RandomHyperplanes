""" Proof of concept for the random-cut-hyperplanes idea """
import numpy as np
from random_hyperplanes.iforest import IsolationForest, IsolationTree


class RandomProjectionForest(IsolationForest):
    def __init__(self, n_estimators=10, method='iforest'):
        super(RandomProjectionForest, self).__init__(
            n_estimators=n_estimators,
            method=method)

    def __str__(self):
        return 'Random Projection Forest'

    def fit(self, points):
        self.estimators = []
        self.random_vectors = self.generate_random_vectors(points)

        for vector, point in self.random_vectors:
            positions = points - point
            positions = positions.dot(vector).reshape(points.shape[0], 1)
            # positions = np.array([p - p.dot(vector) * vector for p in points])
            self.estimators.append((IsolationTree(
                method=self.method).fit(positions), vector))

        return self

    def generate_random_vectors(self, points):
        p = points.shape[1]
        vectors = []
        mean = np.mean(points, axis=0)
        var  = np.var( points, axis=0)

        for _ in range(self.n_estimators):
            vector = var * np.random.randn(p) + mean
            point  = var * np.random.randn(p) + mean

            vector /= np.linalg.norm(vector)
            point  /= np.linalg.norm(point)
            vectors.append((vector - point, point))
            # vectors.append(vector)

        return vectors

    def decision_function(self, points):
        mean_depths = self.get_depths(points)
        # Normalize the points
        scores = 2 ** (-mean_depths / self.average_path_length(points.shape[0]))
        return scores

    def get_depths(self, points):
        depths = []
        for tree, vector in self.estimators:
            positions = points.dot(vector).reshape(points.shape[0], 1)
            # positions = np.sum(points * vector, axis=1)
            # positions = np.array([p - p.dot(vector) * vector for p in points])
            depths.append(tree.decision_function(positions))

        mean_depths = np.mean(depths, axis=0)
        return mean_depths

    def get_depth_per_tree(self, point):
        depths = []
        for tree, vector in self.estimators:
            positions = point.dot(vector).reshape(point.shape[0], 1)
            # positions = np.sum(points * vector, axis=1)
            # positions = np.array([p - p.dot(vector) * vector for p in points])
            depths.append(tree.decision_function(positions)[0])

        return depths

    def average_path_length(self, n):
        return 2 * self.harmonic_approx(n - 1) - (2 * (n - 1) / n)

    def harmonic_approx(self, n):
        return np.log(n) + 0.5772156649
