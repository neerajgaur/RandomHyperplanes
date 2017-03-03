""" Proof of concept for the random-cut-hyperplanes idea """
import numpy as np
from iforest import IsolationForest, IsolationTree

class RandomProjectionForest(IsolationForest):
    def __init__(self, n_estimators=10):
        super(RandomProjectionForest, self).__init__(n_estimators=n_estimators)

    def fit(self, points):
        self.estimators = []
        self.random_vectors = self.generate_random_vectors(points)

        for vector in self.random_vectors:
            positions = np.array([p - p.dot(vector) * vector for p in points])
            self.estimators.append((IsolationTree().fit(positions), vector))

        return self

    def generate_random_vectors(self, points):
        p = points.shape[1]
        vectors = []
        for _ in range(self.n_estimators):
            vector = np.random.randn(p)
            vector /= np.linalg.norm(vector)
            vectors.append(vector)

        return vectors

    def decision_function(self, points):
        mean_depths = self.get_depths(points)
        # Normalize the points
        scores = 2 ** (-mean_depths / self.average_path_length(points.shape[0]))
        return scores

    def get_depths(self, points):
        depths = []
        for tree, vector in self.estimators:
            positions = np.array([p - p.dot(vector) * vector for p in points])
            depths.append(tree.decision_function(positions))

        mean_depths = np.mean(depths, axis=0)
        return mean_depths

    def average_path_length(self, n):
        return 2 * self.harmonic_approx(n - 1) - (2 * (n - 1) / n)

    def harmonic_approx(self, n):
        return np.log(n) + 0.5772156649
