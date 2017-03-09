""" Proof of concept for the random-cut-hyperplanes idea """
import numpy as np
from scipy.stats import scoreatpercentile


class RandomProjectionForestOld(object):
    def __init__(self, n_estimators=10, method='iforest'):
        self.n_estimators = n_estimators

    def fit(self, points):
        self.trees = []
        for i in range(self.n_estimators):
            self.trees.append(RandomProjectionTree().build(points))

        return self

    def decision_function(self, points):
        depths = np.array([tree.decision_function(points) for tree in self.trees])
        mean_depths = np.mean(depths, axis=0)
        # Normalize the points
        scores = 2**-(mean_depths / self.average_path_length(points.shape[0]))
        return scores

    def get_depths(self, points):
        depths = np.array([tree.decision_function(points) for tree in self.trees])
        mean_depths = np.mean(depths, axis=0)
        return mean_depths

    def average_path_length(self, n):
        return 2 * self.harmonic_approx(n - 1) - (2 * (n - 1) / n)

    def harmonic_approx(self, n):
        return np.log(n) + 0.5772156649


    def predict(self, X, score_at=97.5):
        scores = self.decision_function(X)
        preds = np.zeros(shape=(scores.shape[0]))
        threshold = scoreatpercentile(scores, score_at)
        preds[np.where(scores >= threshold)] = 1
        return preds


class RandomProjectionTree(object):
    def __init__(self, depth=1):
        self.child_left = self.child_right = None
        self.depth = depth

    def __str__(self):
        return f"<{self.__class__.__name__} num_points:{self.num_points}>"

    @property
    def is_leaf(self):
        return self.child_left == None and self.child_right == None

    def build(self, points):
        self.num_points = points.shape[0]
        plane, idx_left, idx_right = self.get_split(points)
        if plane is None or self.depth >= 100:
            return self

        self.child_left = RandomProjectionTree(depth=self.depth + 1).build(points[idx_left, :])
        self.child_right = RandomProjectionTree(depth=self.depth + 1).build(points[idx_right, :])
        return self

    def decision_function(self, points):
        return np.array(self.get_depths(points))

    def get_depths(self, points):
        depths = np.zeros(points.shape[0])
        if self.is_leaf:
            depths = np.array([self.depth])

        else:
            _, idx_left, idx_right = self.split_points(points,
                                                       self.splitting_plane,
                                                       self.split_point)
            depths[idx_left] = self.child_left.get_depths(points[idx_left, :])
            depths[idx_right] = self.child_right.get_depths(points[idx_right, :])

        return depths

    def split_points(self, points, splitting_plane=None, split_point=None):
        if splitting_plane is None:
            splitting_plane = self.generate_random_vector(points)
            self.splitting_plane = splitting_plane

        positions = points.dot(splitting_plane)
        if split_point is None:
            split_point = np.random.uniform(np.min(positions), np.max(positions))
            self.split_point = split_point

        idx_left = np.where(positions < split_point)[0]
        idx_right = np.where(positions >= split_point)[0]
        return (splitting_plane, idx_left, idx_right)

    def get_split(self, points):
        if self.num_points < 2:
            return (None, None, None)

        return self.split_points(points)

    def generate_random_vector(self, points):
        p = points.shape[1]
        vector = np.random.randn(p)
        vector /= np.linalg.norm(vector)
        return vector
