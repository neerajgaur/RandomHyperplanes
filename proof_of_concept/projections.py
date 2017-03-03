""" Proof of concept for the random-cut-hyperplanes idea """
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.ensemble.iforest import _average_path_length


class RandomProjectionForest(object):
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.projections = list()  # List of vectors to project points onto

    def generate_random_vectors(self, points):
        features = points.shape[1]
        for _ in range(self.n_estimators):
            vector = np.random.randn(features)
            vector /= np.linalg.norm(vector)
            yield vector

    def point_positions(self, points):
        for projection in self.projections:
            res = list()
            for point in points:
                # min_ = np.min(points, axis=0)
                # max_ = np.max(points, axis=0)
                # pt = np.random.uniform(low=min_, high=max_)
                # tmp = point - np.dot(point - pt, projection) * projection
                tmp = point - np.dot(point, projection) * projection
                res.append(tmp)

            yield np.array(res)

    def grow_trees(self, points):
        for point in points:
            yield IsolationForest(n_estimators=1,
                                  max_samples=point.shape[0]).fit(point)

    def fit(self, points):
        self.projections = list(self.generate_random_vectors(points))
        positions = list(self.point_positions(points))
        self.trees = list(self.grow_trees(positions))
        return self

    def get_depths(self, points):
        for point, tree in zip(points, self.trees):
            return tree.decision_function(points)

    # def decision_function(self, points):
    #     positions = list(self.point_positions(points))
    #     # scores = np.zeros(points.shape[0])
    #     scores = []

    #     for position, tree in zip(positions, self.trees):
    #         scores.append(tree.decision_function(position))
    #         # scores += tree.decision_function(position)

    #     # scores /= scores.shape[0]
    #     # scores = 1 - scores
    #     # depths = np.array(depths)
    #     # print(depths.shape)
    #     # print(np.mean(depths, axis=1).shape)

    #     # depths = np.array([
    #     # tree.decision_function(points) for tree in self.trees])
    #     # mean_depths = np.mean(depths, axis=0)
    #     # Normalize the points
    #     # scores = 2**-(mean_depths / self.average_path_length(points.shape[0]))
    #     return 0.5 - np.mean(scores, axis=0)
    #     # return scores

    def decision_function(self, points):
        # code structure from ForestClassifier/predict_proba
        # Check data
        positions = list(self.point_positions(points))
        # X = self.trees[0].estimators_[0]._validate_X_predict(
        #     positions, check_input=True)
        X = np.array(positions)

        n_samples = points.shape[0]

        # There is a bug here that I need to fix
        depths = np.zeros((n_samples, self.n_estimators), order="f")

        for point, estimator in zip(X, self.trees):
            depths += self._decision_function(estimator, point)

        self.max_samples_ = self.trees[0].max_samples_

        scores = 2 ** (
            -depths.mean(axis=1) / _average_path_length(self.max_samples_))

        # Take the opposite of the scores as bigger is better (here less
        # abnormal) and add 0.5 (this value plays a special role as described
        # in the original paper) to give a sense to scores = 0:
        return 0.5 - scores

    def _decision_function(self, estimator, X):
        n_samples = X.shape[0]
        n_samples_leaf = np.zeros((n_samples, self.n_estimators), order="f")
        depths = np.zeros((n_samples, self.n_estimators), order="f")

        for i, tree in enumerate(estimator.estimators_):
            leaves_index = tree.apply(X)
            node_indicator = tree.decision_path(X)
            n_samples_leaf[:, i] = tree.tree_.n_node_samples[leaves_index]
            depths[:, i] = np.asarray(
                node_indicator.sum(axis=1)).reshape(-1) - 1

        depths += _average_path_length(n_samples_leaf)
        return depths
