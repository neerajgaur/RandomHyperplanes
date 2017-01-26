""" Using RandomHyperplanes as a splitting agent for points. """
import collections
import numpy as np

class RandomHyperplanes(object):
    """ Generates random hyperplanes used to split points. """
    def __init__(self, n_estimators=1):
        self._hyperplanes = []
        self._n_estimators = n_estimators
        pass

    def fit(self, X):
        # self._hyperplanes = [self._gen_hyperplane(X) for _ in range(self._n_estimators)]
        return self

    def predict(self, X):
        pass


    def _split(self, X):
        plane, left, right = self._split_points(X)
        to_split = collections.deque([(plane, left, right)])

        while to_split:
            plane, left, right = to_split.pop()


    def _split_points(self, points):
        splitting_plane = Hyperplane(points)
        points_left = np.where(points <= splitting_plane._coords)
        points_right = np.where(points > splitting_plane._coords)
        return (splitting_plane, points_left, points_right)


class Hyperplane(object):
    """ A hyperplane. """
    def __init__(self, points):
        self._coords = self._generate(points)

    def _generate(self, points):
        """ Given a vector of points of size n generate a series of hyperplanes
        based on these points in a random manner.

        For now just do so by sampling the points from a multivariate Gaussian.
        """
        feature_mins_maxes = list(self._feature_ranges(points))
        hyperplane = np.array([np.random.uniform(low=min_, high=max_) for min_, max_ in feature_mins_maxes])
        return hyperplane

    def _feature_ranges(self, points):
        """ Yields the min and max of each feature in a vector. """
        points_transpose = np.transpose(points)
        for point in points_transpose:
            yield (np.min(point), np.max(point))
