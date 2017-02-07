""" Proof of concept for the random-cut-hyperplanes idea """
import sys
import numpy as np
from scipy.stats import scoreatpercentile
from sklearn.metrics import confusion_matrix

def get_feature_ranges(points):
    """ Yields the min and max of each feature in a vector. """
    for point in points_transpose:
        yield (np.min(point), np.max(point))

def sample_feature(point):
    length = point.shape[-1]
    return np.random.randint(low=0, high=length)

def sample_split_threshold(points, feature):
    return list(generate_point(points))[feature]

def average_path_length(n):
    return 2 * harmonic_approx(n - 1) - (2 * (n - 1) / n)

def harmonic_approx(n):
    return np.log(n) + 0.5772156649

class RandomHyperplanes(object):
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators

    def fit(self, points):
        self.planes = []
        for i in range(self.n_estimators):
            self.planes.append(HyperplaneCollection(points))
        return self

    def decision_function(self, points):
        depths = np.array([plane.decision_function(points) for plane in self.planes])
        mean_depths = np.mean(depths, axis=0)
        # Normalize the points
        scores = 2**-(mean_depths / average_path_length(points.shape[0]))
        return scores

    def get_depths(self, points):
        depths = np.array([plane.decision_function(points) for plane in self.planes])
        mean_depths = np.mean(depths, axis=0)
        return mean_depths

class HyperplaneCollection(object):
    def __init__(self, points, group_threshold=15, depth=1):
        self.child_left = self.child_right = None
        self.points = points
        self.group_threshold = group_threshold
        self.num_points = self.points.shape[0]
        self.split(self.points, depth)

    def __str__(self):
        return f"<{self.__class__.__name__} num_points:{self.num_points}>"

    @property
    def is_leaf(self):
        return self.child_left == None and self.child_right == None

    def split(self, points, depth=1):
        plane, points_left, points_right = self.get_split(points)
        if not plane or depth >= 50:
            return self
        self.splitting_plane = plane
        self.child_left = HyperplaneCollection(points_left, depth=depth + 1)
        self.child_right = HyperplaneCollection(points_right, depth=depth + 1)

    def decision_function(self, points):
        return np.array([self.get_depth(point) for point in points])

    def get_depth(self, point):
        if self.is_leaf:
            return 1
        else:
            if self.splitting_plane.point_relative_to_plane(point) < self.split_point:
                return 1 + self.child_left.get_depth(point)
            else:
                return 1 + self.child_right.get_depth(point)

    def get_split(self, points):
        if self.num_points < 2:
            return (None, None, None)
        splitting_plane = generate_splitting_plane(points)
        positions = splitting_plane.position_of_points(points)
        split_point = np.random.uniform(np.min(positions), np.max(positions))
        points_left = points[np.where(positions < split_point)[0], :]
        points_right = points[np.where(positions >= split_point)[0], :]
        self.split_point = split_point
        return (splitting_plane, points_left, points_right)


class Hyperplane(object):
    def __init__(self, line):
        # self.origin = origin
        # self.normal = normal
        self.line = line

    def point_relative_to_plane(self, point):
        result = np.dot(self.line, point)
        return result

    def position_of_points(self, points):
        positions = np.dot(self.line, points)
        return positions

def generate_splitting_plane(points):
    p = points.shape[1]
    line = np.random.randn(1, p)
    line /= np.linalg.norm(line)

    return Hyperplane(line=line)

def generate_point(points):
    """ Generat an n-dimensional normal vector

    For now just do so by sampling the points from a uniform distribution.
    """
    feature_mins_maxes = get_feature_ranges(points)
    for min_, max_, in get_feature_ranges(points):
        yield np.random.uniform(low=min_, high=max_)
