""" Proof of concept for the random-cut-hyperplanes idea """
import sys
import numpy as np
from scipy.stats import scoreatpercentile
from sklearn.metrics import confusion_matrix

def get_feature_ranges(points):
    """ Yields the min and max of each feature in a vector. """
    points_transpose = np.transpose(points)
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
    def __init__(self, n_estimators=10, method=None, max_depth=50):
        self.method = method
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self, points):
        self.planes = []
        for i in range(self.n_estimators):
            self.planes.append(HyperplaneCollection(points, self.method, max_depth=self.max_depth))
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

    def predict(self, X, score_at=97.5):
        scores = self.decision_function(X)
        preds = np.zeros(shape=(scores.shape[0]))
        threshold = scoreatpercentile(scores, score_at)
        preds[np.where(scores >= threshold)] = 1
        return preds


class HyperplaneCollection(object):
    def __init__(self, points, method=None, group_threshold=15, depth=1, max_depth=50):
        self.method = method
        self.child_left = self.child_right = None
        self.points = points
        self.group_threshold = group_threshold
        self.num_points = self.points.shape[0]
        self.max_depth = max_depth
        self.split(self.points, depth)

    def __str__(self):
        return f"<{self.__class__.__name__} num_points:{self.num_points}>"

    @property
    def is_leaf(self):
        return self.child_left == None and self.child_right == None

    def split(self, points, depth=1):
        plane, points_left, points_right = self.get_split(points)
        if not plane or depth >= self.max_depth:
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

        if not self.method:
            splitting_plane = generate_splitting_plane(points)
            positions = splitting_plane.position_of_points(points)

            split_point = np.random.uniform(np.min(positions), np.max(positions))

            points_left = points[np.where(positions < split_point)[0], :]
            points_right = points[np.where(positions >= split_point)[0], :]
            self.split_point = split_point
        else:
            splitting_plane = generate_splitting_plane_uniform(points)
            positions = splitting_plane.position_of_points(points)

            split_point = 0

            points_left = points[np.where(positions < split_point)]
            points_right = points[np.where(positions >= split_point)]
            self.split_point = split_point

        return (splitting_plane, points_left, points_right)


class Hyperplane(object):
    def __init__(self, line):
        # self.origin = origin
        # self.normal = normal
        self.line = line

    def point_relative_to_plane(self, point):
        result = np.dot(self.line, point)
        # result = np.dot(self.normal, point - self.origin) # - np.dot(self.normal, self.origin)
        #result = np.dot(self.normal, point - self.origin)
        return result

    def foo_bar(self, point):
        result = np.dot(self.normal, point)
        return result

    def point_relative_to_plane_x(self, point):
        result = np.dot(self.normal, point)
        return result

    def position_of_points(self, points):
        # offset = np.dot(self.normal, self.origin)
        position = np.array([self.point_relative_to_plane(point) for point in points])
        # position_x = np.array([self.point_relative_to_plane_x(point) for point in points])
        #print(position - position_x)
        return position

class HyperplaneUniform(object):
    def __init__(self, normal, origin):
        self.origin = origin
        self.normal = normal

    def point_relative_to_plane(self, point):
        result = np.dot(self.normal, point - self.origin)
        return result

    def position_of_points(self, points):
        position = np.array([self.point_relative_to_plane(point) for point in points])
        return position


def generate_splitting_plane(points):
    p = points.shape[1]
    line = np.random.randn(1, p)
    line /= np.linalg.norm(line)

    return Hyperplane(line=line)

def generate_splitting_plane_uniform(points):
    feature_ranges = get_feature_ranges(points)
    origin = np.fromiter(generate_point(points), dtype=float)
    normal = np.fromiter(generate_point(points), dtype=float)
    normal -= origin
    return HyperplaneUniform(origin=origin, normal=normal)


def generate_point(points):
    """ Generat an n-dimensional normal vector

    For now just do so by sampling the points from a uniform distribution.
    """
    feature_mins_maxes = get_feature_ranges(points)
    for min_, max_, in get_feature_ranges(points):
        yield np.random.uniform(low=min_, high=max_)


def generate_point_uniform(points):
    """ Generat an n-dimensional normal vector

    For now just do so by sampling the points from a uniform distribution.
    """
    feature_mins_maxes = get_feature_ranges(points)
    for min_, max_, in get_feature_ranges(points):
        yield np.random.uniform(low=min_, high=max_)
