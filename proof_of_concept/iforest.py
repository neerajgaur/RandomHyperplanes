import sys
import numpy as np
from scipy.stats import scoreatpercentile
from sklearn.metrics import confusion_matrix

 # sys.setrecursionlimit(50000)

# Steps for the algorithm
#
# Given a dataset, X:
#   1. Get the feature ranges for each feature within X.
#   2. Sample a random point between each feature.
#   3. Create a hyperplane using each of these points.
#   4. Get the normal to that plane.
#   5. Determine which side of the plane the point lies on.
#   6. Repeat for the two sides.
#   7. If an individual point is left, stop iteration
#
class IsolationForest(object):
    def __init__(self, n_estimators=10, max_depth=50):
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self, points):
        self.trees = []
        for i in range(self.n_estimators):
            self.trees.append(IsolationTree(points, max_depth=self.max_depth))
            return self

    def decision_function(self, points):
        depths = np.array([tree.decision_function(points) for tree in self.trees])
        mean_depths = np.mean(depths, axis=0)
        # Normalize the points
        scores = 2**-(mean_depths / average_path_length(points.shape[0]))
        return scores

    def get_depths(self, points):
        depths = np.array([tree.decision_function(points) for tree in self.trees])
        mean_depths = np.mean(depths, axis=0)
        return mean_depths

    def predict(self, X, score_at=97.5):
        scores = self.decision_function(X)
        preds = np.zeros(shape=(scores.shape[0]))
        threshold = scoreatpercentile(scores, score_at)
        preds[np.where(scores >= threshold)] = 1
        return preds


class IsolationTree(object):
    def __init__(self, points, group_threshold=15, depth=1, max_depth=50):
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
        node, points_left, points_right = self.get_split(points)

        if not node or depth >= self.max_depth:
            return self

        self.node = node
        self.child_left = IsolationTree(points_left, depth=depth + 1)
        self.child_right = IsolationTree(points_right, depth=depth + 1)

    def decision_function(self, points):
        return np.array([self.get_depth(point) for point in points])

    def get_depth(self, point):
        if self.is_leaf:
            return 1
        else:
            if self.node.is_point_left(point):
                return 1 + self.child_left.get_depth(point)
            else:
                return 1 + self.child_right.get_depth(point)

    def get_split(self, points):
        if self.num_points <2:
            return (None, None, None)
        split_feature = sample_feature(points)
        split_threshold = sample_split_threshold(points, split_feature)
        node = Node(split_threshold, split_feature)

        positions = node.position_of_points(points)

        points_left = points[positions]
        points_right = points[np.logical_not(positions)]

        return (node, points_left, points_right)


class Node(object):
    def __init__(self, threshold, feature):
        self.threshold = threshold
        self.feature = feature

    def is_point_left(self, point):
        return point[self.feature] < self.threshold

    def position_of_points(self, points):
        return np.array([self.is_point_left(point) for point in points])

def get_feature_ranges(points):
    """ Yields the min and max of each feature in a vector. """
    points_transpose = np.transpose(points)
    for point in points_transpose:
        yield (np.min(point), np.max(point))

def generate_point(points):
    """ Generat an n-dimensional normal vector

    For now just do so by sampling the points from a uniform distribution.
    """
    feature_mins_maxes = get_feature_ranges(points)
    for min_, max_, in get_feature_ranges(points):
        yield np.random.uniform(low=min_, high=max_)

def sample_feature(point):
    length = point.shape[-1]
    return np.random.randint(low=0, high=length)

def sample_split_threshold(points, feature):
    return list(generate_point(points))[feature]

def average_path_length(n):
    return 2 * harmonic_approx(n - 1) - (2 * (n - 1) / n)

def harmonic_approx(n):
    return np.log(n) + 0.5772156649
