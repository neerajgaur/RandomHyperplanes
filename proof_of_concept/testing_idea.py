""" Proof of concept for the random-cut-hyperplanes idea """
import sys
import numpy as np
from scipy.stats import scoreatpercentile
from sklearn.metrics import confusion_matrix

N_ESTIMATORS = 100
SCORE_AT = 2.5

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
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators

    def fit(self, X):
        self.trees = list(self._fit(X))
        return self

    def _fit(self, X):
        for i in range(self.n_estimators):
            yield IsolationTree(X)

    def decision_function(self, X):
        depths = np.array([tree.decision_function(X) for tree in self.trees])
        mean_depths = np.mean(depths, axis=0)

        # Normalize the points
        scores = 0.5 - 2 ** -1.0 * (mean_depths / average_path_length(X.shape[0]))

        return scores

    def get_depths(self, points):
        depths = np.array([tree.decision_function(X) for tree in self.trees])
        mean_depths = np.mean(depths, axis=0)
        return mean_depths


class IsolationTree(object):
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
        node, points_left, points_right = self.get_split(points)

        if not node or depth >= 50:
            return self

        self.node = node
        self.child_left = IsolationTree(points_left, depth=depth + 1)
        self.child_right = IsolationTree(points_right, depth=depth + 1)

    def decision_function(self, points):
        return np.array([self.get_depth(point) for point in points])

    def get_depth(self, point):
        if self.is_leaf:
            # return self.num_points + average_path_length(self.num_points)
            if self.num_points == 0:
                print("Empty array uh oh :(")
                self.num_points = 1 # Dirty bug fix

            return self.num_points + harmonic_approx(self.num_points)
        else:
            if self.node.is_point_left(point):
                return 1 + self.child_left.get_depth(point)
            else:
                return 1 + self.child_right.get_depth(point)

    def get_split(self, points):
        if points.shape[0] <= self.group_threshold:
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


class RandomHyperplanes(object):
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators

    def fit(self, X):
        self.planes = list(self._fit(X))
        return self

    def _fit(self, X):
        for i in range(self.n_estimators):
            yield HyperplaneCollection(X)

    def decision_function(self, X):
        depths = np.array([plane.decision_function(X) for plane in self.planes])
        mean_depths = np.mean(depths, axis=0)

        # Normalize the points
        scores = 0.5 - 2 ** -1.0 * (mean_depths / average_path_length(X.shape[0]))

        return scores

    def get_depths(self, points):
        depths = np.array([plane.decision_function(X) for plane in self.planes])
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
            # return self.num_points + average_path_length(self.num_points)
            return self.num_points + harmonic_approx(self.num_points)
        else:
            if self.splitting_plane.point_relative_to_plane(point) < 0:
                return 1 + self.child_left.get_depth(point)
            else:
                return 1 + self.child_right.get_depth(point)

    def get_split(self, points):
        if points.shape[0] <= self.group_threshold:
            return (None, None, None)

        splitting_plane = generate_splitting_plane(points)
        positions = splitting_plane.position_of_points(points)

        points_left = points[np.where(positions < 0)]
        points_right = points[np.where(positions > 0)]

        return (splitting_plane, points_left, points_right)


class Hyperplane(object):
    def __init__(self, origin, normal):
        self.origin = origin
        self.normal = normal

    def point_relative_to_plane(self, point):
        return np.dot(self.normal, point - self.origin)

    def position_of_points(self, points):
        position = np.array([self.point_relative_to_plane(point) for point in points])
        return position


def sample_feature(point):
    length = point.shape[-1]
    return np.random.randint(low=0, high=length)

def sample_split_threshold(points, feature):
    return list(generate_point(points))[feature]

def average_path_length(n):
    return 2 * harmonic_approx(n - 1) - (2 * (n - 1) / n)

def harmonic_approx(n):
    return np.log(n) + 0.5772156649

def generate_splitting_plane(points):
    # Generate n points for each feature range
    # Use those as the coefs of a normal
    # Split on those points

    feature_ranges = get_feature_ranges(points)
    origin = np.zeros(shape=points.shape[-1])

    normal = np.fromiter(generate_point(points), dtype=float)
    normal -= origin

    return Hyperplane(origin=origin, normal=normal)


def generate_point(points):
    """ Generat an n-dimensional normal vector

    For now just do so by sampling the points from a uniform distribution.
    """
    feature_mins_maxes = get_feature_ranges(points)
    for min_, max_, in get_feature_ranges(points):
        yield np.random.uniform(low=min_, high=max_)

def get_feature_ranges(points):
    """ Yields the min and max of each feature in a vector. """
    points_transpose = np.transpose(points)
    for point in points_transpose:
        yield (np.min(point), np.max(point))

def randomly_shuffle_points(points):
    points_prime = points
    np.random.shuffle(points_prime)
    points_prime = points_prime[0:points_prime.shape[-1]] # Make it square
    return points_prime

def calculate_normal(points, origin):
    points_inv = np.linalg.inv(points)
    solution_vec = np.ones(points_inv.shape[0])
    # while np.all(solution_vec > 0) or \
        #         np.all(solution_vec < 0) or \
        #         np.any(solution_vec == 0):
    solution_vec = np.random.uniform(
        low=-1.0, high=1.0, size=(points_inv.shape[0], ))

    normal = np.matmul(points_inv, solution_vec)
    return normal


def run_plane_simul(X, y):
    print("Beginning plane fit...")
    rhp = RandomHyperplanes(n_estimators=N_ESTIMATORS)
    rhp = rhp.fit(X)
    print("done fitting")

    scores = rhp.decision_function(X)
    threshold = scoreatpercentile(scores, SCORE_AT)
    anomalies = scores <= threshold
    y_pred = np.zeros(shape=anomalies.shape)
    y_pred[anomalies] = 1

    correct_guesses = np.count_nonzero(y[np.where(scores <= threshold)])
    incorrect_guesses = y[np.where(scores <= threshold)].shape[0] - \
        correct_guesses

    print("Correct guesses:", correct_guesses)
    print("Incorrect guesses:", incorrect_guesses)
    print("Expected", np.count_nonzero(y), "anomalies")

    cnf_matrix = confusion_matrix(y, y_pred)

    tn, fp, fn, tp = cnf_matrix.ravel()
    print(f"tp: {tp} \ntn: {tn} \nfp: {fp} \nfn: {fn}")

    cnf_matrix = cnf_matrix.astype('float') / \
        cnf_matrix.sum(axis=1)[:, np.newaxis]

    tn, fp, fn, tp = cnf_matrix.ravel()
    print(f"Normalized \ntp: {tp} \ntn: {tn} \nfp: {fp} \nfn: {fn}")

    print(cnf_matrix)

    depths = rhp.get_depths(X)
    anomalous_depths = depths[anomalies]
    print("Average anomalous depth:", np.mean(anomalous_depths))

    non_anomalous_depths = depths[np.logical_not(anomalies)]
    print("Average non-anomalous depth:", np.mean(non_anomalous_depths))


def run_iforest_simul(X, y):
    print("Beginning iforest fit...")
    iforest = IsolationForest(n_estimators=N_ESTIMATORS)
    iforest = iforest.fit(X)
    print("done fitting")

    scores = iforest.decision_function(X)
    threshold = scoreatpercentile(scores, SCORE_AT)
    anomalies = scores <= threshold
    y_pred = np.zeros(shape=anomalies.shape)
    y_pred[anomalies] = 1

    correct_guesses = np.count_nonzero(y[np.where(scores <= threshold)])
    incorrect_guesses = y[np.where(scores <= threshold)].shape[0] - \
        correct_guesses

    print("iforest Correct guesses:", correct_guesses)
    print("iforest Incorrect guesses:", incorrect_guesses)
    print("Expected", np.count_nonzero(y), "anomalies")

    iforest_cnf_matrix = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = iforest_cnf_matrix.ravel()
    print(f"tp: {tp} \ntn: {tn} \nfp: {fp} \nfn: {fn}")

    iforest_cnf_matrix = iforest_cnf_matrix.astype('float') / \
            iforest_cnf_matrix.sum(axis=1)[:, np.newaxis]

    print(iforest_cnf_matrix)

    tn, fp, fn, tp = iforest_cnf_matrix.ravel()
    print(f"Normalized \ntp: {tp} \ntn: {tn} \nfp: {fp} \nfn: {fn}")

    depths = iforest.get_depths(X)
    anomalous_depths = depths[anomalies]
    print("Average anomalous depth:", np.mean(anomalous_depths))

    non_anomalous_depths = depths[np.logical_not(anomalies)]
    print("Average non-anomalous depth:", np.mean(non_anomalous_depths))


def _gen_hard_data(n, p, infection_pct, variance=10.0, mu=5.0):
    X = np.random.randn(n * p).reshape(n, p)

    # hard data
    # Weight it to the number of features
    is_anomaly = np.random.rand(n, p) < (infection_pct / p)
    X[is_anomaly] = variance * np.random.randn() + mu

    y = np.zeros(shape=(n,))

    tmp = np.array([np.any(r) for r in is_anomaly])
    y[tmp] = 1.0

    return (X, y)

def _gen_easy_data(n, p, infection_pct, variance=10.0, mu=5.0):
    X = np.random.randn(n * p).reshape(n, p)

    # hard data
    # Weight it to the number of features
    is_anomaly = np.random.rand(n) < infection_pct
    X[is_anomaly] = variance * np.random.randn() + mu

    y = np.zeros(shape=(n,))
    y[is_anomaly] = 1.0

    return (X, y)

if __name__ == "__main__":
    n = 5000 # number of entries
    p = 300   # features

    infection_pct = 0.05
    X, y = _gen_easy_data(n, p, infection_pct)

    run_plane_simul(X, y)
    print("\nDone plane simul-----\n")
    run_iforest_simul(X, y)
