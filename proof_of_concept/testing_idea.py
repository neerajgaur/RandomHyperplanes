""" Proof of concept for the random-cut-hyperplanes idea """
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

if __name__ == "__main__":
    n = 10000 # number of entries
    p = 200  # features
    infection_pct = 0.05
    num_anomalies = int(n * infection_pct)
    X = np.random.randn(n * p).reshape(n, p)
    is_anomaly = [np.random.randint(100) < 5 for _ in range(n)]
    X[np.where(is_anomaly)] = 10 * np.random.randn(1, p)
    y = np.zeros(shape=(n,))
    y[np.where(is_anomaly)] = 1.0

    print("Beginning fit...")
    rhp = RandomHyperplanes(n_estimators=5)
    rhp = rhp.fit(X)
    print("done fitting")
    scores = rhp.decision_function(X)
    threshold = scoreatpercentile(scores, infection_pct)
    anomalies = scores <= threshold
    y_pred = np.zeros(shape=anomalies.shape)
    y_pred[anomalies] = 1


    correct_guesses = np.count_nonzero(y[np.where(scores <= threshold)])
    incorrect_guesses = y[np.where(scores <= threshold)].shape[0] - correct_guesses

    print("Correct guesses:", correct_guesses)
    print("Incorrect guesses:", incorrect_guesses)
    print("Expected", np.count_nonzero(y), "anomalies")
    cnf_matrix = confusion_matrix(y, y_pred)
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

    print(cnf_matrix)





