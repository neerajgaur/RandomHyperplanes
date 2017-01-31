""" Proof of concept for the random-cut-hyperplanes idea """
import numpy as np
from scipy.stats import scoreatpercentile

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
        for i in range(self.n_estimators):
            if i % 10 == 0:
                print(i)

        self.planes = list(self._fit(X))
        return self

    def _fit(self, X):
        for i in range(self.n_estimators):
            if i % 10 == 0:
                print(i)

            yield HyperplaneCollection(X)


    def decision_function(self, X):
        depths = np.array([plane.decision_function(X) for plane in self.planes])
        mean_depths = np.mean(depths, axis=0)

        # Normalize the points
        scores = 0.5 - 2 ** -1.0 * (mean_depths / average_path_length(X.shape[0]))

        return scores


class HyperplaneCollection(object):
    def __init__(self, points):
        self.child_left = self.child_right = None
        self.points = points
        self.num_points = self.points.shape[0]
        self.split(self.points)

    def __str__(self):
        return f"<{self.__class__.__name__} num_points:{self.num_points}>"

    @property
    def is_leaf(self):
        return self.child_left == None and self.child_right == None

    def split(self, points):
        if points.shape[0] == 1:
            # done splitting
            return self

        else:
            self.splitting_plane = generate_splitting_plane(points)
            positions = self.splitting_plane.position_of_points(points)

            points_left = points[np.where(positions < 0)]
            points_right = points[np.where(positions > 0)]

            while points_left.shape[0] == 0 or points_right.shape[0] == 0:
                self.splitting_plane = generate_splitting_plane(points)
                positions = self.splitting_plane.position_of_points(points)

                points_left = points[np.where(positions < 0)]
                points_right = points[np.where(positions > 0)]

            self.child_left = HyperplaneCollection(points_left)
            self.child_right = HyperplaneCollection(points_right)

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


class Hyperplane(object):
    def __init__(self, z, plane, beta):
        self.z = z
        self.plane = plane
        self.beta = beta

    def point_relative_to_plane(self, point):
        return np.dot(np.transpose(self.plane), point) + self.beta

    def position_of_points(self, points):
        position = np.array([self.point_relative_to_plane(point) for point in points])
        return position

def average_path_length(n):
    return 2 * harmonic_approx(n - 1) - (2 * (n - 1) / n)

def harmonic_approx(n):
    return np.log(n) + 0.5772156649

def generate_splitting_plane(points):
    p_1 = 0
    p_2 = 0

    p = points[np.random.randint(points.shape[0], size=2), :]
    p_1 = p[0]
    p_2 = p[1]

    alpha = np.random.uniform()
    z = alpha * p_1 + (1.0 - alpha) * p_2

    plane = np.random.randn(z.shape[0]).reshape(z.shape)
    beta = -np.matmul(np.transpose(plane), z)

    return Hyperplane(z=z, plane=plane, beta=beta)


def generate_point(points):
    """ Generat an n-dimensional normal vector

    For now just do so by sampling the points from a uniform distribution.
    """
    feature_mins_maxes = list(get_feature_ranges(points))
    normal = \
        np.array(
            [np.random.uniform(
                low=min_, high=max_
            ) for min_, max_ in feature_mins_maxes])
    return normal


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
    n = 100 # number of entries
    p = 20  # features
    infection_pct = 0.05
    num_anomalies = int(n * infection_pct)
    X = np.random.randn(n * p).reshape(n, p)
    is_anomaly = [np.random.randint(100) < 5 for _ in range(n)]
    X[np.where(is_anomaly)] = 10 * np.random.randn(1, p) + 5.0
    y = np.zeros(shape=(n,))
    y[np.where(is_anomaly)] = 1.0

    rhp = RandomHyperplanes().fit(X)
    print("done fitting")
    scores = rhp.decision_function(X)
    threshold = scoreatpercentile(scores, 5.0)
    print(threshold)

    print(np.count_nonzero(scores < threshold))
    print(np.count_nonzero(y))

