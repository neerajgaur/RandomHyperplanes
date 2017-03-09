""" Functions that generate synthetic data using numpy """
import numpy as np


def cluster_with_anomalous_points(num_points,
                                  num_features,
                                  percent_anomalies=0.05,
                                  variance=10.0,
                                  mean=5.0):
    """ Sample data from a multivariate gaussian and generate anomalous points
    outside of that data range with mean `mean` and variance `variance`
    """
    X = np.random.randn(num_points, num_features)
    is_anomaly = np.random.choice(
        num_points,
        size=int(percent_anomalies * num_points),
        replace=False)

    X[is_anomaly] = mean + variance * np.random.randn(
        is_anomaly.shape[0],
        num_features)

    y = np.zeros(shape=(num_points,))
    y[is_anomaly] = 1.0

    return (X, y)


def two_clusters_anomalous_middle(num_points,
                                  num_features,
                                  percent_anomalies=0.05,
                                  variance=0.01 ** 2.0,
                                  mean=0):
    """ Samples from two gaussians with mean +/- 5 as the first feature. Then
    the other features are set to zero (to be irrelevant). Gaussian noise is
    added to every feature and a percentage of these points are then centered
    about zero. This creates two clusters with a central anomalous cluster in
    the middle.
    """
    X = np.zeros(shape=(num_points, num_features))
    for row in X:
        if np.random.rand() < 0.5:
            row[0] = 5
        else:
            row[0] = -5

    X += variance * np.random.randn(num_points, num_features)
    is_anomaly = np.random.choice(num_points, size=10, replace=False)
    X[is_anomaly] = variance * np.random.randn(
        is_anomaly.shape[0], num_features)
    y = np.zeros(shape=(num_points,))
    y[is_anomaly] = 1.0

    return (X, y)
