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
                                  contamination=0.05,
                                  variance=0.01 ** 2.0,
                                  mean=0):
    """ Samples from two gaussians with mean +/- 5 as the first feature. Then
    the other features are set to zero (to be irrelevant). Gaussian noise is
    added to every feature and a percentage of these points are then centered
    about zero. This creates two clusters with a central anomalous cluster in
    the middle.
    """
    size = int(num_points / 2)

    mean_one = np.zeros(shape=(num_features,))
    mean_two = np.zeros(shape=(num_features,))
    mean_one[0] = mean
    mean_two[0] = -mean

    means = [mean_one, mean_two]
    n_points = [size, size]

    cov = variance * np.identity(n=num_features)
    covs = [cov, cov]

    n_anomalies = int(num_points * contamination)

    if n_anomalies == 0:
        n_anomalies = 1

    anon_cov = [cov]
    anon_mean = [np.zeros(shape=(num_features))]

    X = gen_correlated_clusters(n_points, means, covs)
    X_anon = gen_correlated_clusters([n_anomalies], anon_mean, anon_cov)

    X += variance * np.random.randn(num_points, num_features)
    X_anon += variance * np.random.randn(n_anomalies, num_features)

    return (X, X_anon)


def gen_correlated_clusters(n_points, means, covs):
    X = np.array(
        [correlated_data(n, c, m) for n, m, c in zip(n_points, means, covs)])

    try:
        t_points = n_points.sum()
    except AttributeError:
        # Cast to numpy array and sum that
        t_points = np.array(n_points).sum()

    return X.ravel().reshape(t_points, -1)


def correlated_data(num_points,
                    cov,
                    mean):
    return np.random.multivariate_normal(mean=mean, cov=cov, size=(num_points,))
