""" Proof of concept for the random-cut-hyperplanes idea """
import numpy as np
# from scipy.stats import scoreatpercentile
# from sklearn.metrics import confusion_matrix
from planes import RandomProjectionForest
from iforest import IsolationForest


def gen_hard_data(n, p, infection_pct, variance=10.0, mu=5.0):
    X = np.random.randn(n, p)

    # hard data
    # Weight it to the number of features
    is_anomaly = np.random.rand(n, p) < (infection_pct / p)
    X[is_anomaly] = variance * np.random.randn() + mu

    y = np.zeros(shape=(n,))

    tmp = np.array([np.any(r) for r in is_anomaly])
    y[tmp] = 1.0

    return (X, y)


def gen_easy_data(n, p, infection_pct, variance=10.0, mu=5.0):
    X = np.random.randn(n, p)
    is_anomaly = np.random.choice(n, size=int(infection_pct*n), replace=False)
    X[is_anomaly] = variance * np.random.randn(is_anomaly.shape[0], p) + mu
    y = np.zeros(shape=(n,))
    y[is_anomaly] = 1.0
    return (X, y)

def gen_two_clusters(n, p, infection_pct, variance=0.01, mu=0):
    X = np.zeros(shape=(n, p))
    for row in X:
        if np.random.rand() < 0.5:
            row[0] = 5
        else:
            row[0] = -5

    X +=  variance * np.random.randn(n, p)
    is_anomaly = np.random.choice(n, size=10, replace=False)
    X[is_anomaly] = variance * np.random.randn(is_anomaly.shape[0], p)
    y = np.zeros(shape=(n,))
    y[is_anomaly] = 1.0

    return (X, y)

def run_iforest_simul(points, y, n_estimators, method='iforest'):
    # print("Beginning iforest fit...")
    iforest = IsolationForest(n_estimators=n_estimators, method=method)
    iforest = iforest.fit(points)
    # print("done fitting")

#     scores = iforest.decision_function(points)
#     threshold = scoreatpercentile(scores, 100 - SCORE_AT)
#     anomalies = scores >= threshold
#     y_pred = np.zeros(shape=anomalies.s`hape)
#     y_pred[anomalies] = 1

#     """
#     correct_guesses = np.count_nonzero(y[np.where(scores <= threshold)])
#     incorrect_guesses = y[np.where(scores <= threshold)].shape[0] - \
    #         correct_guesses

#     print("iforest Correct guesses:", correct_guesses)
#     print("iforest Incorrect guesses:", incorrect_guesses)
#     print("Expected", np.count_nonzero(y), "anomalies")
#     """
#     iforest_cnf_matrix = confusion_matrix(y, y_pred)
#     """
#     tn, fp, fn, tp = iforest_cnf_matrix.ravel()
#     print(f"tp: {tp} \ntn: {tn} \nfp: {fp} \nfn: {fn}")
#     """
#     iforest_cnf_matrix = iforest_cnf_matrix.astype('float') / \
    #             iforest_cnf_matrix.sum(axis=1)[:, np.newaxis]

#     print(iforest_cnf_matrix)

#     """
#     tn, fp, fn, tp = iforest_cnf_matrix.ravel()
#     print(f"Normalized \ntp: {tp} \ntn: {tn} \nfp: {fp} \nfn: {fn}")
#     """
    depths = iforest.get_depths(points)
    anomalous_depths = depths[np.where(y==1.0)]
    non_anomalous_depths = depths[np.where(y==0.0)]
    # print("Average anomalous depth:", np.mean(anomalous_depths))
    # print("Average non-anomalous depth:", np.mean(non_anomalous_depths))
    return (None, depths, None, y, np.mean(anomalous_depths), np.mean(non_anomalous_depths))


def run_plane_simul(points, y, n_estimators):
    # print("Beginning plane fit...")
    rhp = RandomProjectionForest(n_estimators=n_estimators)
    rhp = rhp.fit(points)
    # print("done fitting")

#     scores = rhp.decision_function(points)
#     threshold = scoreatpercentile(scores, 100 - SCORE_AT)
#     anomalies = scores >= threshold
#     y_pred = np.zeros(shape=anomalies.shape)
#     y_pred[anomalies] = 1

#     """
#     correct_guesses = np.count_nonzero(y[np.where(scores <= threshold)])
#     incorrect_guesses = y[np.where(scores <= threshold)].shape[0] - \
    #         correct_guesses

#     print("Correct guesses:", correct_guesses)
#     print("Incorrect guesses:", incorrect_guesses)
#     print("Expected", np.count_nonzero(y), "anomalies")
#     """
#     cnf_matrix = confusion_matrix(y, y_pred)

#     """
#     tn, fp, fn, tp = cnf_matrix.ravel()
#     print(f"tp: {tp} \ntn: {tn} \nfp: {fp} \nfn: {fn}")
#     """
#     cnf_matrix = cnf_matrix.astype('float') / \
    #         cnf_matrix.sum(axis=1)[:, np.newaxis]

#     """
#     tn, fp, fn, tp = cnf_matrix.ravel()
#     print(f"Normalized \ntp: {tp} \ntn: {tn} \nfp: {fp} \nfn: {fn}")
#     """
#     print(cnf_matrix)

    depths = rhp.get_depths(points)
    anomalous_depths = depths[np.where(y == 1.0)]
    non_anomalous_depths = depths[np.where(y == 0.0)]
    # print("Average anomalous depth:", np.mean(anomalous_depths))
    # print("Average non-anomalous depth:", np.mean(non_anomalous_depths))
    return (None, depths, None, y, np.mean(anomalous_depths), np.mean(non_anomalous_depths))


def run_plane_simul_uniform(points, y):
    rhp = RandomHyperplanes(n_estimators=N_ESTIMATORS, method='uniform')
    rhp = rhp.fit(points)
    depths = rhp.get_depths(points)
    anomalous_depths = depths[np.where(y == 1.0)]
    non_anomalous_depths = depths[np.where(y == 0.0)]
    return (None, depths, None, y, np.mean(anomalous_depths), np.mean(non_anomalous_depths))



if __name__ == "__main__":
    N_ESTIMATORS = 1  # Get a somewhat stable approximation
    SCORE_AT = 2.5

    n = 1000  # number of entries
    p = 30  # features

    infection_pct = 0.05
    X, y = _gen_two_clusters(n, p, infection_pct)

    # scores_r, depths_r, y_pred_r, y_r, anom_r, non_anom_r = run_plane_simul(X, y)
    # print("\nDone plane simul-----\n")
    # scores_i, depths_i, y_pred_i, y_i, anom_i, non_anom_i = run_iforest_simul(X, y)

    anomalous_depths_r = list()
    non_anomalous_depths_r = list()

    # anomalous_depths_ru = list()
    # non_anomalous_depths_ru = list()

    anomalous_depths_i = list()
    non_anomalous_depths_i = list()
    for i in range(1000):
        if i % 100 == 0:
            print(i)

        scores_r, depths_r, y_pred_r, y_r, anom_r, non_anom_r = run_plane_simul(X, y)
        _, _, _, _, anom_ru, non_anom_ru = run_plane_simul_uniform(X, y)
        # print("\nDone plane simul-----\n")
        scores_i, depths_i, y_pred_i, y_i, anom_i, non_anom_i = run_iforest_simul(X, y)

        anomalous_depths_r.append(anom_r)
        non_anomalous_depths_r.append(non_anom_r)

        # anomalous_depths_ru.append(anom_ru)
        # non_anomalous_depths_ru.append(non_anom_ru)

        anomalous_depths_i.append(anom_i)
        non_anomalous_depths_i.append(non_anom_i)


    print(f"For rhp anom/non_anom: {np.mean(anomalous_depths_r)}, {np.mean(non_anomalous_depths_r)}")
    # print(f"For rhp uniform anom/non_anom: {np.mean(anomalous_depths_ru)}, {np.mean(non_anomalous_depths_ru)}")
    print(f"For iforest anom/non_anom: {np.mean(anomalous_depths_i)}, {np.mean(non_anomalous_depths_i)}")
