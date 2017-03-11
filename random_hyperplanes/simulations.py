""" Store the code to get simulation data here and any plotting code. """
import numpy as np
import matplotlib.pyplot as plt
from random_hyperplanes import iforest, projections, planes, synthetic_data

CLASSIFIERS = (
    projections.RandomProjectionForest,
    planes.RandomProjectionForestOld,
    iforest.IsolationForest,
)

FIG_SIZE = (12, 12)

X_LIM = None
Y_LIM = None


def two_cluster_simulation_data(num_points,
                                num_features,
                                contamination=0.05,
                                variance=0.01,
                                mean=5.0,
                                scale='small'):

    X, X_anon = synthetic_data.two_clusters_anomalous_middle(
        num_points, num_features, contamination, variance, mean)

    if scale == 'large':
        n_anomalies = int(contamination * num_points)
        scaling = 2 ** np.random.randint(low=10, high=15)
        X[:, 1:] *= scaling
        X_anon[:, 1:] *= scaling

    return (X, X_anon)


def correlated_simulation_data(n_points,
                               means,
                               covs,
                               anon_n_points,
                               anon_means,
                               anon_covs):

    X = synthetic_data.gen_correlated_clusters(
        n_points=n_points, covs=covs, means=means)
    X_anon = synthetic_data.gen_correlated_clusters(
        n_points=anon_n_points, means=anon_means, covs=anon_covs)

    return (X, X_anon)


def run_comparison_simul(X,
                         X_anon,
                         classifiers=CLASSIFIERS,
                         n_estimators=100,
                         score_at=97.5):

    X_all = np.row_stack([X, X_anon])
    np.random.shuffle(X_all)

    for i, clf in enumerate(classifiers):
        print(f'Running iteration {i + 1}')
        run_simul(
            X=X,
            X_anon=X_anon,
            X_all=X_all,
            model=clf,
            n_estimators=n_estimators,
            score_at=score_at
        )

    print('Done!')

def run_simul(X,
              X_anon,
              X_all,
              model,
              n_estimators,
              score_at):

    # Run the simulations on all data first
    clf_ifo_all = model(n_estimators=n_estimators, method='iforest').fit(X_all)
    clf_rcf_all = model(n_estimators=n_estimators, method='rcf').fit(X_all)

    # Rerun while holding out anomalous points during fitting
    clf_ifo_hld = model(n_estimators=n_estimators, method='iforest').fit(X)
    clf_rcf_hld = model(n_estimators=n_estimators, method='rcf').fit(X)

    y_all_pred_ifo = clf_ifo_all.predict(X_all, score_at=score_at)
    y_all_pred_rcf = clf_rcf_all.predict(X_all, score_at=score_at)

    y_hld_pred_ifo = clf_ifo_hld.predict(X_all, score_at=score_at)
    y_hld_pred_rcf = clf_rcf_hld.predict(X_all, score_at=score_at)

    name_vals = [
        ('Non-proportional Weighting - All Data',                 y_all_pred_ifo, clf_ifo_all),
        ('Proportional Weighting - All Data',                     y_all_pred_rcf, clf_rcf_all),
        ('Non-proportional Weighting - No Anomalies in Training', y_hld_pred_ifo, clf_ifo_hld),
        ('Proportional Weighting - No Anomalies in Training',     y_hld_pred_rcf, clf_rcf_hld),
    ]

    xlim = [np.min(X_all[:, 0]) - 1.0, np.max(X_all[:, 0]) + 1.0]
    ylim = [np.min(X_all[:, 1]) - 1.0, np.max(X_all[:, 1]) + 1.0]

    xx, yy = np.meshgrid(
        np.linspace(
            int(np.min(X[:, 0])) - 5.0,
            int(np.max(X[:, 0])) + 5.0, 100),
        np.linspace(
            int(np.min(X[:, 1])) - 5.0,
            int(np.max(X[:, 1])) + 5.0, 100))

    plt.figure(figsize=FIG_SIZE)

    for i, v in enumerate(name_vals):
        sub_title, y, clf = v
        subplt = int('22'+str(i+1))
        plt.subplot(subplt)
        plt.title(sub_title)

        X_non_pred = X_all[y == 0]
        X_ano_pred = X_all[y == 1]

        Z = 1 - clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

        plt.scatter(
            X_non_pred[:, 0], X_non_pred[:, 1], c='b', alpha=0.2, marker='o')
        plt.scatter(
            X_ano_pred[:, 0], X_ano_pred[:, 1], c='r', alpha=0.5, marker='x')

        if X_LIM:
            plt.xlim(X_LIM)
        else:
            plt.xlim(xlim)

        if Y_LIM:
            plt.ylim(Y_LIM)
        else:
            plt.ylim(ylim)

    plt.suptitle(str(clf), fontsize=16)
    plt.show()
