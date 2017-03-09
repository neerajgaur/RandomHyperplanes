import numpy as np
from projections import RandomProjectionForest

X = np.array([[1, 1, 1], [-100, -10000, -10000], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
forest = RandomProjectionForest(n_estimators=10, method='rcf')

forest = forest.fit(X)
print(forest.decision_function(X))
print(forest.estimators)
