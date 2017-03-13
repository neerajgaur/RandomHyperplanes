import numpy as np
from random_hyperplanes.planes import RandomProjectionForestOld
from random_hyperplanes.projections import RandomProjectionForest

X = np.array([[1, 1, 1], [10_000_000, 10_000_000, 10_000_000], [-10_000_000, -10_000_000, -10_000_000], [-100, -10000, -10000], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
forest = RandomProjectionForestOld(n_estimators=100)
# forest2 = RandomProjectionForest(n_estimators=100)

forest = forest.fit(X)
print(X)
print('Old', forest.decision_function(X))
print('Old', forest.predict(X, score_at=80.0))
print('Old', forest.get_depths(X))

# forest2 = forest2.fit(X)
# print('New',forest2.decision_function(X))
# print(forest.estimators)
