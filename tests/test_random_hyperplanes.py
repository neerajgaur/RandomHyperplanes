""" Test the random_hyperplanes module. """
import pytest
import numpy as np
from random_hyperplanes import random_hyperplanes

class TestRandomHyperplanes(object):
    def test_fit(self):
        rhp = random_hyperplanes.RandomHyperplanes()
        X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        rhp = rhp.fit(X)
        assert isinstance(rhp, random_hyperplanes.RandomHyperplanes)


    def test_prediction(self):
        pass

    def test_hyperplane_gen(self):
        X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        hyperplane = random_hyperplanes.Hyperplane(X)
        print(hyperplane._coords)
        assert hyperplane._coords.shape[0] == X.shape[-1]


    def test_feature_ranges(self):
        X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        plane = random_hyperplanes.Hyperplane(X)
        expected_result = (1, 3) # min, max for all
        for ranges in plane._feature_ranges(X):
            assert np.all(ranges == expected_result)
